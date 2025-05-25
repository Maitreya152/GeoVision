import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import math

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
MODEL_PATH = 'best_angle_model_region_v6.pt'  # Updated to match saved model from train_a.py
MODEL_VERSION = 'efficientnet_b0' # Should match the version used in training
OUTPUT_CSV_PATH = 'angle_v7.csv' # New output file name
BATCH_SIZE = 16 # Can be adjusted based on available memory
NUM_REGIONS = 15 # Assuming 15 regions, should match train_a.py
EMBEDDING_DIM = 128 # Assuming, should match train_a.py

# --- Import Model from train_a.py ---
# This assumes train_a.py is in the same directory or accessible via PYTHONPATH
try:
    from train_a import AngleModelWithRegion
except ImportError:
    print("Error: Could not import AngleModelWithRegion from train_a.py.")
    print("Ensure train_a.py is in the same directory or accessible in PYTHONPATH.")
    exit()

# --- Utility Functions (from Direction_Train_and_Inference.ipynb / train_a.py) ---
def compute_angle_from_sincos(sin_val, cos_val):
    return (math.degrees(math.atan2(sin_val, cos_val)) + 360) % 360

def angular_error_single(pred_angle_deg, target_angle_deg):
    target_angle_deg = target_angle_deg % 360
    pred_angle_deg = pred_angle_deg % 360
    diff = abs(pred_angle_deg - target_angle_deg)
    return min(diff, 360 - diff)

# --- Dataset for Inference ---
# Modified to accept either csv_file or dataframe
class InferenceAngleDataset(Dataset):
    def __init__(self, image_dir, csv_file=None, dataframe=None, transform=None): # MODIFIED
        if dataframe is not None:
            self.data = dataframe
        elif csv_file is not None:
            self.data = pd.read_csv(csv_file)
            self.data.attrs['name'] = csv_file
        else:
            raise ValueError("Must provide either csv_file or dataframe to InferenceAngleDataset")
            
        self.image_dir = image_dir
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        if 'filename' not in self.data.columns:
            raise ValueError(f"DataFrame/CSV must contain a 'filename' column. Found: {self.data.columns}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_filename = row['filename']
        img_path = os.path.join(self.image_dir, img_filename)

        if not os.path.exists(img_path):
            # Fallback for potential nested structures like 'images_val/images_val/'
            base_image_dir_name = os.path.basename(self.image_dir)
            img_path_fallback = os.path.join(self.image_dir, base_image_dir_name, img_filename)
            if os.path.exists(img_path_fallback):
                img_path = img_path_fallback
            else:
                 raise FileNotFoundError(f"Image {img_filename} not found in {self.image_dir} or common subdirectories like {img_path_fallback}.")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        item = {'image': image, 'filename': img_filename}
        
        # Load pred_Region_ID for validation data
        if 'pred_Region_ID' in self.data.columns:
            item['region_id'] = torch.tensor(row['pred_Region_ID'] - 1, dtype=torch.long)
        else:
            # This case should ideally not happen if val_csv_path is correct
            # For test set (TestDirImageDataset), region_id won't be available from CSV.
            # For val set, it's an error if missing.
            print(f"Warning: 'pred_Region_ID' not found in {self.data.columns} for {img_filename}. Using default region_id 0.")
            item['region_id'] = torch.tensor(0, dtype=torch.long) # Default or raise error

        if 'angle' in row: # For MAAE calculation if needed by predict_angles
            item['angle'] = row['angle']
        return item

# --- Dataset for Test Directory (Modified to include region predictions) ---
class TestDirImageDataset(Dataset):
    def __init__(self, img_dir, region_csv_path, transform=None): # MODIFIED: Added region_csv_path
        self.img_dir = img_dir
        self.image_filenames = sorted([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])
        
        try:
            self.region_predictions = pd.read_csv(region_csv_path)
        except FileNotFoundError:
            print(f"Error: Region CSV file not found at {region_csv_path}")
            raise
        
        if 'Region_ID' not in self.region_predictions.columns:
            raise ValueError(f"'Region_ID' column not found in {region_csv_path}")
        
        if len(self.image_filenames) != len(self.region_predictions):
            print(f"Warning: Number of images in {img_dir} ({len(self.image_filenames)}) "
                  f"does not match number of entries in {region_csv_path} ({len(self.region_predictions)}). "
                  "This might lead to errors or mismatches if the CSV 'id' does not align with sorted image indices.")

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image {img_name} not found in {img_path}.")
            raise
            
        if self.transform:
            image = self.transform(image)
        
        try:
            # Assuming the CSV 'id' column is 0-indexed and matches idx,
            # or that the CSV is ordered by image and its rows correspond to sorted image_filenames.
            region_id_value = self.region_predictions.iloc[idx]['Region_ID']
        except IndexError:
            print(f"Error: Index {idx} out of bounds for region predictions (length {len(self.region_predictions)}). Image: {img_name}")
            region_id_value = 1 # Default to region 1 (0-indexed for embedding will be 0)
            print(f"Warning: Using default region_id {region_id_value} (1-based) for image {img_name} due to lookup error.")
        except KeyError: 
            print(f"Error: KeyError looking up 'Region_ID' for index {idx}. Image: {img_name}")
            region_id_value = 1
            print(f"Warning: Using default region_id {region_id_value} (1-based) for image {img_name} due to KeyError.")

        region_id = torch.tensor(region_id_value - 1, dtype=torch.long) # 0-based index for embedding

        return image, img_name, region_id # MODIFIED: return region_id

# --- Functions for Combined Val and Test CSV Output (similar to test_r.py and prediction_angle.py) ---
def predict_and_save_val_angles(model, val_loader, output_csv_path, device):
    model.eval()
    all_pred_angles = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Predicting on validation set for CSV"):
            images = batch['image'].to(device)
            region_ids = batch.get('region_id', None) # Get region_id if present
            if region_ids is not None:
                region_ids = region_ids.to(device)
            
            predictions_sincos = model(images, region_ids) # Pass region_ids
            
            for i in range(predictions_sincos.size(0)):
                sin_val = predictions_sincos[i, 0].item()
                cos_val = predictions_sincos[i, 1].item()
                predicted_angle = compute_angle_from_sincos(sin_val, cos_val)
                all_pred_angles.append(round(predicted_angle)) # Round to nearest integer
            
    result_df = pd.DataFrame({
        'id': np.arange(len(all_pred_angles)),
        'angle': all_pred_angles
    })
    result_df['angle'] = result_df['angle'].astype(int) % 360 # Ensure angle is 0-359
    result_df.to_csv(output_csv_path, index=False)
    print(f"Validation predictions for CSV saved to {output_csv_path}")

def predict_on_test_and_append_angles(model, test_dir_loader, output_csv_path, device):
    model.eval()
    all_pred_angles = []
    # image_filenames_processed = [] # If you need to track filenames

    with torch.no_grad():
        # MODIFIED: Unpack region_ids_batch
        for images, filenames_batch, region_ids_batch in tqdm(test_dir_loader, desc="Predicting on test set for CSV"):
            images = images.to(device)
            region_ids_batch = region_ids_batch.to(device) # NEW: Move region_ids to device
            
            # MODIFIED: Pass region_ids_batch to the model
            predictions_sincos = model(images, region_ids_batch) 

            for i in range(predictions_sincos.size(0)):
                sin_val = predictions_sincos[i, 0].item()
                cos_val = predictions_sincos[i, 1].item()
                predicted_angle = compute_angle_from_sincos(sin_val, cos_val)
                all_pred_angles.append(round(predicted_angle))

    test_predictions_df = pd.DataFrame({'angle': all_pred_angles})
    test_predictions_df['angle'] = test_predictions_df['angle'].astype(int) % 360

    try:
        val_df_existing = pd.read_csv(output_csv_path)
        if 'id' in val_df_existing.columns: # Should always be true from predict_and_save_val_angles
            val_df_existing = val_df_existing.drop(columns=['id'])
    except FileNotFoundError:
        print(f"Error: {output_csv_path} not found. Run validation prediction first.")
        # Or create an empty val_df_existing if you want to proceed with only test, though not standard
        val_df_existing = pd.DataFrame(columns=['angle'])


    combined_df = pd.concat([val_df_existing, test_predictions_df], ignore_index=True)
    combined_df.insert(0, 'id', range(len(combined_df))) 

    combined_df.to_csv(output_csv_path, index=False)
    print(f"Appended test predictions and saved final CSV to {output_csv_path}")

# --- Prediction Function (adapted from notebook's inference and test_r.py structure) ---
# This function is for detailed MAAE calculation and can be kept or removed if not needed.
def predict_angles(model, data_loader, device):
    model.eval()
    results = []
    total_maae = 0
    count_with_angle = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting Angles"): # This is the MAAE calculation function
            images = batch['image'].to(device)
            filenames = batch['filename']
            region_ids = batch.get('region_id', None) # Get region_id if present for MAAE val
            if region_ids is not None:
                region_ids = region_ids.to(device)
            
            predictions_sincos = model(images, region_ids) # Pass region_ids
            
            for i in range(predictions_sincos.size(0)):
                sin_val = predictions_sincos[i, 0].item()
                cos_val = predictions_sincos[i, 1].item()
                predicted_angle = compute_angle_from_sincos(sin_val, cos_val)
                
                entry = {'filename': filenames[i], 'predicted_angle': predicted_angle}
                
                if 'angle' in batch and batch['angle'][i] is not None:
                    actual_angle = batch['angle'][i].item()
                    maae = angular_error_single(predicted_angle, actual_angle)
                    entry['actual_angle'] = actual_angle % 360
                    entry['maae'] = maae
                    total_maae += maae
                    count_with_angle += 1
                results.append(entry)
                
    if count_with_angle > 0:
        avg_maae = total_maae / count_with_angle
        print(f"Average MAAE (for entries with actual angles): {avg_maae:.2f}°")
    else:
        print("No actual angles provided in the test data for MAAE calculation.")
        
    return pd.DataFrame(results)

# --- Collate Functions (defined globally for multiprocessing compatibility) ---
def val_collate_fn(batch):
    collated_batch = {}
    collated_batch['image'] = torch.stack([item['image'] for item in batch])
    collated_batch['filename'] = [item['filename'] for item in batch]
    if 'region_id' in batch[0]: # Ensure region_id was loaded
        collated_batch['region_id'] = torch.stack([item['region_id'] for item in batch])
    # if 'angle' in batch[0]: # For MAAE calculation if predict_angles is used
    #    collated_batch['angle'] = torch.tensor([item['angle'] for item in batch])
    return collated_batch

def test_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    filenames = [item[1] for item in batch]
    region_ids = torch.stack([item[2] for item in batch])
    return images, filenames, region_ids

# --- Main Execution ---
if __name__ == "__main__":
    # --- Data Paths ---
    original_val_labels_path = 'labels_val.csv' # NEW: Path for original validation labels
    val_region_preds_path = 'val_predictions_region_all.csv' # Renamed for clarity
    val_image_dir = 'images_val/'
    test_image_dir_main = 'images_test/images_test/' # MODIFIED: Point to nested directory
    test_region_preds_csv_path = 'test_predictions_region_all.csv'

    print(f"Using model: {MODEL_PATH} ({MODEL_VERSION} with region embeddings)")
    print(f"Output CSV: {OUTPUT_CSV_PATH}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Device: {device}")

    # --- Validate Paths ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at {MODEL_PATH}. Please run train_a.py first or check path.")
        exit()
    if not os.path.exists(original_val_labels_path): # NEW: Validate original val labels
        print(f"Error: Original validation labels CSV not found at {original_val_labels_path}.")
        exit()
    if not os.path.exists(val_region_preds_path): # MODIFIED: Use new variable name
        print(f"Error: Validation region predictions CSV not found at {val_region_preds_path}.")
        exit()
    if not os.path.exists(val_image_dir):
        print(f"Error: Validation image directory not found at {val_image_dir}.")
        exit()
    if not os.path.exists(test_image_dir_main):
        print(f"Error: Test image directory not found at {test_image_dir_main}.")
        exit()
    if not os.path.exists(test_region_preds_csv_path): # NEW: Validate path
        print(f"Error: Test region predictions CSV not found at {test_region_preds_csv_path}.")
        exit()

    # --- Transforms (should be consistent for val and test) ---
    # Using the default transform from InferenceAngleDataset and TestDirImageDataset
    # If a custom one was used in training, it should be defined here and passed.
    # For this example, we rely on the default transforms in the dataset classes.
    # inference_transform = transforms.Compose([...]) # Define if needed

    # --- Load Model ---
    # Instantiate the new model
    model = AngleModelWithRegion(version=MODEL_VERSION, pretrained=False, num_regions=NUM_REGIONS, embedding_dim=EMBEDDING_DIM)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True)) # MODIFIED: Added weights_only=True
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model state_dict from {MODEL_PATH}: {e}")
        exit()
    model.to(device)

    # --- Validation Set Prediction ---
    print(f"\nProcessing Validation Set using {original_val_labels_path}, {val_region_preds_path}, and {val_image_dir}") # MODIFIED
    try:
        # NEW: Load and merge validation data
        val_labels_df = pd.read_csv(original_val_labels_path)
        val_preds_df = pd.read_csv(val_region_preds_path)

        if 'filename' not in val_labels_df.columns:
            raise ValueError(f"'filename' column missing in {original_val_labels_path}")
        if 'Region_ID' not in val_preds_df.columns:
            raise ValueError(f"'Region_ID' column missing in {val_region_preds_path}")

        # Rename predicted Region_ID to avoid clash if original labels also had it
        val_preds_df = val_preds_df.rename(columns={'Region_ID': 'pred_Region_ID'})
        
        # Merge based on index (assuming row order corresponds)
        val_labels_df = val_labels_df.reset_index(drop=True)
        val_preds_df = val_preds_df.reset_index(drop=True)
        val_merged_df = val_labels_df.join(val_preds_df['pred_Region_ID'])
        
        if val_merged_df['pred_Region_ID'].isnull().any():
             print("Warning: NaNs found after merging validation labels and predictions. Check alignment/lengths.")
        val_merged_df.dropna(subset=['pred_Region_ID'], inplace=True)
        val_merged_df['pred_Region_ID'] = val_merged_df['pred_Region_ID'].astype(int)

        print(f"Validation data merged. Shape: {val_merged_df.shape}. Columns: {val_merged_df.columns.tolist()}")

        # MODIFIED: Pass merged dataframe to InferenceAngleDataset
        val_dataset = InferenceAngleDataset(dataframe=val_merged_df, image_dir=val_image_dir) 
        
        # val_collate_fn is now defined globally
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, collate_fn=val_collate_fn)
    except Exception as e:
        print(f"Error creating validation dataset/loader: {e}")
        exit()
    
    predict_and_save_val_angles(model, val_loader, OUTPUT_CSV_PATH, device)

    # --- Test Set Prediction ---
    print(f"\nProcessing Test Set: {test_image_dir_main} using regions from {test_region_preds_csv_path}") # MODIFIED: Updated print
    try:
        # MODIFIED: Pass test_region_preds_csv_path to TestDirImageDataset
        test_dir_dataset = TestDirImageDataset(img_dir=test_image_dir_main, region_csv_path=test_region_preds_csv_path) 
        
        # test_collate_fn is now defined globally
        test_dir_loader = DataLoader(test_dir_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, collate_fn=test_collate_fn)
    except Exception as e:
        print(f"Error creating test dataset/loader: {e}")
        exit()

    predict_on_test_and_append_angles(model, test_dir_loader, OUTPUT_CSV_PATH, device)

    print(f"\nPredictions for validation and test sets saved to {OUTPUT_CSV_PATH}")

    # If you still want to run the MAAE calculation separately, you can call predict_angles here
    # with the val_loader (or a similar loader for test if you have test labels).
    # For example, to calculate MAAE on the validation set:
    # print("\nCalculating MAAE on validation set (using predict_angles function):")
    # predictions_details_df = predict_angles(model, val_loader, device)
    # if not predictions_details_df.empty:
    #     print("\nSample predictions with MAAE details (validation set):")
    #     print(predictions_details_df.head())
