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

# Configuration (Should match train_l.py)
MODEL_VERSION = 'efficientnet_b3' # Make sure this matches the trained model
MODEL_PATH = f'best_latlong_model_v1_aligned.pt' # Path to the trained model
OUTPUT_CSV_PATH = 'latlong_v1.csv' # Output file name
BATCH_SIZE = 32 # Can be adjusted based on available memory
NUM_REGIONS = 15
EMBEDDING_DIM = 256

# --- Import Model from train_l.py ---
try:
    # Ensure train_l.py is accessible
    from train_l import LatLongModelWithRegion 
except ImportError:
    print("Error: Could not import LatLongModelWithRegion from train_l.py.")
    print("Ensure train_l.py is in the same directory or accessible in PYTHONPATH.")
    exit()

# --- Dataset for Validation Inference ---
# Loads data from a pre-merged DataFrame including filename, lat/lon (optional), and pred_Region_ID
class InferenceLatLongDataset(Dataset):
    def __init__(self, image_dir, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

        if not all(col in self.data.columns for col in ['filename', 'pred_Region_ID']):
             raise ValueError(f"DataFrame must contain 'filename', 'pred_Region_ID'. Found: {self.data.columns}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_filename = row['filename']
        
        # Construct image path (using logic similar to train_l.py)
        img_path = os.path.join(self.image_dir, img_filename)
        if not os.path.exists(img_path):
            base_image_dir_name = os.path.basename(self.image_dir)
            img_path_fallback1 = os.path.join(self.image_dir, base_image_dir_name, img_filename)
            img_path_fallback2 = os.path.join(os.path.dirname(self.image_dir), base_image_dir_name, base_image_dir_name, img_filename)
            
            if os.path.exists(img_path_fallback1):
                img_path = img_path_fallback1
            elif os.path.exists(img_path_fallback2) and self.image_dir == base_image_dir_name:
                 img_path = img_path_fallback2
            elif os.path.exists(os.path.join("images_val", "images_val", img_filename)): # Specific for val
                img_path = os.path.join("images_val", "images_val", img_filename)
            else:
                 raise FileNotFoundError(f"Image {img_filename} not found in {self.image_dir} or common fallbacks.")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        region_id = torch.tensor(row['pred_Region_ID'] - 1, dtype=torch.long) # 0-based index

        # Return image, region_id, and filename for potential reference
        return image, region_id, img_filename 

# --- Dataset for Test Directory Inference ---
# Loads images directly from directory and merges with region predictions CSV
class TestDirLatLongDataset(Dataset):
    def __init__(self, img_dir, region_csv_path, transform=None):
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
                  "Assuming CSV rows correspond to sorted image filenames.")

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
            # Assuming the CSV rows correspond to the sorted image filenames
            region_id_value = self.region_predictions.iloc[idx]['Region_ID']
        except IndexError:
            print(f"Error: Index {idx} out of bounds for region predictions (length {len(self.region_predictions)}). Image: {img_name}")
            region_id_value = 1 # Default region
            print(f"Warning: Using default region_id {region_id_value} for image {img_name} due to lookup error.")
        
        region_id = torch.tensor(region_id_value - 1, dtype=torch.long) # 0-based index

        return image, region_id, img_name # Return image, region_id, filename

# --- Collate Functions (defined globally) ---
def val_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    region_ids = torch.stack([item[1] for item in batch])
    filenames = [item[2] for item in batch]
    return images, region_ids, filenames

def test_collate_fn(batch):
    # Same structure as val_collate_fn for this case
    images = torch.stack([item[0] for item in batch])
    region_ids = torch.stack([item[1] for item in batch])
    filenames = [item[2] for item in batch]
    return images, region_ids, filenames

# --- Prediction Functions ---
def predict_and_save_val_latlong(model, val_loader, output_csv_path, device, 
                                 lat_mean, lat_std, lon_mean, lon_std):
    model.eval()
    all_pred_lat = []
    all_pred_lon = []
    all_ids = []

    with torch.no_grad():
        for images, region_ids, sample_ids in tqdm(val_loader, desc="Predicting on validation set"):
            images, region_ids = images.to(device), region_ids.to(device)
            
            preds_normalized = model(images, region_ids)
            
            # Un-normalize predictions
            pred_lat_batch = preds_normalized[:, 0] * lat_std + lat_mean
            pred_lon_batch = preds_normalized[:, 1] * lon_std + lon_mean
            
            all_pred_lat.extend(pred_lat_batch.cpu().numpy())
            all_pred_lon.extend(pred_lon_batch.cpu().numpy())
            all_ids.extend(sample_ids)
            
    result_df = pd.DataFrame({
        'id': all_ids,
        'latitude': all_pred_lat,
        'longitude': all_pred_lon
    })
    # Optional: Rounding predictions if desired
    # result_df['latitude'] = result_df['latitude'].round()
    # result_df['longitude'] = result_df['longitude'].round()
    
    result_df.to_csv(output_csv_path, index=False)
    print(f"Validation predictions saved to {output_csv_path}")
    return len(result_df) # Return number of validation predictions

def predict_on_test_and_append_latlong(model, test_loader, output_csv_path, device, 
                                       lat_mean, lat_std, lon_mean, lon_std, val_count):
    model.eval()
    all_pred_lat = []
    all_pred_lon = []

    with torch.no_grad():
        for images, region_ids, _ in tqdm(test_loader, desc="Predicting on test set"):
            images, region_ids = images.to(device), region_ids.to(device)
            
            preds_normalized = model(images, region_ids)
            
            # Un-normalize predictions
            pred_lat_batch = preds_normalized[:, 0] * lat_std + lat_mean
            pred_lon_batch = preds_normalized[:, 1] * lon_std + lon_mean
            
            all_pred_lat.extend(pred_lat_batch.cpu().numpy())
            all_pred_lon.extend(pred_lon_batch.cpu().numpy())

    test_predictions_df = pd.DataFrame({
        'latitude': all_pred_lat,
        'longitude': all_pred_lon
    })
    # Optional: Rounding predictions
    # test_predictions_df['latitude'] = test_predictions_df['latitude'].round()
    # test_predictions_df['longitude'] = test_predictions_df['longitude'].round()

    try:
        # Load existing validation predictions
        combined_df = pd.read_csv(output_csv_path)
        # Append test predictions
        combined_df = pd.concat([combined_df, test_predictions_df], ignore_index=True)
        # Ensure 'id' column covers all rows
        combined_df['id'] = combined_df.index
    except FileNotFoundError:
        print(f"Warning: {output_csv_path} not found. Saving only test predictions.")
        test_predictions_df.insert(0, 'id', range(val_count, val_count + len(test_predictions_df)))
        combined_df = test_predictions_df

    combined_df.to_csv(output_csv_path, index=False)
    print(f"Appended test predictions and saved final CSV to {output_csv_path}")

# --- Main Execution ---
if __name__ == "__main__":
    # --- Define Paths ---
    original_train_labels_path = 'labels_train.csv'
    original_val_labels_path = 'labels_val.csv'
    train_region_preds_path = 'train_predictions_region_all.csv'
    val_region_preds_path = 'val_predictions_region_all.csv'
    test_region_preds_csv_path = 'test_predictions_region_all.csv'
    train_image_dir = 'images_train/' # Used for calculating normalization stats
    val_image_dir = 'images_val/'
    test_image_dir_main = 'images_test/images_test/' # Adjusted path

    print(f"Using model: {MODEL_PATH} ({MODEL_VERSION})")
    print(f"Output CSV: {OUTPUT_CSV_PATH}")
    print(f"Device: {device}")

    # --- Recalculate Normalization Statistics from Filtered Training Data ---
    # This is crucial to ensure consistency with train_l.py
    try:
        print("Loading training data to calculate normalization statistics...")
        train_labels_df = pd.read_csv(original_train_labels_path)
        train_preds_df = pd.read_csv(train_region_preds_path)
        train_preds_df = train_preds_df.rename(columns={'Region_ID': 'pred_Region_ID'})
        
        train_labels_df = train_labels_df.reset_index(drop=True)
        train_preds_df = train_preds_df.reset_index(drop=True)
        train_merged_df_for_norm = train_labels_df.join(train_preds_df['pred_Region_ID'])
        train_merged_df_for_norm.dropna(subset=['pred_Region_ID'], inplace=True)
        train_merged_df_for_norm['pred_Region_ID'] = train_merged_df_for_norm['pred_Region_ID'].astype(int)

        # Apply the same outlier filtering as in train_l.py
        lat_lower = train_merged_df_for_norm['latitude'].quantile(0.01)
        lat_upper = train_merged_df_for_norm['latitude'].quantile(0.99)
        lon_lower = train_merged_df_for_norm['longitude'].quantile(0.01)
        lon_upper = train_merged_df_for_norm['longitude'].quantile(0.99)
        train_df_filtered = train_merged_df_for_norm[
            (train_merged_df_for_norm['latitude'].between(lat_lower, lat_upper)) &
            (train_merged_df_for_norm['longitude'].between(lon_lower, lon_upper))
        ].copy()

        lat_mean = train_df_filtered['latitude'].mean()
        lat_std = train_df_filtered['latitude'].std()
        lon_mean = train_df_filtered['longitude'].mean()
        lon_std = train_df_filtered['longitude'].std()
        print(f"Recalculated Normalization params: LatMean={lat_mean:.4f}, LatStd={lat_std:.4f}, LonMean={lon_mean:.4f}, LonStd={lon_std:.4f}")

    except Exception as e:
        print(f"Error loading or processing training data for normalization: {e}")
        exit()

    # --- Load Model ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at {MODEL_PATH}.")
        exit()
    
    model = LatLongModelWithRegion(version=MODEL_VERSION, pretrained=False, 
                                   num_regions=NUM_REGIONS, embedding_dim=EMBEDDING_DIM)
    try:
        # Load weights safely
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model state_dict from {MODEL_PATH}: {e}")
        exit()
    model.to(device)

    # --- Define Transforms (consistent with train_l.py) ---
    try:
        model_class = getattr(models, MODEL_VERSION)
        image_size = model_class(weights='IMAGENET1K_V1').default_image_size
        if isinstance(image_size, tuple) and len(image_size) >= 2:
            image_size = image_size[0] 
    except Exception as e:
        print(f"Warning: Could not get default image size for {MODEL_VERSION}, using 224. Error: {e}")
        image_size = 224
        
    inference_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Validation Set Prediction ---
    print(f"\nProcessing Validation Set...")
    try:
        val_labels_df = pd.read_csv(original_val_labels_path)
        val_preds_df = pd.read_csv(val_region_preds_path)
        val_preds_df = val_preds_df.rename(columns={'Region_ID': 'pred_Region_ID'})

        val_labels_df = val_labels_df.reset_index(drop=True)
        val_preds_df = val_preds_df.reset_index(drop=True)
        val_merged_df = val_labels_df.join(val_preds_df['pred_Region_ID'])
        val_merged_df.dropna(subset=['pred_Region_ID'], inplace=True)
        val_merged_df['pred_Region_ID'] = val_merged_df['pred_Region_ID'].astype(int)
        print(f"Validation data loaded: {val_merged_df.shape}")
        # Apply same row removal as in train_l.py
        # rows_to_remove_val = [95, 145, 146, 158, 159, 160, 161] 
        
        # valid_indices_to_drop = val_merged_df.index.intersection(rows_to_remove_val)
        # val_df_filtered = val_merged_df.drop(index=valid_indices_to_drop).copy()
        # val_df_filtered = val_merged_df.drop(index=valid_indices_to_drop).reset_index(drop=True).copy()
        # print(f"Validation data loaded and filtered: {val_df_filtered.shape}")

        val_dataset = InferenceLatLongDataset(dataframe=val_merged_df, image_dir=val_image_dir, transform=inference_transform)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, collate_fn=val_collate_fn)
        
        val_count = predict_and_save_val_latlong(model, val_loader, OUTPUT_CSV_PATH, device, lat_mean, lat_std, lon_mean, lon_std)

    except Exception as e:
        print(f"Error processing validation set: {e}")
        exit()

    # --- Test Set Prediction ---
    print(f"\nProcessing Test Set: {test_image_dir_main}")
    if not os.path.exists(test_image_dir_main):
         print(f"Error: Test image directory not found at {test_image_dir_main}")
         exit()
    if not os.path.exists(test_region_preds_csv_path):
         print(f"Error: Test region predictions CSV not found at {test_region_preds_csv_path}")
         exit()
         
    try:
        test_dataset = TestDirLatLongDataset(img_dir=test_image_dir_main, region_csv_path=test_region_preds_csv_path, transform=inference_transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, collate_fn=test_collate_fn)
        
        predict_on_test_and_append_latlong(model, test_loader, OUTPUT_CSV_PATH, device, lat_mean, lat_std, lon_mean, lon_std, val_count)

    except Exception as e:
        print(f"Error processing test set: {e}")
        exit()

    print(f"\nPredictions for validation and test sets saved to {OUTPUT_CSV_PATH}")
