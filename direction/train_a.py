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
import torch.nn.functional as F
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_loss(output, target_sincos):
    mse_loss = F.mse_loss(output, target_sincos)
    
    # Angular loss (1 - cosine similarity)
    cosine_sim = F.cosine_similarity(output, target_sincos)
    angular_loss = 1 - cosine_sim.mean()
    return mse_loss + 0.5 * angular_loss

# Hyperparameters (can be adjusted or parsed from args)
batch_size = 16 # From notebook
num_epochs = 100 # From notebook
lr = 1e-4 # From notebook
MODEL_SAVE_PATH = 'best_angle_model_region_v6.pt' # Ensuring correct name
MODEL_VERSION = 'efficientnet_b3' # Default from notebook

# --- Dataset Definition (from Direction_Train_and_Inference.ipynb) ---
class AngleDataset(Dataset):
    # Modified __init__ to accept either csv_file or dataframe
    def __init__(self, image_dir, csv_file=None, dataframe=None, transform=None): 
        if dataframe is not None:
            self.data = dataframe
        elif csv_file is not None:
            self.data = pd.read_csv(csv_file)
            # Store csv_file name in df attributes for better error messages if needed
            self.data.attrs['name'] = csv_file 
        else:
            raise ValueError("Must provide either csv_file or dataframe to AngleDataset")
            
        self.image_dir = image_dir
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Direct column access then iloc[idx]
        try:
            img_filename = self.data['filename'].iloc[idx]
            angle_value = self.data['angle'].iloc[idx]
            pred_region_id_value = self.data['pred_Region_ID'].iloc[idx]
        except KeyError as e:
            print(f"KeyError in AngleDataset: {e} not found in columns: {self.data.columns}. CSV: {self.data.attrs.get('name', 'Unknown')}")
            raise e
        except IndexError as e:
            print(f"IndexError in AngleDataset: idx {idx} out of bounds for DataFrame of length {len(self.data)}. CSV: {self.data.attrs.get('name', 'Unknown')}")
            raise e
            
        # Attempt to construct the full path, checking common locations
        img_path = os.path.join(self.image_dir, img_filename)
        if not os.path.exists(img_path):
            # Fallback for potential nested structures like 'images_train/images_train/'
            base_image_dir_name = os.path.basename(self.image_dir)
            img_path_fallback = os.path.join("images_val/", img_filename)
            if os.path.exists(img_path_fallback):
                img_path = img_path_fallback
            else:
                # Try another common pattern if the first fallback doesn't work
                # e.g. if self.image_dir is 'images_train' and files are in 'images_train/images_train'
                # This part might need adjustment based on actual directory structures
                parts = self.image_dir.split(os.sep)
                if len(parts) > 0:
                    potential_base = os.path.join(*parts[:-1], parts[-1], parts[-1], img_filename)
                    if os.path.exists(potential_base):
                        img_path = potential_base
                    else:
                        # Last resort, print error and raise exception or return None
                        print(f"Error: Image not found at {img_path} or fallback paths for {img_filename}")
                        # Depending on desired behavior, either raise an error or handle it
                        # For now, let's assume the primary path should work or a simple fallback
                        # If issues persist, this path logic needs to be more robust or configurable
                        raise FileNotFoundError(f"Image {img_filename} not found in {self.image_dir} or common subdirectories.")
        
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
            
        angle_rad = math.radians(angle_value % 360)
        target = torch.tensor([math.sin(angle_rad), math.cos(angle_rad)], dtype=torch.float32)
        
        region_id = torch.tensor(pred_region_id_value - 1, dtype=torch.long) # 0-based index for embedding

        return image, region_id, target

# --- Model Definition (Adapted to include Region Embedding) ---
class AngleModelWithRegion(nn.Module):
    def __init__(self, version='efficientnet_b0', pretrained=True, num_regions=15, embedding_dim=128): # num_regions from context, embedding_dim can be tuned
        super().__init__()
        backbone_map = {
            'efficientnet_b0': models.efficientnet_b0,
            'efficientnet_b1': models.efficientnet_b1,
            'efficientnet_b2': models.efficientnet_b2,
            'efficientnet_b3': models.efficientnet_b3,
        }
        if version not in backbone_map:
            raise ValueError(f"Unsupported model version: {version}")
        
        self.backbone = backbone_map[version](weights='IMAGENET1K_V1' if pretrained else None)
        
        # Image feature extractor (remove original classifier)
        if hasattr(self.backbone, 'features') and hasattr(self.backbone, 'avgpool'): # Typical for EfficientNet
             self.image_feature_extractor = nn.Sequential(
                self.backbone.features,
                self.backbone.avgpool
            )
             # Get the number of output features from the feature extractor
             # For EfficientNet-B0, it's 1280 after avgpool
             dummy_input = torch.randn(1, 3, 256, 256) # Match resize in transforms
             image_features_dim = self.image_feature_extractor(dummy_input).view(1, -1).shape[1]
        else: # Fallback for other architectures, might need adjustment
            children = list(self.backbone.children())
            self.image_feature_extractor = nn.Sequential(*children[:-1]) # Remove original FC/classifier
            dummy_input = torch.randn(1, 3, 256, 256)
            image_features_dim = self.image_feature_extractor(dummy_input).view(1, -1).shape[1]


        self.embedding_dim = embedding_dim
        self.region_embedding = nn.Embedding(num_regions, embedding_dim)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(image_features_dim + embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # Output for sin, cos
        )

        # self.region_embedding = nn.Embedding(num_regions, embedding_dim)
        # self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4)
        # self.image_proj = nn.Linear(image_features_dim, embedding_dim)
        
        # # Classifier head
        # self.classifier = nn.Sequential(
        #     nn.Linear(embedding_dim * 2, 512),  # Concatenate attended features + image features
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(256, 2)  # Output for sin, cos
        # )

        

    def forward(self, image, region_id=None): # region_id is now optional
        img_feat = self.image_feature_extractor(image)
        img_feat = img_feat.view(img_feat.size(0), -1) # Flatten
        
        if region_id is not None:
            region_emb = self.region_embedding(region_id)
        else:
            # Use a zero embedding if region_id is not provided (e.g., during test inference)
            region_emb = torch.zeros(img_feat.size(0), self.embedding_dim, device=img_feat.device)
        
        
        combined_features = torch.cat([img_feat, region_emb], dim=1)
        
        # image_proj = self.image_proj(img_feat).unsqueeze(0)  # Add sequence dimension for attention
        # region_emb = region_emb.unsqueeze(0)  # Add sequence dimension for attention

        # attn_output, _ = self.attn(image_proj, region_emb, region_emb)  # Self-attention on region embedding
        # attn_output = attn_output.squeeze(0)  # Remove sequence dimension
        # image_proj = image_proj.squeeze(0)  # Remove sequence dimension
        # combined_features = torch.cat([attn_output, image_proj], dim=1)  # Concatenate image features and attended region embedding

        output = self.classifier(combined_features)
        return output

# --- Utility Function (from Direction_Train_and_Inference.ipynb) ---
def angular_error(pred, target):
    # Ensure pred and target are 2D tensors (batch_size, 2)
    if pred.ndim == 1: pred = pred.unsqueeze(0)
    if target.ndim == 1: target = target.unsqueeze(0)

    pred_angle = torch.atan2(pred[:, 0], pred[:, 1])
    target_angle = torch.atan2(target[:, 0], target[:, 1])
    
    diff = torch.abs(pred_angle - target_angle)
    # Normalize difference to be between 0 and pi
    error_rad = torch.minimum(diff, 2 * math.pi - diff)
    error_deg = torch.rad2deg(error_rad)
    return torch.mean(error_deg)

# --- Training Function (adapted from Direction_Train_and_Inference.ipynb and train_r.py) ---
def train_model_angle(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path, fine_tune=False):
    best_maae = float('inf')
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(num_epochs):
        model.train()
        train_loss_epoch = 0
        train_maae_epoch = 0
        
        loop_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)
        for x, region_ids, y in loop_train: # Unpack image, region_id, target
            x, region_ids, y = x.to(device), region_ids.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x, region_ids) # Pass region_ids to model
            # loss = criterion(pred, y)
            loss = custom_loss(pred, y) # Use custom loss function
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Optional: gradient clipping
            optimizer.step()
            
            train_loss_epoch += loss.item()
            train_maae_epoch += angular_error(pred.detach(), y.detach()).item()
            loop_train.set_postfix(loss=loss.item(), maae=angular_error(pred.detach(), y.detach()).item())

        avg_train_loss = train_loss_epoch / len(train_loader)
        avg_train_maae = train_maae_epoch / len(train_loader)

        # Evaluation
        model.eval()
        val_loss_epoch = 0
        val_maae_epoch = 0
        
        loop_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False)
        with torch.no_grad():
            for x, region_ids, y in loop_val: # Unpack image, region_id, target
                x, region_ids, y = x.to(device), region_ids.to(device), y.to(device)
                pred = model(x, region_ids) # Pass region_ids to model
                # loss = criterion(pred, y)
                loss = custom_loss(pred, y) # Use custom loss function
                # if fine_tune: # As per notebook, but not used in train_r.py structure
                # loss *= 1.5 
                
                val_loss_epoch += loss.item()
                val_maae_epoch += angular_error(pred, y).item()
                loop_val.set_postfix(loss=loss.item(), maae=angular_error(pred, y).item())

        avg_val_loss = val_loss_epoch / len(val_loader)
        avg_val_maae = val_maae_epoch / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train MAAE: {avg_train_maae:.2f}° | Val Loss: {avg_val_loss:.4f} | Val MAAE: {avg_val_maae:.2f}°")

        # scheduler.step(avg_val_maae)

        # Save best model based on validation MAAE
        if avg_val_maae < best_maae:
            best_maae = avg_val_maae
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved model to {save_path} (New best Val MAAE: {best_maae:.2f}°)")
    
    print(f"Training finished. Best Validation MAAE: {best_maae:.2f}°")

# --- Main Execution ---
if __name__ == "__main__":
    # --- Define Paths ---
    original_train_labels_path = 'labels_train.csv'
    original_val_labels_path = 'labels_val.csv'
    train_region_preds_path = 'train_predictions_region_all.csv' 
    val_region_preds_path = 'val_predictions_region_all.csv'     
    train_image_dir = 'images_train/' 
    val_image_dir = 'images_val/'   

    # --- Load and Merge Data ---
    try:
        print(f"Loading original labels: {original_train_labels_path}, {original_val_labels_path}")
        train_labels_df = pd.read_csv(original_train_labels_path)
        val_labels_df = pd.read_csv(original_val_labels_path)

        print(f"Loading region predictions: {train_region_preds_path}, {val_region_preds_path}")
        train_preds_df = pd.read_csv(train_region_preds_path)
        val_preds_df = pd.read_csv(val_region_preds_path)

        # Validate required columns
        required_label_cols = ['filename', 'angle']
        required_pred_cols = ['Region_ID'] # Assuming 'id' is just index or join key
        if not all(col in train_labels_df.columns for col in required_label_cols):
            raise ValueError(f"Missing required columns in {original_train_labels_path}. Need: {required_label_cols}")
        if not all(col in val_labels_df.columns for col in required_label_cols):
             raise ValueError(f"Missing required columns in {original_val_labels_path}. Need: {required_label_cols}")
        if not all(col in train_preds_df.columns for col in required_pred_cols):
            raise ValueError(f"Missing required columns in {train_region_preds_path}. Need: {required_pred_cols}")
        if not all(col in val_preds_df.columns for col in required_pred_cols):
            raise ValueError(f"Missing required columns in {val_region_preds_path}. Need: {required_pred_cols}")

        # Merge based on index (assuming row order corresponds)
        # Rename predicted Region_ID to avoid clash if original labels also had it
        train_preds_df = train_preds_df.rename(columns={'Region_ID': 'pred_Region_ID'})
        val_preds_df = val_preds_df.rename(columns={'Region_ID': 'pred_Region_ID'})

        # Merge - using index. If prediction files have 'id' matching index, this works.
        # If 'id' corresponds to a different column in labels_df, adjust merge 'on' key.
        # Ensure indices are clean before joining
        train_labels_df = train_labels_df.reset_index(drop=True)
        val_labels_df = val_labels_df.reset_index(drop=True)
        train_preds_df = train_preds_df.reset_index(drop=True)
        val_preds_df = val_preds_df.reset_index(drop=True)
        
        train_merged_df = train_labels_df.join(train_preds_df['pred_Region_ID'])
        val_merged_df = val_labels_df.join(val_preds_df['pred_Region_ID'])
        
        # Check for NaNs after merge, which indicates misalignment or length mismatch
        if train_merged_df['pred_Region_ID'].isnull().any():
             print("Warning: NaNs found after merging train labels and predictions. Check alignment/lengths.")
        if val_merged_df['pred_Region_ID'].isnull().any():
             print("Warning: NaNs found after merging val labels and predictions. Check alignment/lengths.")
             
        # Drop rows with NaN predictions if any, or handle differently
        train_merged_df.dropna(subset=['pred_Region_ID'], inplace=True)
        val_merged_df.dropna(subset=['pred_Region_ID'], inplace=True)
        
        # Convert region ID to int just in case
        train_merged_df['pred_Region_ID'] = train_merged_df['pred_Region_ID'].astype(int)
        val_merged_df['pred_Region_ID'] = val_merged_df['pred_Region_ID'].astype(int)

        print("Data loading and merging complete.")
        print(f"Train samples: {len(train_merged_df)}, Val samples: {len(val_merged_df)}")

        # --- Debug: Inspect Merged DataFrames ---
        print("\n--- Inspecting train_merged_df ---")
        print(f"Columns: {train_merged_df.columns.tolist()}")
        print(f"Shape: {train_merged_df.shape}")
        print("Head:\n", train_merged_df.head()) # Uncommented head
        # print("Tail:\n", train_merged_df.tail()) # Print tail if needed
        print(f"NaN check:\n{train_merged_df.isnull().sum()}")

        print("\n--- Inspecting val_merged_df ---")
        print(f"Columns: {val_merged_df.columns.tolist()}")
        print(f"Shape: {val_merged_df.shape}")
        print("Head:\n", val_merged_df.head()) # Uncommented head
        # print("Tail:\n", val_merged_df.tail()) # Print tail if needed
        print(f"NaN check:\n{val_merged_df.isnull().sum()}")
        # --- End Debug ---

    except FileNotFoundError as e:
        print(f"Error: CSV file not found: {e}. Please check paths.")
        exit()
    except ValueError as e:
        print(f"Error: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during data loading/merging: {e}")
        exit()
        
    # --- Validate Image Directories ---
    if not os.path.exists(train_image_dir):
        print(f"Warning: Train image directory not found at {train_image_dir}. Please check the path.")
    if not os.path.exists(val_image_dir):
        print(f"Warning: Validation image directory not found at {val_image_dir}. Please check the path.")
        
    # --- Transforms ---
    # Using transforms similar to train_r.py for consistency, but adapted for Angle task
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)), # EfficientNet default input size often 224 for b0, but notebook uses 256
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Datasets and DataLoaders ---
    # Pass the MERGED dataframes to the dataset constructor
    try:
        # We need a way to pass the dataframe directly, let's modify AngleDataset slightly
        # Or, save the merged dfs to temp files and pass paths (less ideal)
        # Let's modify AngleDataset to accept a DataFrame directly
        
        # Modify AngleDataset __init__ to accept df instead of csv_file
        # We'll do this with replace_in_file *before* this block executes
        
        train_dataset = AngleDataset(dataframe=train_merged_df, image_dir=train_image_dir, transform=train_transform)
        val_dataset = AngleDataset(dataframe=val_merged_df, image_dir=val_image_dir, transform=val_transform)
        
        # Combine train and val datasets for training
        combined_train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
        
        # DataLoader for the combined dataset, shuffled for training (num_workers=0 for debugging)
        print("Initializing train_loader with num_workers=0 for debugging...")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        
        # DataLoader for the original validation dataset (no shuffling) for evaluation (num_workers=0 for debugging)
        print("Initializing val_loader with num_workers=0 for debugging...")
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    except FileNotFoundError as e:
        print(f"Error creating datasets: {e}")
        print("Please ensure CSV files and image directories are correctly specified and accessible.")
        exit()
    except ValueError as e: # Catch missing 'angle' column
        print(f"Error creating datasets: {e}")
        exit()


    # --- Model, Criterion, Optimizer ---
    # Determine num_regions dynamically if possible, or use a fixed value like 15
    # For dynamic, you'd load train_df here first:
    # temp_train_df = pd.read_csv(train_csv_path)
    # num_unique_regions = temp_train_df['pred_Region_ID'].nunique()
    # print(f"Number of unique regions found in training data: {num_unique_regions}")
    num_unique_regions = 15 # Assuming 15 regions as per other scripts
    
    model = AngleModelWithRegion(version=MODEL_VERSION, pretrained=True, num_regions=num_unique_regions).to(device)
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training for angle prediction using {MODEL_VERSION} with region embeddings...")
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}, Epochs: {num_epochs}, LR: {lr}")
    print(f"Model will be saved to: {MODEL_SAVE_PATH}")

    train_model_angle(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        save_path=MODEL_SAVE_PATH
    )
