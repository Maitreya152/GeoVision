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

# Hyperparameters
MODEL_VERSION = 'efficientnet_b3' # Backbone model
MODEL_SAVE_PATH = f'best_latlong_model_v1_aligned.pt' # Updated save path name
BATCH_SIZE = 32
NUM_EPOCHS = 100 # Aligned with train_latlong.py
LEARNING_RATE = 1e-4
NUM_REGIONS = 15 # Assuming 15 regions, consistent with other scripts
EMBEDDING_DIM = 256 # Aligned with train_latlong.py

# --- Dataset Definition ---
class LatLongDataset(Dataset):
    def __init__(self, image_dir, dataframe, transform=None, 
                 lat_mean=None, lat_std=None, lon_mean=None, lon_std=None):
        self.data = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.lat_mean = lat_mean
        self.lat_std = lat_std
        self.lon_mean = lon_mean
        self.lon_std = lon_std

        if not all(col in self.data.columns for col in ['filename', 'latitude', 'longitude', 'pred_Region_ID']):
            raise ValueError(f"DataFrame must contain 'filename', 'latitude', 'longitude', 'pred_Region_ID'. Found: {self.data.columns}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_filename = row['filename']
        
        # Construct image path, trying common structures
        img_path = os.path.join(self.image_dir, img_filename)
        if not os.path.exists(img_path):
            # Fallback for potential nested structures like 'images_train/images_train/'
            # This logic is adapted from train_a.py and train_latlong.py
            base_image_dir_name = os.path.basename(self.image_dir) # e.g. images_train
            img_path_fallback1 = os.path.join(self.image_dir, base_image_dir_name, img_filename) # e.g. images_train/images_train/file.jpg
            img_path_fallback2 = os.path.join(os.path.dirname(self.image_dir), base_image_dir_name, base_image_dir_name, img_filename) # e.g. images_train/images_train/file.jpg if self.image_dir is just 'images_train'

            if os.path.exists(img_path_fallback1):
                img_path = img_path_fallback1
            elif os.path.exists(img_path_fallback2) and self.image_dir == base_image_dir_name : # only use fallback2 if image_dir is simple like 'images_train'
                 img_path = img_path_fallback2
            elif os.path.exists(os.path.join("images_train", "images_train", img_filename)) and "train" in self.image_dir.lower(): # Specific for train
                img_path = os.path.join("images_train", "images_train", img_filename)
            elif os.path.exists(os.path.join("images_val", "images_val", img_filename)) and "val" in self.image_dir.lower(): # Specific for val
                img_path = os.path.join("images_val", "images_val", img_filename)
            else:
                raise FileNotFoundError(f"Image {img_filename} not found in {self.image_dir} or common fallbacks like {img_path_fallback1}, {img_path_fallback2}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Normalize latitude and longitude
        lat = (row['latitude'] - self.lat_mean) / self.lat_std
        lon = (row['longitude'] - self.lon_mean) / self.lon_std
        target = torch.tensor([lat, lon], dtype=torch.float32)
        
        region_id = torch.tensor(row['pred_Region_ID'] - 1, dtype=torch.long) # 0-based index

        return image, region_id, target

# --- Model Definition ---
class LatLongModelWithRegion(nn.Module):
    def __init__(self, version=MODEL_VERSION, pretrained=True, num_regions=NUM_REGIONS, embedding_dim=EMBEDDING_DIM):
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
        
        if hasattr(self.backbone, 'features') and hasattr(self.backbone, 'avgpool'):
             self.image_feature_extractor = nn.Sequential(
                self.backbone.features,
                self.backbone.avgpool
            )
             dummy_input = torch.randn(1, 3, 224, 224) # EfficientNet typical input size
             image_features_dim = self.image_feature_extractor(dummy_input).view(1, -1).shape[1]
        else:
            children = list(self.backbone.children())
            self.image_feature_extractor = nn.Sequential(*children[:-1])
            dummy_input = torch.randn(1, 3, 224, 224)
            image_features_dim = self.image_feature_extractor(dummy_input).view(1, -1).shape[1]

        self.region_embedding = nn.Embedding(num_regions, embedding_dim)
        
        combined_features_dim = image_features_dim + embedding_dim
        
        # Attention mechanism inspired by train_latlong.py
        # It produces two weights, and one is used to scale the combined features.
        self.attention_module = nn.Sequential(
            nn.Linear(combined_features_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 2), # Output two attention "scores" or "weights"
            nn.Softmax(dim=1)  # Apply softmax to get normalized weights
        )
        
        # Classifier head, takes the attended features
        # Aligned with train_latlong.py's head structure
        self.classifier = nn.Sequential(
            nn.Linear(combined_features_dim, 512),
            nn.ReLU(),
            # Removed extra layers and dropout to match train_latlong.py
            nn.Linear(512, 2)  # Output for lat, lon
        )

    def forward(self, image, region_id):
        img_feat = self.image_feature_extractor(image)
        img_feat = img_feat.view(img_feat.size(0), -1) # Flatten
        
        region_emb = self.region_embedding(region_id)
        
        combined_features = torch.cat([img_feat, region_emb], dim=1)
        
        # Calculate attention weights using the attention_module
        attn_weights = self.attention_module(combined_features) # Shape: [batch_size, 2]
        
        # Use the first set of attention weights to scale the combined_features
        # This mimics the behavior in train_latlong.py: combined * attn_weights[:,0].unsqueeze(1)
        attended_features = combined_features * attn_weights[:, 0].unsqueeze(1)
        
        output = self.classifier(attended_features)
        return output

# --- Loss Function: Unnormalized and Rounded MSE (average_mse from train_latlong.py) ---
# Renamed for clarity and to be used as the main criterion
def unnormalized_rounded_mse_loss(pred_normalized, target_normalized, lat_mean, lat_std, lon_mean, lon_std):
    # Un-normalize
    pred_lat = pred_normalized[:, 0] * lat_std + lat_mean
    pred_lon = pred_normalized[:, 1] * lon_std + lon_mean
    target_lat = target_normalized[:, 0] * lat_std + lat_mean
    target_lon = target_normalized[:, 1] * lon_std + lon_mean
    
    # Round (as in train_latlong.py's average_mse)
    pred_lat = torch.round(pred_lat)
    pred_lon = torch.round(pred_lon)
    target_lat = torch.round(target_lat)
    target_lon = torch.round(target_lon)

    lat_mse = nn.functional.mse_loss(pred_lat, target_lat)
    lon_mse = nn.functional.mse_loss(pred_lon, target_lon)
    return (lat_mse + lon_mse) / 2.0

# --- Training Function ---
# MODIFIED: Takes criterion (for backprop) and validation_loss_function (for scoring)
def train_model_latlong(model, train_loader, val_loader, criterion, validation_loss_function, optimizer, num_epochs, save_path,
                        lat_mean, lat_std, lon_mean, lon_std):
    best_val_loss = float('inf')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(num_epochs):
        model.train()
        train_loss_epoch = 0
        
        loop_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)
        for images, region_ids, targets_normalized in loop_train:
            images, region_ids, targets_normalized = images.to(device), region_ids.to(device), targets_normalized.to(device)
            
            optimizer.zero_grad()
            preds_normalized = model(images, region_ids)
            # MODIFIED: Use standard MSE loss on normalized values for backpropagation
            loss = criterion(preds_normalized, targets_normalized) 
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_epoch += loss.item()
            loop_train.set_postfix(loss=loss.item())

        avg_train_loss = train_loss_epoch / len(train_loader)

        avg_train_loss = train_loss_epoch / len(train_loader) # This is average MSE on normalized values

        # Evaluation
        model.eval()
        val_score_epoch = 0 # Will store unnormalized_rounded_mse for scoring
        
        loop_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False)
        with torch.no_grad():
            for images, region_ids, targets_normalized in loop_val:
                images, region_ids, targets_normalized = images.to(device), region_ids.to(device), targets_normalized.to(device)
                preds_normalized = model(images, region_ids)
                
                # MODIFIED: Use the validation_loss_function for scoring
                current_val_score = validation_loss_function(preds_normalized, targets_normalized, lat_mean, lat_std, lon_mean, lon_std)
                val_score_epoch += current_val_score.item()
                loop_val.set_postfix(val_score=current_val_score.item())
        
        avg_val_score = val_score_epoch / len(val_loader) # This is average unnormalized rounded MSE
        scheduler.step(avg_val_score) # Step scheduler based on validation score
        
        # MODIFIED: Print train loss (normalized MSE) and val score (unnormalized rounded MSE)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss (Norm MSE): {avg_train_loss:.6f} | Val Score (Unnorm/Rounded MSE): {avg_val_score:.6f}")

        if avg_val_score < best_val_loss:
            best_val_loss = avg_val_score
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved model to {save_path} (New best Val Score: {best_val_loss:.6f})")
    
    print(f"Training finished. Best Validation Score: {best_val_loss:.6f}")

# --- Main Execution ---
if __name__ == "__main__":
    # Define Paths
    original_train_labels_path = 'labels_train.csv' # Original labels
    original_val_labels_path = 'labels_val.csv'   # Original labels
    train_region_preds_path = 'train_predictions_region_all.csv' # Region predictions
    val_region_preds_path = 'val_predictions_region_all.csv'     # Region predictions
    train_image_dir = 'images_train/' 
    val_image_dir = 'images_val/'   

    # Load and Merge Data (similar to train_a.py)
    try:
        print(f"Loading original labels: {original_train_labels_path}, {original_val_labels_path}")
        train_labels_df = pd.read_csv(original_train_labels_path)
        val_labels_df = pd.read_csv(original_val_labels_path)

        print(f"Loading region predictions: {train_region_preds_path}, {val_region_preds_path}")
        train_preds_df = pd.read_csv(train_region_preds_path)
        val_preds_df = pd.read_csv(val_region_preds_path)

        # Validate required columns in original labels
        required_label_cols = ['filename', 'latitude', 'longitude']
        if not all(col in train_labels_df.columns for col in required_label_cols):
            raise ValueError(f"Missing required columns in {original_train_labels_path}. Need: {required_label_cols}")
        if not all(col in val_labels_df.columns for col in required_label_cols):
             raise ValueError(f"Missing required columns in {original_val_labels_path}. Need: {required_label_cols}")
        
        # Validate required columns in predictions
        required_pred_cols = ['Region_ID'] 
        if not all(col in train_preds_df.columns for col in required_pred_cols):
            raise ValueError(f"Missing required columns in {train_region_preds_path}. Need: {required_pred_cols}")
        if not all(col in val_preds_df.columns for col in required_pred_cols):
            raise ValueError(f"Missing required columns in {val_region_preds_path}. Need: {required_pred_cols}")

        # Rename predicted Region_ID to avoid clash
        train_preds_df = train_preds_df.rename(columns={'Region_ID': 'pred_Region_ID'})
        val_preds_df = val_preds_df.rename(columns={'Region_ID': 'pred_Region_ID'})

        # Merge based on index (assuming row order corresponds)
        train_labels_df = train_labels_df.reset_index(drop=True)
        val_labels_df = val_labels_df.reset_index(drop=True)
        train_preds_df = train_preds_df.reset_index(drop=True)
        val_preds_df = val_preds_df.reset_index(drop=True)
        
        train_merged_df = train_labels_df.join(train_preds_df['pred_Region_ID'])
        val_merged_df = val_labels_df.join(val_preds_df['pred_Region_ID'])
        
        # Check for NaNs after merge
        if train_merged_df['pred_Region_ID'].isnull().any():
             print("Warning: NaNs found after merging train labels and predictions. Check alignment/lengths.")
             train_merged_df.dropna(subset=['pred_Region_ID'], inplace=True)
        if val_merged_df['pred_Region_ID'].isnull().any():
             print("Warning: NaNs found after merging val labels and predictions. Check alignment/lengths.")
             val_merged_df.dropna(subset=['pred_Region_ID'], inplace=True)
             
        # Convert region ID to int
        train_merged_df['pred_Region_ID'] = train_merged_df['pred_Region_ID'].astype(int)
        val_merged_df['pred_Region_ID'] = val_merged_df['pred_Region_ID'].astype(int)

        print("Data loading and merging complete.")
        print(f"Train samples: {len(train_merged_df)}, Val samples: {len(val_merged_df)}")

    except FileNotFoundError as e:
        print(f"Error: CSV file not found: {e}. Please check paths.")
        exit()
    except ValueError as e:
        print(f"Error: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during data loading/merging: {e}")
        exit()

    # Data Preprocessing (using merged dataframes)
    # Outlier removal for latitude and longitude from training data
    lat_lower = train_merged_df['latitude'].quantile(0.01)
    lat_upper = train_merged_df['latitude'].quantile(0.99)
    lon_lower = train_merged_df['longitude'].quantile(0.01)
    lon_upper = train_merged_df['longitude'].quantile(0.99)

    # Apply outlier filtering to the merged training dataframe
    train_df = train_merged_df[
        (train_merged_df['latitude'].between(lat_lower, lat_upper)) &
        (train_merged_df['longitude'].between(lon_lower, lon_upper))
    ].copy() # Use .copy() to avoid SettingWithCopyWarning
    print(f"Train data after outlier removal: {train_df.shape}")

    # Apply row removal to the merged validation dataframe
    rows_to_remove_val = [95, 145, 146, 158, 159, 160, 161] 
    # Check if indices exist before dropping
    valid_indices_to_drop = val_merged_df.index.intersection(rows_to_remove_val)
    val_df = val_merged_df.drop(index=valid_indices_to_drop).reset_index(drop=True).copy()
    print(f"Validation data after potential row removal: {val_df.shape}")

    # Normalization parameters from the filtered training set (train_df)
    lat_mean = train_df['latitude'].mean()
    lat_std = train_df['latitude'].std()
    lon_mean = train_df['longitude'].mean()
    lon_std = train_df['longitude'].std()
    print(f"Normalization params: LatMean={lat_mean:.4f}, LatStd={lat_std:.4f}, LonMean={lon_mean:.4f}, LonStd={lon_std:.4f}")

    # Transforms (EfficientNet typically uses 224x224 or other sizes depending on version)
    # For b0, 224x224 is common.
    # image_size = models.efficientnet_b0().default_image_size # Get default for chosen model
    # Corrected way to get image size for the selected MODEL_VERSION
    try:
        # Dynamically get the model class from torchvision.models
        model_class = getattr(models, MODEL_VERSION)
        # Instantiate the model (without weights to just get info, or with default weights)
        # Using pretrained=True might download weights if not present, so just instantiating is safer if only for default_image_size
        # However, some models might only define default_image_size on instances with weights.
        # Let's assume weights='IMAGENET1K_V1' is acceptable for this check.
        image_size = model_class(weights='IMAGENET1K_V1').default_image_size
        if isinstance(image_size, tuple) and len(image_size) >= 2 : # default_image_size can be (H,W) or int
            image_size = image_size[0] # take height
    except AttributeError:
        print(f"Warning: default_image_size not found for {MODEL_VERSION} or model class not found. Defaulting to 224.")
        image_size = 224
    except Exception as e:
        print(f"Error getting default_image_size for {MODEL_VERSION}: {e}. Defaulting to 224.")
        image_size = 224

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and DataLoaders
    train_dataset = LatLongDataset(dataframe=train_df, image_dir=train_image_dir, transform=train_transform,
                                   lat_mean=lat_mean, lat_std=lat_std, lon_mean=lon_mean, lon_std=lon_std)
    val_dataset = LatLongDataset(dataframe=val_df, image_dir=val_image_dir, transform=val_transform,
                                 lat_mean=lat_mean, lat_std=lat_std, lon_mean=lon_mean, lon_std=lon_std)
    
    # Using num_workers=0 for compatibility, especially on Windows, can be increased if stable
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Model, Criterion, Optimizer
    model = LatLongModelWithRegion(version=MODEL_VERSION, pretrained=True, 
                                   num_regions=NUM_REGIONS, embedding_dim=EMBEDDING_DIM).to(device)
    # Criterion for backpropagation (MSE on normalized values)
    criterion = nn.MSELoss() 
    # Validation loss function (unnormalized, rounded MSE) is defined above
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training for Lat/Long prediction using {MODEL_VERSION} with region embeddings and attention...")
    print(f"Device: {device}, Batch Size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")
    print(f"Image Size: {image_size}x{image_size}")
    print(f"Model will be saved to: {MODEL_SAVE_PATH}")

    train_model_latlong(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion, # Pass MSELoss for backprop
        validation_loss_function=unnormalized_rounded_mse_loss, # Pass unnorm/rounded MSE for validation score
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        save_path=MODEL_SAVE_PATH,
        lat_mean=lat_mean, lat_std=lat_std, lon_mean=lon_mean, lon_std=lon_std
    )
