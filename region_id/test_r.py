import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_FILE_NAME='best_region_model_data_val.pt'
batch_size=16

class ImageDataset(Dataset):
    def __init__(self, df, img_dir, task, transform, data_type="train",
                 lat_mean=None, lat_std=None, lon_mean=None, lon_std=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.task = task
        self.transform = transform
        self.lat_mean = lat_mean
        self.lat_std = lat_std
        self.lon_mean = lon_mean
        self.lon_std = lon_std
        self.data_type = data_type
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            img_path = os.path.join("images_val/", row['filename'])
            image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.task == 'latlong':
            lat = (row['latitude'] - self.lat_mean) / self.lat_std
            lon = (row['longitude'] - self.lon_mean) / self.lon_std
            target = torch.tensor([lat, lon], dtype=torch.float32)
        elif self.task == 'angle':
            mod_360_angle=row['angle']%360
            target = torch.tensor(mod_360_angle, dtype=torch.float32)
        elif self.task == 'region':
            target = torch.tensor(row['Region_ID'] - 1, dtype=torch.long)  # 0-based indexing
        else:
            raise ValueError(f"Unknown task {self.task}")

        return image, target

class RegionClassification(nn.Module):
    def __init__(self):
        super().__init__()
        resnet_model = models.resnet101(weights='IMAGENET1K_V1') 
        self.resnet_feature_extractor = nn.Sequential(*list(resnet_model.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 15)
        )
        
    def forward(self, x):
        x = self.resnet_feature_extractor(x)
        x = x.view(x.size(0), -1) 
        return self.head(x)
    
class TestImageDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.image_filenames = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name

def predict_and_save(model, val_loader, output_csv):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, targets in val_loader:
            print(images, targets)

            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1) + 1 
            all_preds.extend(preds.cpu().numpy()) 
    
    result_df = pd.DataFrame({
        'id': np.arange(len(all_preds)),
        'Region_ID': all_preds,
    })
    
    result_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

def predict_on_test_and_append(model, test_loader, output_csv):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="Predicting on test set"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1) + 1  # Add 1 to match Region_ID
            all_preds.extend(preds.cpu().numpy())

    test_df = pd.DataFrame({
        'Region_ID': all_preds
    })

    val_df_existing = pd.read_csv(output_csv)
    if 'id' in val_df_existing.columns:
        val_df_existing = val_df_existing.drop(columns=['id'])

    # Concatenate val and test
    final_df = pd.concat([val_df_existing, test_df], ignore_index=True)
    final_df.insert(0, 'id', range(len(final_df))) 

    final_df.to_csv(output_csv, index=False)
    print(f"Appended test predictions and saved to {output_csv}")

# --- Data Preparation ---

# Load Data
train_df = pd.read_csv("labels_train.csv")
val_df = pd.read_csv("labels_val.csv")
test_image_dir = "images_test/images_test/"


# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load model
model = RegionClassification()
model.load_state_dict(torch.load("best_region_model_data_combined.pt"))  # <- change filename if needed
model = model.to(device)
model.eval()

# Dataloader
val_dataloader   = DataLoader(ImageDataset(val_df,"images_val/", task='region', transform=transform,data_type="val"), batch_size=batch_size)
test_dataset = TestImageDataset(test_image_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

train_dataloader = DataLoader(
    ImageDataset(train_df, "images_train/", task='region', transform=transform,
                 data_type="train"),
    batch_size=batch_size, shuffle=False
)

# Predict and save results for val set
output_csv="test_predictions_region_all.csv"
# predict_and_save(model, test_loader, output_csv)

# Predict and append results of test set
predict_on_test_and_append(model, test_loader, output_csv)