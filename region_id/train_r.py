import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16
num_epochs = 50
learning_rate = 1e-4

class CustomImageLoader(Dataset):
    def __init__(self, task, dataframe, image_transform, image_dir, data_category="train",
                 lat_mean_val=None, lat_std_val=None, lon_mean_val=None, lon_std_val=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_directory = image_dir
        self.task_type = task
        self.transform = image_transform
        self.lat_mean = lat_mean_val
        self.lat_std = lat_std_val
        self.lon_mean = lon_mean_val
        self.lon_std = lon_std_val
        self.data_type = data_category

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        data_row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.image_directory, data_row['filename'])
        image = None # Initialize image to None
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Attempt to load from an alternative validation path if the primary fails
            alternative_val_path_prefix = "images_val"
            img_path_alt = os.path.join(alternative_val_path_prefix, "images_val", data_row['filename'])
            try:
                image = Image.open(img_path_alt).convert("RGB")
            except FileNotFoundError:
                # If both attempts fail, raise an error.
                raise FileNotFoundError(f"Image file not found at primary path {img_path} or alternative {img_path_alt}")
        
        if image is None: # Should not happen if FileNotFoundError is properly caught
             raise Exception(f"Image could not be loaded for {data_row['filename']}")

        if self.transform:
            image = self.transform(image)

        target = None
        if self.task_type == 'latlong':
            lat = (data_row['latitude'] - self.lat_mean) / self.lat_std
            lon = (data_row['longitude'] - self.lon_mean) / self.lon_std
            target = torch.tensor([lat, lon], dtype=torch.float32)
        if self.task_type == 'angle':
            target = torch.tensor(data_row['angle']%360, dtype=torch.float32)
        if self.task_type == 'region':
            target = torch.tensor(data_row['Region_ID'] - 1, dtype=torch.long)
        
        if target is None:
            raise ValueError("Unknown task {}".format(self.task_type))

        return image, target

class GeoRegionClassifier(nn.Module):
    def __init__(self):
        super(GeoRegionClassifier, self).__init__() # Calling super with class name for style
        
        # --- Define ResNet-based Feature Extractor ---
        # Load a pretrained ResNet101 model
        core_resnet_model_extractor = models.resnet101(weights='IMAGENET1K_V1')
        # Extract all layers except the final fully connected (classification) layer
        resnet_convolutional_layers_list = list(core_resnet_model_extractor.children())[:-1]
        self.feature_extractor_backbone = nn.Sequential() # Initialize sequential container
        # Add each ResNet layer to our sequential feature extractor
        for layer_idx_val, resnet_module_item in enumerate(resnet_convolutional_layers_list):
            self.feature_extractor_backbone.add_module(f"resnet_feature_extraction_layer_{layer_idx_val}", resnet_module_item)

        # --- Define the Classification Head ---
        # This head will take the features from ResNet and map them to the number of regions
        self.classification_head_fc = nn.Sequential() # Initialize another sequential container
        # First fully connected layer (Input: ResNet features, Output: 512)
        self.classification_head_fc.add_module("fully_connected_1", nn.Linear(2048, 512))
        self.classification_head_fc.add_module("activation_1_relu", nn.ReLU())
        self.classification_head_fc.add_module("batchnorm_layer_1", nn.BatchNorm1d(512))
        # Second fully connected layer (Input: 512, Output: 256)
        self.classification_head_fc.add_module("fully_connected_2", nn.Linear(512, 256))
        self.classification_head_fc.add_module("activation_2_relu", nn.ReLU())
        self.classification_head_fc.add_module("batchnorm_layer_2", nn.BatchNorm1d(256))
        # Final output layer (Input: 256, Output: 15 regions)
        self.classification_head_fc.add_module("output_projection", nn.Linear(256, 15))

    def forward(self, input_tensor):
        features = self.feature_extractor_backbone(input_tensor)
        flattened_features = features.view(features.size(0), -1)
        output_logits = self.classification_head_fc(flattened_features)
        return output_logits

def evaluate_predictions(output_logits, ground_truth_labels):
    predicted_labels = torch.argmax(output_logits, dim=1)
    num_correct = (predicted_labels == ground_truth_labels).sum().item()
    return num_correct / len(ground_truth_labels)

def execute_training_cycle(model, train_loader, val_loader,criterion, num_epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    best_validation_accuracy = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        num_train_correct_predictions = 0
        num_train_samples = 0

        progress_bar = tqdm(train_loader, desc="Epoch {}/{} - Training".format(epoch+1, num_epochs), leave=False)
        for images, targets in progress_bar:

            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss =criterion(outputs, targets)
            train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            num_train_correct_predictions += (outputs.argmax(1) == targets).sum().item()
            num_train_samples += targets.size(0)

        train_acc = num_train_correct_predictions / num_train_samples

        model.eval()
        validation_accuracies = []
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                val_accuracy = evaluate_predictions(outputs, targets)
                validation_accuracies.append(val_accuracy)

        mean_validation_accuracy = np.mean(validation_accuracies)
        scheduler.step(mean_validation_accuracy)

        print("Epoch {} \t Train Loss: {:.6f} \t Train Acc: {:.4f} Validation Accuracy: {:.6f}".format(epoch+1, train_loss/len(train_loader), train_acc, mean_validation_accuracy))

        if mean_validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = mean_validation_accuracy
            torch.save(model.state_dict(), "best_region_model_data_val.pt")
            print("Model saved at epoch {}".format(epoch+1))

    print("Best Validation Score: {:.6f}".format(best_validation_accuracy))

def prepare_data_loaders_and_weights(batch_size_for_loader, target_device_for_tensors):
    # Load initial datasets from CSV files
    source_train_records = pd.read_csv("labels_train.csv")
    source_validation_records = pd.read_csv("labels_val.csv")

    # Define quantile boundaries for filtering outliers from training data
    latitude_filter_min = source_train_records['latitude'].quantile(0.01)
    latitude_filter_max = source_train_records['latitude'].quantile(0.99)
    longitude_filter_min = source_train_records['longitude'].quantile(0.01)
    longitude_filter_max = source_train_records['longitude'].quantile(0.99)

    # Apply quantile filtering to training data
    train_records_quant_filtered = source_train_records[
        (source_train_records['latitude'].between(latitude_filter_min, latitude_filter_max)) &
        (source_train_records['longitude'].between(longitude_filter_min, longitude_filter_max))
    ]

    # Specify indices of validation samples to be removed
    validation_rows_to_exclude_indices = [95, 145, 146, 158, 159, 160, 161]
    validation_records_cleaned = source_validation_records.drop(index=validation_rows_to_exclude_indices).reset_index(drop=True)

    # Filter training data for valid Region IDs and reset index
    final_train_records_for_loader = train_records_quant_filtered[(train_records_quant_filtered['Region_ID'] >= 1) & (train_records_quant_filtered['Region_ID'] <= 15)]
    final_train_records_for_loader = final_train_records_for_loader.reset_index(drop=True)
    # Assert that cleaned validation data also has valid Region IDs
    assert set(validation_records_cleaned['Region_ID']).issubset(set(range(1, 16))), "Validation data (after cleaning) contains out-of-range Region_IDs"

    # Image transformations for the training set
    image_processing_pipeline_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Image transformations for the validation set
    image_processing_pipeline_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Combine processed training and validation data for the training loader (if needed, or just use final_train_records_for_loader)
    # For this specific case, the training loader uses a combination, while validation uses its own cleaned set.
    combined_training_val_records = pd.concat([final_train_records_for_loader, validation_records_cleaned], ignore_index=True)
    
    training_set_loader = DataLoader(
        CustomImageLoader(task='region', dataframe=combined_training_val_records, image_transform=image_processing_pipeline_train, image_dir="images_train/images_train/", data_category="train"),
        batch_size=batch_size_for_loader, shuffle=False # Shuffle is typically True for training
    )

    validation_set_loader = DataLoader(
        CustomImageLoader(task='region', dataframe=validation_records_cleaned, image_transform=image_processing_pipeline_val, image_dir="images_val/images_val/", data_category="val"),
        batch_size=batch_size_for_loader, shuffle=False
    )

    # Calculate class weights for handling imbalance
    counts_per_region_class = final_train_records_for_loader['Region_ID'].value_counts()
    class_loss_weighting_factors = 1. / counts_per_region_class.to_numpy()
    class_loss_weighting_factors_tensor = torch.tensor(class_loss_weighting_factors, dtype=torch.float32).to(target_device_for_tensors)
    
    return training_set_loader, validation_set_loader, class_loss_weighting_factors_tensor

if __name__ == "__main__":
    # Setup data pipeline: get data loaders and loss weights
    data_loader_train, data_loader_validation, calculated_loss_weights = prepare_data_loaders_and_weights(batch_size, device)
    
    # Initialize the model and move it to the configured device
    geo_classifier_nn_model = GeoRegionClassifier().to(device)
    # Define the loss function, incorporating the calculated class weights
    training_objective_loss_func = nn.CrossEntropyLoss(weight=calculated_loss_weights)
    
    # Start the training process
    execute_training_cycle(geo_classifier_nn_model, data_loader_train, data_loader_validation, training_objective_loss_func, num_epochs, learning_rate)
