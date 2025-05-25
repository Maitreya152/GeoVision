# Sub-Project: Latitude and Longitude Prediction from Images

This section of the project aims to predict the geographical coordinates (latitude and longitude) of a location based on an input image.

## Model Overview

The prediction of continuous latitude and longitude values is treated as a regression problem, addressed using a Convolutional Neural Network (CNN).
- **Core Architecture**: A CNN, often a pre-trained model like ResNet, EfficientNet, or VGG, is used as a feature extractor.
- **Fine-Tuning for Regression**: The chosen CNN is fine-tuned. Its final classification layer is replaced with a regression head designed to output two continuous values: one for latitude and one for longitude.
- **Output Representation**: The model outputs two scalar values. These might be normalized during training and then de-normalized to obtain the actual coordinate values.

## Pre-processing and Data Handling

Careful data preparation is essential for this task:
- **Image Resizing**: Images are resized to a consistent input size (e.g., 224x224, 299x299) compatible with the CNN architecture.
- **Data Augmentation (Training)**: To improve model generalization and prevent overfitting, various augmentations can be applied to training images, such as:
    - Random flips (if geographically appropriate)
    - Minor rotations, scaling, and translations
    - Adjustments to brightness, contrast, and color.
- **Normalization**: Image pixel values are normalized, typically using ImageNet statistics if a pre-trained model is used.
- **Coordinate Normalization**: Latitude and longitude values in the training data are often normalized (e.g., to a [0, 1] or [-1, 1] range) by subtracting the mean and dividing by the standard deviation of the training set coordinates. This helps with stable training.

## Training Process
- **Loss Function**: Mean Squared Error (MSE) is a common loss function for regressing latitude and longitude. Other options include Mean Absolute Error (MAE) or Haversine distance loss (if coordinates are not normalized or can be de-normalized within the loss calculation).
- **Optimizer**: Adam or SGD with momentum are frequently used optimizers.
- **Learning Rate**: A well-chosen learning rate, possibly with a decay schedule (e.g., ReduceLROnPlateau, step decay), is important.
- **Evaluation Metrics**: The primary metric is often the mean Haversine distance (great-circle distance) between predicted and true coordinates. MAE on latitude and longitude (in degrees) can also be reported.

## Key Considerations and Innovations
- **Geographical Feature Learning**: The model learns to associate visual patterns in images with specific geographical locations.
- **Normalization Strategy**: The method of normalizing and de-normalizing coordinates can significantly impact performance and the interpretability of errors.
- **Spatial Priors**: For more advanced approaches, incorporating spatial priors or hierarchical models (e.g., first predict a coarse region, then refine coordinates) could be considered.
- **Uncertainty Estimation**: Some models might be adapted to predict not just the coordinates but also an uncertainty estimate for the prediction.

### Link to models: 
https://iiithydresearch-my.sharepoint.com/:f:/g/personal/maitreya_chitale_research_iiit_ac_in/Eo_Isu8_IexJiPe3_YtJpscBcWrQN6KMuWvvGDHEMlU1Qg?e=qIiVnA
