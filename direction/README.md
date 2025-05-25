# Sub-Project: Image Orientation Angle Prediction

This component of the project is dedicated to predicting the orientation angle from images. The goal is to determine the viewing direction or angle from which an image was captured.

## Model Approach

A Convolutional Neural Network (CNN) forms the basis for predicting the orientation angle.
- **Model Type**: Typically, a regression model adapted from a standard CNN architecture (e.g., ResNet, VGG, EfficientNet) is used.
- **Training Strategy**: The model can be trained from scratch or, more commonly, a pre-trained CNN (on ImageNet) is fine-tuned for this regression task. The final layer(s) are modified to output a continuous angle value (or parameters like sine/cosine of the angle).
- **Output**: The model predicts the angle, often normalized (e.g., 0-360 degrees or radians). Sometimes, predicting sine and cosine of the angle can be more stable for circular quantities.

## Pre-processing Steps

Standard image pre-processing techniques are applied:
- **Resizing**: Images are resized to a fixed size (e.g., 224x224 or 256x256 pixels) suitable for the CNN.
- **Augmentation (Training)**: To improve generalization, training images might undergo augmentations such as:
    - Minor random rotations (carefully chosen not to obscure the true angle)
    - Flips (if appropriate for the dataset and angle definition)
    - Color jitter and brightness/contrast adjustments.
- **Normalization**: Pixel values are normalized using dataset-specific or standard (e.g., ImageNet) mean and standard deviation.
- **Angle Normalization**: Target angles are consistently normalized (e.g., to a 0-360 degree range or -pi to pi radians).

## Training Details
- **Loss Function**: For angle regression, common loss functions include Mean Squared Error (MSE) or Mean Absolute Error (MAE). If predicting sine/cosine, a combination of MSE losses on both components might be used.
- **Optimizer**: Adam or SGD with momentum are common choices.
- **Learning Rate**: A suitable learning rate is chosen, often with a scheduler (e.g., ReduceLROnPlateau or cosine annealing).
- **Metrics**: Evaluation metrics could include MAE on the angle, or accuracy within a certain tolerance (e.g., percentage of predictions within +/- 10 degrees of the true angle).

## Potential Innovations & Considerations
- **Cyclical Nature of Angles**: Special care is taken for the cyclical nature of angles (e.g., 0 and 360 degrees are the same). This can be handled by predicting `sin(angle)` and `cos(angle)` or using specialized cyclical loss functions.
- **Feature Extraction**: Leveraging strong feature extractors from pre-trained models is key.
- **Robustness to Visual Variations**: The model should be robust to variations in lighting, weather, and minor occlusions that don't fundamentally change the scene's orientation.

### Link to models: 
https://iiithydresearch-my.sharepoint.com/:f:/g/personal/maitreya_chitale_research_iiit_ac_in/Eo_Isu8_IexJiPe3_YtJpscBcWrQN6KMuWvvGDHEMlU1Qg?e=qIiVnA
