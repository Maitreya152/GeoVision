# Sub-Project: Geographical Region Classification

This part of the project focuses on classifying images into one of 15 distinct geographical regions based on their visual content.

## Model Architecture

A Convolutional Neural Network (CNN) is employed for this image classification task.
- **Base Model**: ResNet-101, pre-trained on ImageNet, serves as the foundational architecture, providing robust learned visual features.
- **Fine-Tuning Strategy**: The pre-trained ResNet-101 is adapted and fine-tuned specifically for geographical region classification. This involves replacing its original classification head.
- **Custom Classification Head**: The new head comprises two fully connected layers, each followed by ReLU activation and Batch Normalization. A final linear layer then outputs logits corresponding to the 15 target geographical regions.

## Pre-processing Techniques

Rigorous pre-processing is applied to optimize model input:
- **Image Resizing**: All input images are uniformly resized to 224x224 pixels.
- **Training Data Augmentation**: To enhance model robustness and prevent overfitting, the following augmentations are applied to the training dataset:
    - Random Horizontal Flips
    - Random Rotations (up to 15 degrees)
    - Color Jitter (adjustments to brightness, contrast, saturation, and hue)
- **Image Normalization**: Pixel values are normalized using the standard ImageNet mean and standard deviation.
- **Data Cleaning & Outlier Removal**: The initial data loading phase includes filtering potential outliers based on latitude and longitude quantiles. Specific known problematic samples are also explicitly removed from the validation set.

## Training Configuration
- **Loss Function**: Cross-Entropy Loss is utilized. Class weights are incorporated to address potential imbalances in the distribution of images across different regions.
- **Optimizer**: The Adam optimization algorithm is used for model training.
- **Learning Rate Scheduling**: A ReduceLROnPlateau learning rate scheduler dynamically adjusts the learning rate based on validation accuracy improvements.
- **Training Duration**: The model is typically trained for a set number of epochs (e.g., 50).
- **Batching**: Training and validation are performed using a defined batch size (e.g., 16).

## Key Methodological Aspects
- **Transfer Learning**: The use of a pre-trained ResNet-101 model accelerates training and improves performance by leveraging features learned from a large-scale dataset.
- **Weighted Loss for Imbalance**: Applying weights to the loss function ensures that regions with fewer samples are adequately learned during the training process.
- **Systematic Data Cleaning**: The proactive removal of outliers and identified problematic data points contributes to a more stable and effective training outcome.

### Link to models: 
https://iiithydresearch-my.sharepoint.com/:f:/g/personal/maitreya_chitale_research_iiit_ac_in/Eo_Isu8_IexJiPe3_YtJpscBcWrQN6KMuWvvGDHEMlU1Qg?e=qIiVnA