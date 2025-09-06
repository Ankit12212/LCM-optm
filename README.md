# Lung and Colon Cancer Detection

This project uses deep learning to classify images of lung and colon tissue into cancerous and non-cancerous categories. The workflow includes data preprocessing, model training, evaluation, and visualization.

## Dataset

The dataset used is the [Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) from Kaggle. It contains thousands of labeled images for lung and colon cancer types.

- Download link: https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images

## Project Structure

- `notebook.ipynb`: Implements a custom Convolutional Neural Network (CNN) from scratch for multi-class classification of lung and colon cancer images. Includes data exploration, model building, training, learning rate tuning, and evaluation.
- `transfer_learning.ipynb`: Uses transfer learning with ResNet50 as a feature extractor, followed by fine-tuning. Includes preprocessing, model construction, training, fine-tuning, and evaluation.
- `inference.ipynb`: Demonstrates how to perform inference using the trained `best.keras` model on a single sample image. Includes code for loading the model, preparing the image, and displaying the predicted class and probability.
- `data_config.yaml`: Configuration file for dataset paths and parameters.
- `best.keras`: Saved best model checkpoint, ignored by gitignore.
- `lung_colon_image_set/`: Folder containing the image dataset, ignored by gitignore

## Workflow Overview

Both notebooks address the same classification problem and use the same dataset, but differ in modeling approach:

### 1. Custom CNN Approach (`notebook.ipynb`)

- **Data Loading & Exploration:** Loads and visualizes images, splits into train/validation sets.
- **Model Architecture:** Builds a CNN from scratch using Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, BatchNormalization, and Dropout layers.
- **Training:** Trains with different learning rates, uses callbacks for early stopping and best model saving.
- **Evaluation:** Plots loss curves, computes classification report and confusion matrix.
- **Prediction:** Includes code for single image prediction.

### 2. Transfer Learning Approach (`transfer_learning.ipynb`)

- **Data Loading & Preprocessing:** Loads images, applies ResNet50-specific preprocessing.
- **Feature Extraction:** Uses ResNet50 (pre-trained on ImageNet) as a frozen feature extractor.
- **Custom Head:** Adds new layers for classification on top of ResNet50.
- **Training:** Trains only the custom head initially, then fine-tunes deeper layers of ResNet50.
- **Fine Tuning:** Unfreezes part of ResNet50 and retrains with a lower learning rate.
- **Evaluation:** Plots training/validation curves, evaluates performance after fine-tuning.

## How to Use

1. Download and extract the dataset from Kaggle.
2. Install required Python packages (see notebooks for details).
3. Run either notebook to train and evaluate a model:
   - Use `notebook.ipynb` for a custom CNN approach.
   - Use `transfer_learning.ipynb` for a transfer learning approach with ResNet50.
4. Use `inference.ipynb` to test the trained model (`best.keras`) on new images and obtain predictions.

## References

- [Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
