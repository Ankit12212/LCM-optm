# Lung and Colon Cancer Detection

This project uses deep learning to classify images of lung and colon tissue into cancerous and non-cancerous categories. The workflow includes data preprocessing, model training, evaluation, and visualization.

## Dataset

The dataset used is the [Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) from Kaggle. It contains thousands of labeled images for lung and colon cancer types.

- Download link: https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images

## Project Structure

- `notebook.ipynb`: Main Jupyter notebook for data analysis and model training.
- `data_config.yaml`: Configuration file for dataset paths and parameters.

## Getting Started

1. Download the dataset from Kaggle and extract it to the project folder.
2. Install required Python packages (see notebook for details).
3. Run the notebook to train and evaluate the model.

## Workflow Details (notebook.ipynb Specific)

The workflow in `notebook.ipynb` is organized as follows:

1. **Library Imports**

   - Imports essential libraries: NumPy, Pandas, Matplotlib, PIL, OpenCV, TensorFlow, and Keras for data handling, visualization, and modeling.

2. **Data Extraction**

   - The dataset is provided as `archive.zip`. Extraction code is included (commented out) to extract images into `lung_colon_image_set/`.

3. **Data Exploration & Visualization**

   - Visualizes random images from each lung cancer class (`lung_aca`, `lung_n`, `lung_scc`) using Matplotlib and PIL to understand the dataset.

4. **Dataset Preparation**

   - Loads images using TensorFlow's `image_dataset_from_directory`, inferring labels from folder names.
   - Splits the dataset into training and validation sets (80/20 split), resizes images to 256x256 pixels, and normalizes pixel values to [0, 1].

5. **Model Construction**

   - Defines a function to build a Sequential CNN model with Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, BatchNormalization, and Dropout layers.
   - The model is designed for multi-class classification (3 classes).

6. **Callbacks Setup**

   - Configures EarlyStopping, ReduceLROnPlateau, TensorBoard, and ModelCheckpoint to monitor training, log metrics, and save the best model.

7. **Learning Rate Tuning**

   - Trains the model with different learning rates (`1e-3`, `5e-4`, `1e-4`) and stores training histories.
   - Plots validation loss curves for each learning rate to compare performance.

8. **Model Evaluation**

   - Predicts on the validation set and computes predicted labels.
   - Calculates true labels from the validation set.
   - Generates a classification report and confusion matrix using scikit-learn to assess model performance.

9. **Model Saving**
   - The best model is saved as `best.keras` for future use.

For step-by-step code and outputs, see `notebook.ipynb` in this repository.

## References

- [Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

---
