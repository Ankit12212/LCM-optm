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

   - The notebook starts by importing essential libraries: NumPy, Pandas, Matplotlib, PIL, OpenCV, TensorFlow, and Keras.

2. **Data Extraction**

   - The dataset is provided as `archive.zip`. Extraction code is included but commented out; users can uncomment to extract the data into `lung_colon_image_set/`.

3. **Data Exploration & Visualization**

   - The notebook visualizes random images from each lung cancer class (`lung_aca`, `lung_n`, `lung_scc`) using Matplotlib and PIL to help users understand the dataset.

4. **Dataset Preparation**

   - TensorFlow's `image_dataset_from_directory` is used to load images and infer labels from folder names.
   - The dataset is split into training and validation sets (80/20 split), with images resized to 256x256 pixels.
   - Images are normalized to the [0, 1] range using a custom function and TensorFlow's data pipeline.

5. **Model Construction**

   - A Sequential CNN model is built with multiple Conv2D and MaxPooling2D layers, followed by GlobalAveragePooling2D and Dense layers.
   - BatchNormalization and Dropout are used for regularization.

6. **Model Compilation**

   - The model is compiled with `sparse_categorical_crossentropy` loss, Adam optimizer, and accuracy metric.

7. **Callbacks Setup**

   - EarlyStopping, ReduceLROnPlateau, TensorBoard, and ModelCheckpoint are configured to monitor training and validation performance, log metrics, and save the best model.

8. **Model Training**

   - The model is trained on the training set and validated on the validation set for 10 epochs.
   - Training progress and metrics are displayed in the notebook output.

9. **Results Visualization**

   - Training and validation accuracy are plotted using Pandas and Matplotlib to visualize model performance over epochs.

10. **Model Saving**
    - The best model is saved as `best.keras` for future use.

For step-by-step code and outputs, see `notebook.ipynb` in this repository.

## References

- [Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

---
