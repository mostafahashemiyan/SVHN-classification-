# SVHN Classification Project

## Overview

This project aims to classify images of digits from the [SVHN dataset](http://ufldl.stanford.edu/housenumbers/), a real-world dataset obtained from house numbers in Google Street View images. The classification task involves building, training, and evaluating both a Multi-Layer Perceptron (MLP) and a Convolutional Neural Network (CNN) to identify digits (0-9) from grayscale images.

## Dataset

The SVHN dataset consists of over 600,000 digit images with labels ranging from 0 to 9. The dataset is split into:

- **Training Set**: Used to train the models.
- **Validation Set**: Used during training to monitor and tune model performance.
- **Test Set**: Used for final evaluation of model performance.

The dataset is preprocessed by converting the images to grayscale and normalizing the pixel values.

## Project Workflow

### 1. Data Preprocessing

- Loaded the dataset and extracted images and labels.
- Converted the images to grayscale to reduce input complexity.
- Visualized random samples from the dataset to confirm preprocessing steps.

### 2. Multi-Layer Perceptron (MLP) Classifier

- Built an MLP model using the Keras Sequential API.
- Model architecture:
  - Input layer: Flattened grayscale image.
  - Hidden layers: 3 fully connected layers with ReLU activations.
  - Output layer: Fully connected layer with softmax activation for 10 classes.
- Trained the model for up to 30 epochs with early stopping and ModelCheckpoint callbacks.
- Visualized learning curves (loss and accuracy) for training and validation sets.
- Evaluated the model on the test set and achieved satisfactory performance.

### 3. Convolutional Neural Network (CNN) Classifier

- Built a CNN model using the Keras Sequential API.
- Model architecture:
  - Convolutional layers: 3 Conv2D layers with ReLU activation and BatchNormalization.
  - Pooling: MaxPooling2D layers for spatial reduction.
  - Dropout: Applied dropout for regularization.
  - Fully connected layers: 2 Dense layers for classification.
- Trained the model for up to 30 epochs with early stopping and ModelCheckpoint callbacks.
- Visualized learning curves (loss and accuracy) for training and validation sets.
- Evaluated the model on the test set, achieving better performance than the MLP model with fewer parameters.

### 4. Model Predictions

- Loaded the best saved weights for both models.
- Selected 5 random test images and visualized the true labels.
- Displayed predictive distributions for both models as bar charts alongside each image.
- Compared MLP and CNN predictions qualitatively.

## Results

- The CNN model outperformed the MLP model in terms of both test accuracy and qualitative performance on randomly selected samples.
- The CNN achieved better generalization with fewer trainable parameters, demonstrating the advantage of convolutional layers for image classification tasks.

## How to Run the Code

1. Clone the repository.
2. Ensure you have the following dependencies installed:
   - Python 3.x
   - TensorFlow
   - NumPy
   - Matplotlib
   - SciPy (for loading `.mat` files)
3. Place the SVHN dataset files (`train_32x32.mat` and `test_32x32.mat`) in a `data/` directory.
4. Run the Python script step by step or as a complete workflow:
   - Data preprocessing
   - Training the MLP
   - Training the CNN
   - Generating predictions
5. View plots and logs to analyze the model performance.



## Future Improvements

- Use data augmentation to improve generalization.
- Experiment with deeper CNN architectures or transfer learning.
- Implement additional metrics like F1-score for performance evaluation.

## References

- Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, and A. Y. Ng. *Reading Digits in Natural Images with Unsupervised Feature Learning*. NIPS Workshop on Deep Learning and Unsupervised Feature Learning, 2011.



