# Image Classification with PyTorch

## Project Overview

This project implements an image classification model using PyTorch. The model classifies images into one of ten categories such as airplane, automobile, bird, cat, and more. The goal is to demonstrate the complete pipeline from data preprocessing, model training, evaluation, to visualization.

---

## Dataset

- **CIFAR-10**: A widely used dataset in machine learning, consisting of 60,000 32x32 color images in 10 different classes (6,000 images per class).
- The dataset is split into 50,000 training images and 10,000 test images.
- At least two classes are required by the project; here, all 10 classes are used for a more comprehensive example.

---

## Data Preprocessing

- Images are converted to PyTorch tensors using `transforms.ToTensor()`.
- Normalization is applied with mean `(0.5, 0.5, 0.5)` and standard deviation `(0.5, 0.5, 0.5)` to standardize pixel values between -1 and 1.
- Data is loaded in batches of 64 for efficient training.

---

## AI Framework Choice

- **PyTorch** is chosen because it provides a flexible and intuitive interface for building deep learning models.
- It offers dynamic computation graphs, which makes debugging and experimentation easier.
- PyTorch has strong community support and extensive prebuilt utilities for vision tasks through `torchvision`.

---

## Model Architecture and Algorithms

- A simple **Convolutional Neural Network (CNN)** is implemented with the following layers:
  - Two convolutional layers (`Conv2d`) each followed by ReLU activation and max pooling.
  - Two fully connected (`Linear`) layers, the last one outputs class scores.
- **Cross-Entropy Loss** is used as the loss function, suitable for multi-class classification problems.
- **Stochastic Gradient Descent (SGD)** optimizer with momentum is used to update model weights during training.

---

## Training Parameters

- **Epochs:** 5  
- **Batch size:** 64  
- **Learning rate:** 0.001  
- **Momentum:** 0.9  
- The training loop iterates over the dataset, performs forward and backward passes, and updates weights to minimize loss.

---

## Evaluation Metrics

- **Accuracy:** Percentage of correctly classified images per epoch during training.  
- **Confusion Matrix:** Shows the number of correct and incorrect predictions broken down by class.  
- **Precision, Recall, F1-Score:** Provided in the classification report for detailed class-wise performance analysis.  
- Visual inspection of sample correct and incorrect predictions is included through plotted images.
