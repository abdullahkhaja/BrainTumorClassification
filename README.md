# Brain Tumor Classification

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) This project implements a deep learning model to classify brain MRI scans into different tumor types using PyTorch. The dataset used is the "Brain Tumor MRI Scans" from Kaggle.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Visualization](#visualization)
- [License](#license)

## Installation
Make sure to install the necessary packages. You can use the following command:
```bash
pip install torch torchvision kagglehub matplotlib seaborn tqdm
```

## Usage
You can run this project in Google Colab. The dataset can be downloaded using the `kagglehub` library:
```python
import kagglehub
path = kagglehub.dataset_download("rm1000/brain-tumor-mri-scans")
print("Path to dataset files:", path)
```

## Dataset
The dataset contains MRI scans of brain tumors classified into four categories: **Glioma**, **Healthy**, **Meningioma**, and **Pituitary**. The dataset is split into training, validation, and test sets.

## Model Architecture
The model is a compact version of the VGG architecture, designed for efficient classification of MRI images. It consists of convolutional layers followed by a fully connected layer to output class predictions.
### Code Example
```python
import torch
import torch.nn as nn

class BrainTumorClassifier(nn.Module):
    def __init__(self):
        super(BrainTumorClassifier, self).__init__()
        # Define model layers...

    def forward(self, x):
        # Define forward pass...
```

## Training
The model is trained using Cross Entropy Loss and the Adam optimizer. The training process includes logging the training and validation loss as well as accuracy for each epoch.
### Training Code Example
```python
for epoch in range(num_epochs):
    # Training loop...
    print(f'Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}')
```

## Evaluation
After training, the model is evaluated on the test set, and metrics such as loss, accuracy, F1 score, and a confusion matrix are computed.
### Evaluation Code Example
```python
from sklearn.metrics import classification_report, confusion_matrix

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_loader)
y_true, y_pred = [], []
for images, labels in test_loader:
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    y_true.extend(labels.numpy())
    y_pred.extend(predicted.numpy())

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
```

## Results
After training for 10 epochs, the model achieves a test accuracy of approximately **94.12%**.
### Sample Output
```
Test Loss: 0.2610, Test Accuracy: 0.9412
```

## Visualization
The project includes functions to visualize the class distribution, sample images from each class, and predictions made by the model.
### Example Visualization Code
```python
import matplotlib.pyplot as plt

def show_sample_images(dataset, class_names):
    # Function to show sample images
    for i in range(len(class_names)):
        plt.imshow(dataset[i][0].numpy().transpose((1, 2, 0)))
        plt.title(class_names[i])
        plt.axis('off')
        plt.show()
```

## License
This project is licensed under the [MIT License](LICENSE).
