# AlexNet Implementation in TensorFlow 2.0

This project demonstrates the implementation of the **AlexNet** architecture using **TensorFlow 2.0**. AlexNet is one of the pioneering deep convolutional neural networks that revolutionized the field of computer vision, particularly for object classification tasks.

## Project Overview

In this project, we implement the AlexNet model, which was originally designed to perform image classification on the ImageNet dataset. While we use a subset of the ImageNet dataset for demonstration purposes, this notebook focuses on understanding and learning the AlexNet architecture and how to implement it in TensorFlow.

AlexNet was proposed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton and achieved state-of-the-art performance in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012.

## Features of the Project

- **Model Architecture**: AlexNet contains 8 layers: 5 convolutional layers and 3 fully connected layers.
- **Input Size**: The network takes in images of size 227x227x3.
- **Activation Function**: ReLU is used as the activation function after each convolutional and fully connected layer.
- **Pooling**: Max pooling is applied after certain convolutional layers to down-sample the feature maps.
- **Regularization**: Dropout is used to prevent overfitting, and Local Response Normalization (LRN) is applied after the ReLU activations.

## Key Components

### 1. Convolutional Layers
The convolutional layers in AlexNet extract features from the input images. In our implementation:
- Layer 1 uses 96 filters of size 11x11 with stride 4.
- Layer 2 uses 256 filters of size 5x5 with stride 1.
- Layers 3, 4, and 5 each use 384, 384, and 256 filters of size 3x3 respectively.

### 2. Pooling Layers
Max pooling layers are applied after some convolutional layers to reduce the spatial dimensions of the feature maps.

### 3. Fully Connected Layers
The fully connected layers form the classifier part of AlexNet. They output probabilities for each of the classes in the dataset.
- **FC1**: 4096 neurons
- **FC2**: 4096 neurons
- **FC3**: Output layer with the number of neurons equal to the number of classes (e.g., 1000 for ImageNet).

### 4. Dropout and LRN
- Dropout is applied in the fully connected layers with a dropout rate of 50% to prevent overfitting.
- Local Response Normalization (LRN) is used to normalize the output of the activation function in certain layers.

### 5. Training Setup
- **Optimizer**: We use the Adam optimizer for training.
- **Loss Function**: Categorical cross-entropy is employed as the loss function.
- **Accuracy**: The notebook evaluates the model's accuracy on a subset of the ImageNet dataset.

## Dataset

For this implementation, we used a small subset of the **ImageNet** dataset. AlexNet was originally designed to classify 1000 different object categories. The notebook demonstrates the model's performance on a simplified classification task using two classes: `bike` and `ship`.

## Results

While the model achieves reasonable accuracy on this smaller subset, better results would require training on the full ImageNet dataset with a larger number of epochs.

## Instructions for Running the Project

### Requirements

To run this project, you will need the following libraries:

- TensorFlow 2.0+
- NumPy
- Matplotlib

You can install the required packages using:

```bash
pip install tensorflow numpy matplotlib
```

### Running the Notebook

1. Clone the repository.
   ```bash
   git clone https://github.com/yourusername/alexnet-implementation.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook and run the cells:
   ```bash
   jupyter notebook alexnet-architecture-a-complete-guide.ipynb
   ```

### Model Architecture Visualization

The project also provides a visual representation of the AlexNet architecture, including the input size at each layer and the filters used.

## Future Work

For future improvements, the following can be considered:
- Training on the full **ImageNet** dataset.
- Fine-tuning the model by adjusting the learning rate and experimenting with different optimizers.
- Implementing additional image augmentation techniques to improve generalization.
