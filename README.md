# Handwritten Digits Generator using DCGAN

This repository contains a project for generating handwritten digits using a Deep Convolutional Generative Adversarial Network (DCGAN). The project demonstrates the power of GANs in generating new, realistic data from a learned distribution of the MNIST dataset.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)

## Introduction

The Handwritten Digits Generator is a deep learning project leveraging a DCGAN to create new images of handwritten digits. DCGANs consist of two neural networks: a generator that creates fake images and a discriminator that distinguishes between real and fake images. The generator improves iteratively by learning to produce images that the discriminator cannot distinguish from real ones.

This project focuses on training a DCGAN using the MNIST dataset of handwritten digits to generate new, high-quality digit images.

## Dataset

The dataset used for training is the MNIST dataset, which contains grayscale images of digits (0–9).

- Source: The dataset is loaded using TensorFlow's Keras API.
- Dimensions: 28x28 grayscale images with labels for digits 0–9.

## Installation

To set up the project, follow these steps:

1. Clone the repository to your local machine:

```
   git clone https://github.com/srijosh/Handwritten-Digits-Generator-using-DCGAN.git
```

2. Navigate to the project directory:
   cd Handwritten-Digits-Generator-using-DCGAN

3. Install the required dependencies:

```
   pip install tensorflow imageio matplotlib numpy pillow
```

### Requirements

Python 3.x
TensorFlow
Matplotlib
NumPy
PIL (Pillow)
ImageIO

## Usage

1. Clone the repository and navigate to the directory.
2. Open and run the Jupyter Notebook:
   jupyter notebook HandwrittenDigitsGenerator.ipynb

## Model

1. Generator
   The generator is designed to take random noise as input and produce 28x28 grayscale images. It uses transposed convolutional layers to upsample the noise to the desired image dimensions.

2. Discriminator
   The discriminator is a convolutional neural network that classifies input images as real or fake. It uses convolutional layers with LeakyReLU activation and a single neuron output with sigmoid activation for binary classification.

### Training Procedure

1.  Adversarial Loss:

- The generator is trained to maximize the discriminator's error.
- The discriminator is trained to correctly classify real vs. generated images.

2. Training Steps:

- Normalize the images to the range [-1, 1].
- Generate random noise as input to the generator.
- Train the discriminator on real and fake images.
- Update the generator based on the discriminator's feedback.
