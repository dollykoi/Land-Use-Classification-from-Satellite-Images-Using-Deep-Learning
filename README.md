# Land Use Classification from Satellite Images

A PyTorch CNN that classifies Sentinel-2 satellite images into 10 land use classes, with an interactive Gradio dashboard for trying the model in a browser. Built as the DSA4050 semester term paper.

## Overview

The goal is to predict the type of land shown in a small satellite image, for example forest, farmland, a residential area or a highway. A convolutional neural network is a natural fit because land use is defined by visual patterns like shapes and textures, and convolutions preserve the spatial structure that a flattened pixel vector would lose.

## Dataset

EuroSAT RGB, downloaded from Kaggle:

https://www.kaggle.com/datasets/waseemalastal/eurosat-rgb-dataset

- 27,000 RGB images, 64 x 64 pixels, from the Sentinel-2 satellite
- 10 classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
- Published by Helber et al. (2019), IEEE JSTARS

## Files

- `EuroSAT_Land_Use_CNN.ipynb` - the full notebook: data download, training, evaluation and the dashboard. Runs end to end in Google Colab.
- `EuroSAT_Land_Use_Term_Paper.docx` - the 2,000 word write-up with figures and references.

## Approach

- 70 / 15 / 15 train, validation and test split with class proportions kept equal
- CNN with three blocks of convolution, batch normalization, ReLU and max pooling (16, 32 and 64 feature maps), then a fully connected layer with 128 units
- Cross entropy loss, mini-batches of 64
- Optimizer comparison: plain SGD, SGD with momentum 0.9, and Adam
- Regularization: dropout 0.5 plus data augmentation (random horizontal flips and 90 degree rotations)
- Two baselines on flattened pixels: a linear softmax model and an ANN with two hidden layers
- 5-fold cross validation of the final configuration

## Results

- Flat baselines reach about 43% test accuracy, the CNNs reach 86%+
- Adam and momentum converge much faster than plain SGD (86.9% and 86.8% vs 66.4%)
- Dropout plus augmentation nearly removes the overfitting gap
- Final model: 86.5% test accuracy
- 5-fold cross validation: mean 81.6%, standard deviation 1.1%

## Interactive dashboard

The last part of the notebook launches a Gradio app with two tabs:

1. Try the model: upload a satellite image or click an example, and get the predicted class with the top three confidence scores
2. How the model was trained: the training curves, optimizer comparison and confusion matrix

In Colab the app starts with a public link, so it can be opened from any browser.

## How to run

1. Open `EuroSAT_Land_Use_CNN.ipynb` in Google Colab
2. Run all cells top to bottom. The first cell downloads the dataset automatically.
3. The dashboard starts at the end with a shareable link.

Main dependencies: torch, torchvision, scikit-learn, gradio, pandas, matplotlib.
