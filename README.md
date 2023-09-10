# Mouse Neuron Activity Prediction Project

## Project Overview

This project aimed to predict stimulus or image categories from mouse neural activity. It utilized spontaneous stimulus
responses to normalize brain activity. The primary focus was on analyzing recordings from a single mouse, but the
framework can potentially be extended to work with data from multiple mice, provided certain adjustments are made, such
as handling the hyperparameter selection for the network and voxelization method.
Project Status

In the previous version of this project, an attempt was made to perform voxelization, but a critical bug was discovered
during the revisiting process, which invalidated all previous results. The issue arose because neurons were measured on
a grid, and adjustments to the grid led to sparse grids instead of the intended dense grids. This problem must be
addressed before extending the project to multiple mice datasets.
Dataset & Classes

The project used the dataset provided by Stringer, Pachitariu et al. 2018b. This dataset includes neural activity
recordings from mice exposed to various visual stimuli. The goal was to classify neural responses based on the images
the mice were looking at.
Project History

This project originally started as a part of the Introduction to Computational Neuroscience course at the University of
Tartu in 2019. It has been revisited and updated in 2023.
Installation

### Setup

- Create a Conda environment:

`conda create -n neuroscience python=3.10.12`

- Install the required dependencies:

`pip install -r requirements.txt`

### Reproduce

To reproduce the project results, follow these steps:

- Download the required data:

`python3 download_data.py`

- Run the main script to execute the models:

`python3 main.py`

### Results & Discussion

In the current state of the project, both Convolutional and Linear neural networks produce results similar to the most
frequent class. This behavior is attributed to class imbalance within the dataset. Attempts were made to address this
issue, but the results remained unsatisfactory.

Given the current limitations, further improvements and experimentation are needed to develop more accurate models for
classifying mouse neural activity based on stimulus images.