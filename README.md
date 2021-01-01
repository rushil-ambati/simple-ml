# Simple Machine Learning Algorithms

## Classification

A set of synthetic datasets and classification methods, inside the "classification" directory.

### Binary Classification
Inside the "binary" directory:
- `linear_batch.py` contains a simple gradient descent algorithm made from scratch which is used to classify a two cluster, linearly-separable dataset.
- `linear_stochastic.py` uses Keras to create a single-layer (input and output) perceptron model and the Adam optimiser to classify a two cluster, linearly-separable dataset.
- `nonlinear_stochastic.py` uses Keras to create a multilayer perceptron model and the Adam optimiser to classify a two cluster, nonlinearly-separable dataset - more specifically, one cluster is inside another.

### Multiclass Classification
Inside the "multiclass" directory, `nonlinear_stochastic.py` uses Keras to create a single layer perceptron model and the Adam optimiser to classify a five cluster, nonlinearly-separable dataset.

## Perceptron
A set of simple neural network models made from scratch, inside the "perceptron" directory.

### Single Layer
Inside the "singlelayer" directory:
- `perceptron.py` creates and trains a single-layer perceptron.
- `error_plot.py` visualises the loss over training rounds for the above model.

### Multi Layer
Inside the "multilayer" directory, `multilayer.py` creates and trains a multilayer perceptron.
