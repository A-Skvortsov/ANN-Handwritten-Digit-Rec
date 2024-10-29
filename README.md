# ANN-Handwritten-Digit-rec

A small artificial neural network (multi-layer perceptron) built from scratch. The ANN itself is a very general structure and can probably be used for a lot of things, here I train it with the task of handwritten digit recognition using the MNIST dataset from Keras. So far have achieved ~23\% accuracy on the test dataset. T^T

Implements backpropogation, reLU as the activation function. Layout of the mlp can be adjusted by simply changing the MLP_layout variable in nn.py and initializing a new mlp with it. I set it to have 784 neurons in the first layer (corresponding to the 28x28 pixel size of MNIST images), 10 neurons in the last (1 for each possible digit) and the hidden layers are arbitrary.

First venture with ANNs. s/o Andrej Karpathy.