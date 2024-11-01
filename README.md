# ANN-Handwritten-Digit-rec

A small artificial neural network (multi-layer perceptron) built from scratch. The ANN itself is a very general structure and can probably be used for a lot of things, here I train it with the task of handwritten digit recognition using the MNIST dataset from Keras. So far have achieved ~23\% accuracy on the test dataset. T^T

Characteristics:
- Uses stochastic gradient descent (backprop after each training example)
- ReLU as activation function
- Mean-squared loss function
- layout of the mlp can be adjusted w MLP_layout variable in nn.py
	- initial layer requires 784 neurons (one per input pixel)
	- final layer requires 10 neurons (1 for each possible digit)
	- hidden layers are arbitrary, can be played with

First venture with ANNs. s/o Andrej Karpathy.