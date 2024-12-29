# ANN-Handwritten-Digit-Rec

A small artificial neural network built from scratch to take on the handwritten digit recognition task given by the MNIST database. So far have achieved 95.3% accuracy on the full 10000 samples of the test dataset. This was achieved with the most basic ANN characteristics (with the exception of that mentioned in the 'more nuanced details' section), constant learning rate and no convolution.

ANN Characteristics:
- Dense/fully connected
- 2 Hidden layers, each with 100 neurons
- Uses stochastic gradient descent with batch size of 1 (i.e. ANN updates after each training example)
- Learning rate of 0.01
- LReLU as activation function with slope parameter of 0.1 for negative values (ALPHA variable in nn.py)

More Nuianced Details:
- Cross-entropy loss function, softmax for output layer
- He Kaiming initialization for the weights, biases initialized to 0

First venture with ANNs. s/o Andrej Karpathy.
