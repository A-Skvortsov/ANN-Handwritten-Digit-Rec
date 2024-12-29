# ANN-Handwritten-Digit-Rec

A small artificial neural network built from scratch to take on the handwritten digit recognition task given by the MNIST database. So far have achieved 95.3% accuracy on the full 10000 samples of the test dataset. This was achieved with the most basic ANN characteristics (with the exception of that mentioned in the 'more nuanced details' section), constant learning rate and no convolution.

###### ANN Characteristics:
- Dense/fully connected
- 2 Hidden layers, each with 100 neurons
- Uses stochastic gradient descent with batch size of 1 (i.e. ANN updates after each training example)
- Learning rate of 0.01
- LReLU as activation function with slope parameter of 0.1 for negative values (ALPHA variable in nn.py)

###### More Nuianced Details:
- Cross-entropy loss function, softmax for output layer
- He Kaiming initialization for the weights, biases initialized to 0

###### File Organization:
- first_nn.ipynb; the Jupyter Notebook on which I did my work. Its parts are organized and saved among the Python files below
- nn.py; contains the ANN structure
- data_driver.py; contains the code to import and prepare the MNIST dataset
- example.py; contains the code to test and train the model. This code depends on nn.py and data_driver.py
- 0\_all\_biases.csv; the saved biases of the ANN that achieved 95.3% accuracy on the complete MNIST test set. B# columns correspond to bias values, BG# columns correspond to gradient values for the corresponding biases (used for debugging). Note that the first layer biases are not actually used for forward passes (as input layers should not have biases), but they are redundantly computed during backpropogation and updating.
- 0\_all\_weights.csv; the saved weights of the ANN that achieved 95.3% accuracy on the complete MNIST test set. W# columns correspond to bias values, WG# columns correspond to gradient values for the corresponding biases (used for debugging). Note that the last layer has no weights and so its columns (W# and WG#) are empty.
- save_ann.py; code for saving a currently available MLP instance (its weights, biases and current gradients and neuron values) to csv files. TODO: add load() functionality to load saved MLP instance from csv



First venture with ANNs. s/o Andrej Karpathy.
