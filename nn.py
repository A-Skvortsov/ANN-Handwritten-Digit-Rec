import random
import math

# 784 pixels for input, 10 possible outputs (0, 1,...,9), arbitrary hidden layers
MLP_layout = [784, 16, 16, 10]

class Value:

    def __init__(self, val, grad=0.0):
        self.val = val
        self.grad = grad

    def __add__(self, other):
        return Value(self.val + other.val)

    def __mul__(self, other):
        return Value(self.val * other.val)

    def __gt__(self, other):
        return self.val > other.val

    def __repr__(self):
        return "Value obj: " + str(self.val)

    def reLU(self):
        return Value(max(0, self.val))


class Neuron:

    def __init__(self, nout):
        self.weights = [Value(random.uniform(-1,1)) for i in range(nout)]
        self.val = Value(0.0)
        self.bias = Value(random.uniform(-1,1))


class Layer:

    def __init__(self, n, nout):
        self.neurons = [Neuron(nout) for i in range(n)]


class MLP:

    def __init__(self, MLP_layout):
        self.layers = [Layer(size, next_size)
                       for size, next_size in zip(MLP_layout, MLP_layout[1:] + [0])]


    # currently O(n^3) T∆T
    def forward(self, inpt):
        # for every layer in mlp.layers[1:]
        # for every neuron2 in layer2
            # accum_val = neuron2bias
            # for every neuron1 in layer1
                # accum_val += neuronval * weight2
            # neuron2val = sigmoid(accum_val)


        # initialize neuron values to 0 before beginning
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.val = Value(0.0)

        for i in range(len(inpt)):  # loading first layer of mlp
            self.layers[0].neurons[i].val = Value(inpt[i])

        for layer1, layer2 in zip(self.layers[:-1], self.layers[1:]):
            for neuron2, n in zip(layer2.neurons, range(len(layer2.neurons))):
                neuron2.val += neuron2.bias;  # load with its bias
                for neuron1 in layer1.neurons:  # load with val*weight of each prev neuron
                    neuron2.val += neuron1.val * neuron1.weights[n]
                neuron2.val = neuron2.val.reLU()  # commpression/activation f'n
        # print(self.result())


    # just prints the result of the last forward pass
    def result(self):
        last_layer = self.layers[-1]
        x = 0

        for i in range(1, len(last_layer.neurons)):
            if (last_layer.neurons[i].val > last_layer.neurons[x].val):
                x = i
        return x


    def loss(self, desired):
        last_layer = self.layers[-1]
        loss = 0.0
        for i in range(len(last_layer.neurons)):
            loss += (last_layer.neurons[i].val.val - desired[i])**2
        return loss


    # assigns a gradient to each value (weight, bias, neuron) in the mlp
    # based on cost f'n which is computed as sum of squared diffs
    def backward(self, desired):
        # 1) zero out previous grads
        # 2) initialize grads of final layer considering cost f'n def'n
        # 3) for each layer in mlp_reverse except final layer
            # for each neuron in layer
                # for each weight from neuron
                    # gradw = (grad neuron of prev layer)*(sigmoid deriv)*val_neuron
                # if not in last layer...
                    # gradn = sum((grad neuron of prev layer)*(sigmoid deriv)*w)
                    # gradb = sum((grad neuron of prev layer)*(sigmoid deriv)*1)

        # 1)
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.val.grad = 0.0
                neuron.bias.grad = 0.0
                for weight in neuron.weights:
                    weight.grad = 0.0

        # 2)
        last_layer = self.layers[len(self.layers)-1].neurons
        for neuron, n in zip(last_layer, range(len(last_layer))):
            neuron.val.grad = 2*(desired[n]- neuron.val.val)

        # 3)
        for layer2, layer1 in zip(reversed(self.layers[:-1]), reversed(self.layers[1:])):
            for neuron2 in layer2.neurons:
                # keep in mind; # of weights in any neuron of layer1 = # of neurons in layer2
                for neuron1, n in zip(layer1.neurons, range(len(layer1.neurons))):
                    if (neuron1.val.grad <= 0): x = 0
                    else: x = 1
                    # keep in mind that derivative of our activation (reLU) is 1 or 0
                    # it is 1 if neuron1.val is non-0 (so can be omitted)
                    # it is 0 otherwise
                    neuron2.weights[n].grad = neuron1.val.grad * neuron2.val.val
                    neuron2.val.grad += neuron1.val.grad * neuron2.weights[n].val
                    neuron2.bias.grad += neuron1.val.grad * x
                # If we are in the last layer, no need to compute gradb or gradn
                # as they will never be used. Just unecessary
                # what we want: "while NOT in the last layer, then compute grab & gradn"


    # updates mlp using the gradients of each value
    def update(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                # update bias
                neuron.bias.val += 0.00001 * neuron.bias.grad
                # update weights
                for weight in neuron.weights:
                    weight.val += 0.00001 * weight.grad

    # used for debugging
    def printwb(self):
        w = Value(0.0)
        b = Value(0.0)
        for layer in self.layers:
            for neuron in layer.neurons:
                if (abs(b.val) < abs(neuron.bias.val)): b = neuron.bias
                for weight in neuron.weights:
                    if (abs(w.val) < abs(weight.val)): w = weight
        print(w)
        print(b)