BATCH_SIZE = 32
mlp = MLP(MLP_layout)  # initializing our mlp

# loading training data
train = Data(60000)
print(len(train.inpt[0]), train.outpt[0])

# training, batch size = 1
l = 0.0
for i in range(60000):
    mlp.forward(train.inpt[i])
    print(i, ". ", mlp.loss(train.outpt[i]), mlp.loss(train.outpt[i]) - l)
    l = mlp.loss(train.outpt[i])
    mlp.backward(train.outpt[i])
    mlp.update()
# mlp.result()

# training, variable batch size
"""
counter = 0
l = [0.0 for i in range(BATCH_SIZE)]
for i in range(3000):
    mlp.forward(train.inpt[i])
    mlp.initgrads(train.outpt[i])
    l[counter] = mlp.lossCE(train.outpt[i])
    counter += 1
    
    if (counter == BATCH_SIZE):
        av_loss = sum(l) / BATCH_SIZE
        print(av_loss)
        counter = 0
        mlp.divide_first_layer_grads()
        mlp.backward_v2()
        mlp.update()
        mlp.zero_grads()
"""

# testing
test = Data(10000, 1)
correct = 0.0
total = 0.0
for i in range(10000):
    total += 1
    mlp.forward(test.inpt[i])
    if (test.outpt[i].index(1) == mlp.result()): 
        correct += 1
        print(i, "            exp: ", test.outpt[i].index(1), "act: ", mlp.result(), "         ", correct / total, "             !")
    else: print(i, "            exp: ", test.outpt[i].index(1), "act: ", mlp.result(), "         ", correct / total)



"""
from matplotlib import pyplot  
# for visualizing data
# see https://www.askpython.com/python/examples/load-and-plot-mnist-dataset-in-python

for i in range(9568, 9577):
    pyplot.imshow(train_x[i], cmap=pyplot.get_cmap('gray'))
    pyplot.show()
pyplot.imshow(train_x[9573], cmap=pyplot.get_cmap('gray'))
pyplot.show()
print(train.outpt[9573])
"""

"""
# saving the ANN weights and biases and gradients to a CSV file
import csv
EMPTY = "          "
with open('0_all_biases.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['B0', 'B1', 'B2', 'B3', '', 'BG0', 'BG1', 'BG2', 'BG3'])
    
    for i in range(max([len(mlp.layers[i].neurons) for i in range(len(mlp.layers))])):  # 'for neuron in layer with most neurons'
        row = [0.0 for j in range(2 * len(mlp.layers) + 1)]
        for j in range(len(mlp.layers)):  # biases
            try: row[j] = mlp.layers[j].neurons[i].bias.val
            except: row[j] = EMPTY
        row[len(mlp.layers)] = EMPTY  # space column for readability
        for j in range(len(mlp.layers) + 1, 2*len(mlp.layers) + 1):  # bias gradients
            try: row[j] = mlp.layers[j - len(mlp.layers) - 1].neurons[i].bias.grad
            except: row[j] = EMPTY
        writer.writerow(row)

with open('0_all_weights.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['W0', 'W1', 'W2', 'W3', '', 'WG0', 'WG1', 'WG2', 'WG3'])

    for i in range(max([len(mlp.layers[x].neurons) for x in range(len(mlp.layers))])):  # 'for neuron in layer with most neurons'
        row = [0.0 for j in range(2 * len(mlp.layers) + 1)]
        for w in range(max([len(mlp.layers[x].neurons[0].weights) for x in range(len(mlp.layers))])):  # 'for weight in layer with most weights'
            for j in range(len(mlp.layers)):  # weights
                try: row[j] = mlp.layers[j].neurons[i].weights[w].val
                except: row[j] = EMPTY
            row[len(mlp.layers)] = EMPTY  # space column for readability
            for j in range(len(mlp.layers) + 1, 2*len(mlp.layers) + 1):  # weight gradients
                try: row[j] = mlp.layers[j - len(mlp.layers) - 1].neurons[i].weights[w].grad
                except: row[j] = EMPTY
            writer.writerow(row)
        writer.writerow([EMPTY for i in range(2 * len(mlp.layers) + 1)])
"""
