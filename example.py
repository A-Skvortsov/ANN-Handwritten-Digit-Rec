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
