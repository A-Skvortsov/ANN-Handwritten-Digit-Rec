mlp = MLP(MLP_layout)  # initializing our mlp

# loading training data
train = Data(60000)
print(len(train.inpt[0]), train.outpt[0])

# training
l = 0.0
for i in range(60000):
    mlp.forward(train.inpt[i])
    print(i, ". ", mlp.loss(train.outpt[i]), mlp.loss(train.outpt[i]) - l)
    l = mlp.loss(train.outpt[i])
    mlp.backward(train.outpt[i])
    mlp.update()
# mlp.result()


# testing
test = Data(10000, 1)
correct = 0.0
total = 0.0
for i in range(10000):
    total += 1
    mlp.forward(test.inpt[i])
    if (test.outpt[i].index(1) == mlp.result()): correct += 1
    print(i, "            exp: ", test.outpt[i].index(1), "act: ", mlp.result(), "         ", correct / total)