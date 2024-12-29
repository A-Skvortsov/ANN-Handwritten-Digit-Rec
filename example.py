# Training the model
# With sufficient training examples,
# this will achieve ~95% accuracy on the 10000 available tests

mlp = MLP(MLP_LAYOUT)
NUM_TESTS = 1000
NUM_TRAINING_EXAMPLES = 1000

# Testing first 
test = Data(10000, 1)
correct = 0.0
total = 0.0
for i in range(NUM_TESTS):
    total += 1
    mlp.forward(test.inpt[i])
    if (test.outpt[i].index(1) == mlp.result()): 
        correct += 1
        print(i, "            exp: ", test.outpt[i].index(1), "act: ", mlp.result(), "         ", correct / total, "             !")
    else: print(i, "            exp: ", test.outpt[i].index(1), "act: ", mlp.result(), "         ", correct / total)

# Training
counter = 0
l = [0.0 for i in range(BATCH_SIZE)]
for i in range(NUM_TRAINING_EXAMPLES):
    mlp.forward(train.inpt[i])
    mlp.initgrads(train.outpt[i])
    l[counter] = mlp.lossCE(train.outpt[i])
    counter += 1
    
    if (counter == BATCH_SIZE):
        av_loss = sum(l) / BATCH_SIZE
        print(i, ". ", av_loss)
        counter = 0
        mlp.divide_first_layer_grads()
        mlp.backward_v2()
        mlp.update()
        mlp.zero_grads()
        
        
# Testing again, after training
correct = 0.0
total = 0.0
for i in range(NUM_TESTS):
    total += 1
    mlp.forward(test.inpt[i])
    if (test.outpt[i].index(1) == mlp.result()): 
        correct += 1
        print(i, "            exp: ", test.outpt[i].index(1), "act: ", mlp.result(), "         ", correct / total, "             !")
    else: print(i, "            exp: ", test.outpt[i].index(1), "act: ", mlp.result(), "         ", correct / total)