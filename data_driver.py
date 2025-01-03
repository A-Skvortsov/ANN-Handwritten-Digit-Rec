# Data driver
import keras
from keras.datasets import mnist

class Data():

    # i=1 for testing, i=0 (default) for training
    def __init__(self, num_exercises, i=0):
        (train_x, train_y), (test_x, test_y) = mnist.load_data()
        self.inpt = []
        self.outpt = []
        if (i == 0):
            self.load_data(train_x, train_y, num_exercises)
        else: 
            self.load_data(test_x, test_y, num_exercises)
    
    def load_data(self, t_x, t_y, n):
        t = []
        d = []
        for i in range(n):  # loads n examples
            self.inpt.append(t_x[i].flatten())
            self.outpt.append(t_y[i].flatten())
        self.normalize_inpt()
        self.normalize_outpt()
        return (t, d)

    # maps 0-255 pixel values to values b/t 0 and 1
    def normalize_inpt(self):
        for i in range(len(self.inpt)):
            self.inpt[i] = self.inpt[i] / 255.0

    # casts desired output into list with a 1 at the index equal to 
    # desired output (i.e. if desired output is 6, gives 
    # desired[i] = [0 0 0 0 0 0 1 0 0 0])
    def normalize_outpt(self):
        for i in range(len(self.outpt)):
            x = self.outpt[i].item()
            self.outpt[i] = []
            for j in range(10):  # 10 possible digits, one index for each
                if (x == j): self.outpt[i].append(1)
                else: self.outpt[i].append(0)

# loading training data
train = Data(60000)
print(len(train.inpt[0]), train.outpt[0])

# loading testing data
test = Data(10000, 1)
print(len(test.inpt[0]), test.outpt[0])