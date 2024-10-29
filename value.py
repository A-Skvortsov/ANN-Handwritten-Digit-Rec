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