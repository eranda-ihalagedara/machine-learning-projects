import numpy as np

def relu(x):
    return max(0,x)

def relu_prime(x):
    if x >= 0:
        return 1
    else:
        return 0

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    sig = sigmoid(x)
    return sig(1-sig)

def linear(x):
    return x

def linear_prime(x):
    return 1