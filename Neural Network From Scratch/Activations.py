import numpy as np

def relu(val):
    return max(0,val)

def sigmoid(val):
    return 1/(1+np.exp(-val))
    