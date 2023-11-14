import numpy as np 

def mse(a,y):
    da = a - y
    loss = np.mean(np.sum(np.square(da), axis=0,keepdims=True))
    return loss, da

def softmax_loss(a,y):
    dz = a - y
    loss = -np.mean(np.sum(np.exp(a)*y, axis=0,keepdims=True))
    return loss, dz