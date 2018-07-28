import numpy as np

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

def relu(x):
    return np.maximum(0, x)





