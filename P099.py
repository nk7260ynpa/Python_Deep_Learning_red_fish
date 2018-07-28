from P098 import simpleNet
import numpy as np

from common.gradient import numerical_gradient

net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))
t = np.array([0,0,1])
print(net.loss(x, t))
print('-------------')
def f(W):
    return net.loss(x, t)
dW = numerical_gradient(f, net.W)
print(dW)

f = lambda w:net.loss(x, t)
dW = numerical_gradient(f, net.W)