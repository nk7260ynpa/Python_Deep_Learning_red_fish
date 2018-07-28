from P101 import TwoLayerNet
import numpy as np

net = TwoLayerNet(784, 100, 10)
#print(net.params['W1'].shape)
#print(net.params['b1'].shape)
#print(net.params['W2'].shape)
#print(net.params['b2'].shape)

x = np.random.rand(100, 784)
y = net.predict(x)
t = np.random.rand(100, 10)

grads = net.numerical_gradient(x, t)
print(grads)
print(grads['W1'].shape)
print(grads['b1'].shape)
print(grads['W2'].shape)
print(grads['b2'].shape)
