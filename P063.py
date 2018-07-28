import numpy as np

a = np.array([1010, 1000, 990])
#print(np.exp(a) / np.sum(np.exp(a)))
c = np.max(a)
print(a - c)

print(np.exp(a - c) / np.sum(np.exp(a - c)))

