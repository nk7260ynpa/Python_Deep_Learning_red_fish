import numpy as np

a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)



c = np.concatenate([a,b],axis=1)
print(c)

onv_param={'filter_num':30, 'filter_size':5,'pad':0, 'stride':1}
print(onv_param)