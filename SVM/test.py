import numpy as np

a = np.array([[0,0,3],[0,0,0],[0,0,9]])
b = np.nonzero(a)
print(b)
print(np.array(b).ndim)
