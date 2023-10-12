import numpy as np


a = np.array([[0, 0], [1,2]])
b = np.array([3, 4])

c = np.sqrt(np.sum((a - b) **2, axis=1))
print(c)