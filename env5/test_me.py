import numpy as np

a = [1, 2]
print(np.array(a))
print(np.array(np.array(a)))

i = 0
while i < 10:
    print(i)
    i+=1
else:
    print(i)
