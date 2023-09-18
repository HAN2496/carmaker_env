import numpy as np
first_cone_arr = [[i, -5.25] for i in range(100, 400, 30)]
second_cone_arr = [[i, -5.25] for i in range(600, 800, 30)]
cone_arr = np.array(first_cone_arr + second_cone_arr)
print(cone_arr)
