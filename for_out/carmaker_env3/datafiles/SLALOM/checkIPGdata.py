import numpy as np
import pandas as pd

data = pd.read_csv('check_IPG.csv')
sight = 27
num = data.loc[:, "num"].values
devDist = data.loc[:, "devDist"].values
devAng = data.loc[:, "devAng"].values
avg_dist = np.sum(np.abs(devDist)) / len(data)
avg_ang = np.sum(np.abs(devAng)) / len(data)
print(avg_dist, avg_ang, avg_dist / avg_ang)
print(np.max(np.abs(devDist)), np.max(np.abs(devAng)))
