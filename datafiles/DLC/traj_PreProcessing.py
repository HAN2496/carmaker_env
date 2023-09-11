import numpy as np
import pandas as pd

data = pd.read_csv('datasets_traj_DLC1.csv')
sight = 27
traj_tx = data.loc[:, "traj_tx"].values
traj_tx_last = traj_tx[-1]
traj_ty = data.loc[:, "traj_ty"].values
num = data.loc[:, "num"].values

tx_spaces = []
for i in range(20):
    tx_space = traj_tx[-1-i] - traj_tx[-2-i]
    tx_spaces.append(tx_space)
avg_space = np.sum(tx_spaces) / 20
print(avg_space)

lookahead_traj_tx = []
lookahead_traj_ty = []
k=0
while k <=sight + 1:
    k += avg_space
    lookahead_traj_tx.append(traj_tx_last + k)
    lookahead_traj_ty.append(-3)
new_traj_tx = np.append(traj_tx, lookahead_traj_tx)
new_traj_ty = np.append(traj_ty, lookahead_traj_ty)

new_traj = np.column_stack((new_traj_tx, new_traj_ty))
indexes= ["traj_tx", "traj_ty"]
new_data = pd.DataFrame(data=new_traj, columns=indexes)
new_data.to_csv("datasets_traj_SLALOM_env17_test.csv")
