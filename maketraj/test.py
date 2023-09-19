import numpy as np
import pandas as pd
df = pd.read_csv('traj_slalom_onefifth.csv').loc[:, ["traj_tx", "traj_ty"]].values
print(df)
