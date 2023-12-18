import os
import numpy as np

def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

directory = r"models\DLC\rws_low"
files = list_files(directory)

filtered_files = [f for f in files if f.endswith('.pkl')]
filtered_files = [f for f in filtered_files if f != 'model.pkl']
filtered_files.sort(key=lambda x: int(x.split('_')[0]))
filtered_files2 = [filtered_files[i] for i in range(len(filtered_files)) if (i > 100 and i % 50 == 0) or (i < 100 and i % 10 == 0)]
print(np.size(filtered_files2))
for i in filtered_files2:
    print(i)
