import os
import pandas as pd
import numpy as np

files = [
    int(x.split('_')[1].split('.')[0]) for x in os.listdir('boards/')
    if x.endswith('.npy')
]
number = max(files) + 1
df = pd.read_csv('boards/scratch_pad.csv', header=None)
np.save(f'boards/start_{number}.npy', df.to_numpy(dtype=np.int8))
