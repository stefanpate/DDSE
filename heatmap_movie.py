'''
Make an animated movie from 2d space by time tensor.

Args:
    - y, x: Size of y and x dimensions
    - k : Number of timesteps
    - List of file names separated by spaces to load from
'''


import numpy as np
import sys
from library import make_movie

data_dir = '/cnl/data/spate/Corn/'
y, x = int(sys.argv[1]), int(sys.argv[2])
k = int(sys.argv[3])
files = sys.argv[4:]
mask = np.loadtxt(data_dir + 'sst_land_mask.csv', delimiter=',').reshape(180, 360, 1) # For SST

for f in files:
    print("Loading file:", f)
    data = np.loadtxt(data_dir + f, delimiter=',') # Load data
    data = data.reshape(y, x, -1)

    if 'sst' in f: # Mask sea surface data
        data *= mask
    
    save_as = f.split('.')[0]
    print("Making movie:", f)
    make_movie(data, k, data_dir + save_as, cmap='plasma', interval=50, a=6) # Make movie