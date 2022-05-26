'''
Make an animated movie from 2d space by time tensor.

Args:
    - y, x: Size of y and x dimensions
    - k : Number of timesteps
    - List of file names separated by spaces to load from
'''


import numpy as np
import sys
from library import make_movie, mask_sst

data_dir = '/cnl/data/spate/Datasets/'
save_dir = '/cnl/data/spate/Corn/'
y, x = int(sys.argv[1]), int(sys.argv[2])
k = int(sys.argv[3])
files = sys.argv[4:]

# Get mask for SST data
mask = np.loadtxt(data_dir + 'sst_mask.csv', delimiter=',')
mask = mask.astype(bool).reshape(180, 360, 1)

for f in files:
    print("Loading file:", f)
    data = np.loadtxt(data_dir + f, delimiter=',') # Load data
    
    if 'sst' in f:
        data = mask_sst(data, mask) # Mask sea surface data, shape inherited from mask
    else:
        data = data.reshape(y, x, -1) # Use user inputted shape
    
    save_as = f.split('.')[0]
    print("Making movie:", f)
    make_movie(data, k, save_dir + save_as, cmap='plasma', interval=50, a=6) # Make movie