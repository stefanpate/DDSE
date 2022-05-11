import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mat73 import loadmat
import sys

# Load data
data_dir = '/cnl/data/spate/Corn/'
fn = data_dir + sys.argv[1]
data = np.loadtxt(fn, delimiter=',')
y, x, t = 351, 451, data.shape[-1]
k = 200

data = data.reshape(y, x, t)
data = data[:,:,:k]

# Set up figure
y, x, t = 351, 451, data.shape[-1]
fig = plt.figure()
ax = plt.axes(xlim=(0, x), ylim=(0, y))

cax = ax.pcolormesh(data[:-1, :-1, 0], cmap='viridis')
fig.colorbar(cax)

def animate(i):
     cax.set_array(data[:-1, :-1, i].flatten())

anim = FuncAnimation(fig, animate, interval=20, frames=t-1)

# Save
video_fn = fn.split('.')[0]
anim.save(video_fn + '.gif', writer='imagemagick')

plt.draw()
plt.show()