import torch
import torch.nn.functional as F
import numpy as np
from ff_neural_net import Net
import matplotlib.pyplot as plt
import sys
from library import data2fn

'''
Loads and tests saved models.
'''
# Model
model_dir = '/cnl/data/spate/Corn/models/'
save_dir = '/cnl/data/spate/Corn/'
n_inputs = 3 # Input dimension
n_outputs = 3 # Output dimension
model_lo, model_hi = 30, 34
model_numbers = np.arange(model_lo, model_hi + 1)
do_norm = True
gpu = -1 # GPU number. -1 for CPU

# Data
n_steps = 4000 # Length of time series
n_samples = 500 # Total number of data samples
dt = 0.01
dataset = 'lorenz'
data_fn = data2fn[dataset]
train_frac = 0.8 # Train / test data split
switch_t = 20 # When to switch between teacher forcing and output feedback
switch_step = int(20 / dt) # Timestep associated with switch_t

# Select hardware
if gpu < 0:
    device = torch.device('cpu')
else:
    device = torch.device(f"cuda:{gpu}")

# Load data
data = np.loadtxt(data_fn, delimiter=',').reshape(n_samples, n_inputs, n_steps)

# Normalize data
if do_norm:
    flat_data = np.transpose(data, axes=[1, 0, 2]).reshape(n_inputs, -1)
    mean, std = np.mean(flat_data, axis=1), np.std(flat_data, axis=1)
    data -= mean.reshape(1, -1, 1)
    data /= std.reshape(1, -1, 1)

# Split data
n_train_samples = int(n_samples * train_frac)
batch_size = int((1 - train_frac) * n_samples)
test_data = data[n_train_samples:,:,:] # (batch_size, n_inputs, n_steps)

# Time-shift data
test_target_np = test_data[:,:,1:]
test_target = torch.tensor(test_data[:,:,1:], dtype=torch.float, device=device)
test_input = torch.tensor(test_data[:,:,:-1], dtype=torch.float, device=device)

# Load and test models
test_output_traces = []
save_last_t = []
for model_no in model_numbers:
    print(f"Loading model #{model_no}")
    net = torch.load(model_dir + f"{dataset}_predict_net_{model_no}.pth", map_location=device)

    output = []
    for i in range(n_steps - 1):
        if i < switch_step:
            this_output = net(test_input[:,:,i]) # (batch_size, n_inputs)  
        else:
            this_output = net(this_output)

        output.append(this_output.detach().cpu().numpy())

    output = np.stack(output, axis=-1) # (batch_size, n_inputs, n_steps - 1)
    mse = np.square(output - test_target_np).mean(axis=1)

    last_t = []
    for j in range(batch_size):
        # Find last timestep of accurate prediction
        error_mask = mse[j,switch_step:] > 1
        if np.all(error_mask): # Error above thresh at all times
            last_t.append(0)
        elif not np.any(error_mask): # Error stays below thresh
            last_t.append((n_steps - switch_step) * dt)
        else:
            last_t.append((np.where(error_mask)[0][0] + 1) * dt)

    save_last_t.append(last_t)
    test_output_traces.append(output.reshape(-1, n_steps - 1))

save_last_t = np.vstack(save_last_t) # (n_models, n_test_samples)
test_output_traces = np.vstack(test_output_traces) # (n_models * n_test_samples * n_inputs, n_steps - 1)

np.savetxt(save_dir + f"ff_nn_predict_{dataset}_test_output_traces_models_{model_numbers[0]}_{model_numbers[-1]}.csv", test_output_traces, delimiter=',')
np.savetxt(save_dir + f"ff_nn_predict_{dataset}_test_last_t_models_{model_numbers[0]}_{model_numbers[-1]}.csv", save_last_t, delimiter=',')