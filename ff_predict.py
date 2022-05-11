import torch
import torch.nn.functional as F
import numpy as np
from ff_neural_net import Net
import matplotlib.pyplot as plt
import sys

'''
Cmd line instructions:

1. Model number (model_no)
2. GPU (gpu): -1 => cpu, 0 and above => gpu
3-6. do_train, do_test, do_save, do_load. Booleans

Example command:
python ff_predict.py 99 0 true false false false
'''

# Model
model_dir = '/cnl/data/spate/Corn/models/'
n_inputs = 1 # Input dimension
n_outputs = n_inputs # Output dimension
hidden_widths = [10, 10, 10] # Width of each hidden layer
activation_fcn = F.relu
model_no = int(sys.argv[1])
do_train, do_test, do_save, do_load = [sys.argv[i].lower() == 'true' for i in range(3,7)]
do_norm = True
gpu = int(sys.argv[2]) # GPU number. -1 for CPU

# Data
n_steps = 4000 # Length of time series
n_samples = 500 # Total number of data samples
dt = 0.01
data_dir = '/home/spate/Res/targets/'
data_fn = data_dir + 'mackey_glass_beta_2_gamma_1_n_9.65_tau_2_n_samples_500_n_steps_4000_dt_0.01.csv'
# data_fn = data_dir + 'lorenz_params_sig_10.00_rho_28.00_beta_2.67_n_samples_500_n_steps_4000_dt_0.01.csv'

# Training
batch_size = 50
alpha = 1e-3 # Learning rate
train_eps = 5000 # Training epochs
train_frac = 0.8 # Train / test data split
train_timesteps = n_steps

# Select hardware
if gpu < 0:
    device = torch.device('cpu')
else:
    device = torch.device(f"cuda:{gpu}")

# Create net
if do_load:
    print("Loading")
    net = torch.load(model_dir + f"lorenz_predict_net_{model_no}.pth", map_location=device)
else:
    net = Net(n_inputs, n_outputs, hidden_widths, activation_fcn).to(device)
    print("Feedforward net created")

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
train_data = data[:n_train_samples,:,:train_timesteps-1]
target_data = data[:n_train_samples,:,1:train_timesteps]
test_data = data[n_train_samples:,:,:]

optimizer = torch.optim.Adam(net.parameters(), lr=alpha) # Create optimizer
loss_fcn = torch.nn.MSELoss() # Define loss loss_fcn

# Train
if do_train:
    print("Training")
    loss = []
    for i in range(train_eps):

        batch_loss = []
        for j in range(n_train_samples // batch_size):

            idx = np.random.choice(n_train_samples, size=(batch_size,), replace=False) # Draw random batch
            this_input = np.transpose(train_data[idx,:,:], axes=[0,2,1]).reshape(-1, n_inputs) # Reshape to [batch_size * n_steps, n_inputs]
            this_target = np.transpose(target_data[idx,:,:], axes=[0,2,1]).reshape(-1, n_inputs)
            this_input, this_target = torch.tensor(this_input, dtype=torch.float, device=device), torch.tensor(this_target, dtype=torch.float, device=device)
            output = net(this_input) # Forward pass
            optimizer.zero_grad() # Zero the gradients buffer
            this_loss = loss_fcn(output, this_target) # Compute loss
            this_loss.backward() # Backprop
            optimizer.step() # Gradient descent
            numpy_loss = this_loss.detach().cpu().numpy()
            batch_loss.append(numpy_loss)
        
        loss.append(np.array(batch_loss).mean())

        if i % 100 == 0:
            print("Epoch: ", i, ", MSE: ", np.array(batch_loss).mean())

    loss = np.sqrt(np.array(loss))
    plt.plot(loss)
    plt.show()

if do_test:
    print("Testing")
    idx = np.random.choice(n_samples - n_train_samples)
    print(f"Test Sample # {idx + n_train_samples}")
    test_ex = test_data[idx,:,:].T # (timesteps, d)
    test_target = torch.tensor(test_ex[1:,:], dtype=torch.float, device=device)
    test_input = torch.tensor(test_ex[:-1, :], dtype=torch.float, device=device)

    output = []
    for i in range(n_steps - 1):
        if i < (20 / dt):
            this_output = net(test_input[i,:].reshape(1,-1))
            
        else:
            this_output = net(this_output)

        output.append(this_output.detach().cpu().numpy())

    output = np.vstack(output).T # Transpose to (n_inputs, timesteps)
    test_target = test_target.t() # Transpose back (n_inputs, timesteps)

    # Plot time series from test
    fig, ax = plt.subplots(3, 1, sharex=True)
    var_name = ["x", "y", "z"]
    t = np.arange(n_steps - 1) * dt

    for i in range(n_inputs):
        ax[i].plot(t, output[i,:])
        ax[i].plot(t, test_target[i,:], 'k--')
        ax[i].set_ylabel(var_name[i])

    ax[-1].set_xlabel("time")
    fig.tight_layout()
    plt.show()

    # Plot 2D projection of attractor from test
    fig, ax = plt.subplots(1,2)
    ax[0].plot(test_target[0,:], test_target[2,:], 'k--')
    ax[1].plot(output[0,:], output[2,:], 'b-')
    plt.show()

if do_save:
    print(f"Saving model #{model_no}")
    torch.save(net, model_dir + f"lorenz_predict_net_{model_no}.pth")