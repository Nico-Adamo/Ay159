from epsilon_machines.stochastic_epsilon import StochasticEpsilonMachine
from data import TimeSeriesDataset, generate_time_series
from models import HeavyCNN, SimpleCNN
from constants import num_measurements, num_series, series_length

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset

# UNCOMMENT TO Load labels.npy and plot the distribution of labels, save to a file
# import matplotlib.pyplot as plt
# import numpy as np
# labels = np.load("labels.npy")
# plt.hist(labels, bins=50)
# plt.xlabel("Shannon Entropy")
# plt.ylabel("Frequency")
# plt.savefig("labels_hist.png")

# Load the model
model = HeavyCNN()
model.load_state_dict(torch.load("checkpoints/best.ckpt"))

# Set the model to evaluation mode
model.eval()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# # UNCOMMENT BELOW FOR AVERAGE LOSS VS STATE NUMBER
# def evaluate_model(model, num_states):
#     series, labels = generate_time_series(500, 100, "stochastic", state_range=(num_states, num_states), measurements=num_measurements)
#     dataset = TimeSeriesDataset(series, labels, num_measurements)
#     loader = DataLoader(dataset, batch_size=64, shuffle=False)

#     criterion = torch.nn.MSELoss()
#     total_loss = 0.0
#     with torch.no_grad():
#         for inputs, labels in loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs).squeeze()
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#     return total_loss / len(loader)

# # Evaluate the model for different numbers of states
# num_states_range = range(2, 51)
# average_losses = []

# for num_states in num_states_range:
#     avg_loss = evaluate_model(model, num_states)
#     average_losses.append(avg_loss)
#     print(f"Average loss for {num_states} states: {avg_loss}")

# # Plot the average loss against the number of states
# plt.figure(figsize=(10, 6))
# plt.plot(num_states_range, average_losses, marker='o')
# plt.title("Average MSE vs. Number of States")
# plt.xlabel("Number of States")
# plt.ylabel("Average Mean Squared Error")
# plt.grid(True)
# plt.savefig("average_loss_vs_states.png")
# plt.show()

# # UNCOMMENT BELOW FOR AVERAGE LOSS VS ENTROPY
# def evaluate_model_on_machines(model, num_machines=100):
#     losses = []

#     series, labels = generate_time_series(num_machines, series_length, "stochastic", state_range=(2, 51), measurements=num_measurements)
#     dataset = TimeSeriesDataset(series, labels, num_measurements)
#     loader = DataLoader(dataset, batch_size=1, shuffle=False)
#     entropies = labels

#     criterion = torch.nn.MSELoss()
#     total_loss = 0.0
#     with torch.no_grad():
#         for inputs, labels in loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs).squeeze(dim=0)
#             loss = criterion(outputs, labels)
#             losses.append(loss.item())

#     return entropies, losses

# entropies, average_losses = evaluate_model_on_machines(model)

# plt.figure(figsize=(8, 6), dpi=300)  # High-resolution output
# plt.scatter(entropies, average_losses, alpha=0.75, edgecolors='w', s=50, c='blue')  # Add marker properties
# plt.title("Relationship Between Machine Entropy and Model Loss", fontsize=14, fontweight='bold')
# plt.xlabel("Machine Entropy", fontsize=12)
# plt.ylabel("Average Loss", fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Improved grid
# plt.tight_layout()  # Adjust layout to make room for the larger labels
# plt.savefig("entropy_vs_loss.png")

# UNCOMMENT TO GENERATE PLOT OF MACHINE STATES VS ENTROPY
# entropies = []
# state_nums = []
# for i in range(1000):
#     statenum = np.random.randint(2, 51)
#     machine = StochasticEpsilonMachine(statenum, range(num_measurements))
#     machine.generate_series(100, 1)
#     entropies.append(machine.entropy())
#     state_nums.append(statenum)

# plt.figure(figsize=(8, 6), dpi=300)
# plt.scatter(state_nums, entropies, alpha=0.75, edgecolors='w', s=50, c='blue')  # Add marker properties
# # Draw a horizontal line at y=1.5 in a different color

# # Calculate entropy of random series
# series = np.random.randint(0, 100, size=(100,))
# unique, counts = np.unique(series, return_counts=True)
# probabilities = counts / len(series)
# entropy = -np.sum(probabilities * np.log2(probabilities))
# plt.axhline(y=entropy, color='r', linestyle='-', label='Entropy of a random timeseries')

# plt.title("Relationship Between Machine States and Entropy", fontsize=14, fontweight='bold')
# plt.xlabel("Number of States", fontsize=12)
# plt.ylabel("Entropy", fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.legend()
# plt.savefig("states_vs_entropy.png")