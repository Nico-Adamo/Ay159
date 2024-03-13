from epsilon_machines.stochastic_epsilon import StochasticEpsilonMachine
from models import HeavyCNN, SimpleCNN, HeavierCNN
from data import TimeSeriesDataset, generate_time_series
from constants import num_measurements, num_series, series_length

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    model.to(device)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    best_val_loss = float('inf')  # Initialize best validation loss as infinity

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()  # Update the learning rate

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}, Validation Loss: {val_loss}')

        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/best.ckpt')
            print(f'Model saved as best.ckpt at epoch {epoch+1} with Validation Loss: {val_loss}')

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            # if loss > 0.5:
            #     print(outputs, labels)
            test_loss += loss.item()

    print(f'Test Loss: {test_loss/len(test_loader)}')


if True:
    # Generate new data
    print("Preparing data...")
    series, labels = generate_time_series(num_series, series_length, "stochastic", state_range=(2, 50), measurements=num_measurements)
    np.save("data/series.npy",series)
    np.save("data/labels.npy",labels)
    print("Number of labels", len(labels))
else:
    # Use existing
    series = np.load("data/series.npy")
    labels = np.load("data/labels.npy")


# Shuffle the dataset
indices = np.arange(series.shape[0])
np.random.shuffle(indices)
series = series[indices]
labels = labels[indices]

# Split the data into training, validation, and test sets
series_train, series_temp, labels_train, labels_temp = train_test_split(series, labels, test_size=0.4, random_state=42)
series_val, series_test, labels_val, labels_test = train_test_split(series_temp, labels_temp, test_size=0.5, random_state=42)

# Create dataset objects
train_dataset = TimeSeriesDataset(series_train, labels_train, num_measurements)
val_dataset = TimeSeriesDataset(series_val, labels_val, num_measurements)
test_dataset = TimeSeriesDataset(series_test, labels_test, num_measurements)

# Create DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, criterion, and optimizer
model = HeavyCNN()
# model = HeavierCNN()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# Train the model
train(model, train_loader, val_loader, criterion, optimizer, num_epochs=15)

# Test the model
test(model, test_loader, criterion)

# Example model outputs
# Plot the distribution of model outputs
model.eval()
outputs = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs.append(model(inputs).squeeze().numpy())
outputs = np.concatenate(outputs)
import matplotlib.pyplot as plt
plt.hist(outputs, bins=50)
# Save to file
plt.savefig("model_outputs_hist.png")