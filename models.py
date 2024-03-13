import torch
import torch.nn as nn
import torch.nn.functional as F

class HeavyCNN(nn.Module):
    def __init__(self):
        super(HeavyCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        # Calculate the size of the output from conv2 + pool layers to correctly size the fc1 layer
        self.fc1 = nn.Linear(32 * 25 * 25, 120)
        self.fc2 = nn.Linear(120, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output for the linear layer
        x = x.view(-1, 32 * 25 * 25)  # Adjust the size based on your conv/pool layers' output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation here as we want a continuous range output
        return x

class SimpleCNN(nn.Module):
    def __init__(self, num_measurements):
        """
        Initializes the CNN model.

        Parameters:
        - num_measurements (int): The number of measurements (unique states) in the 1-hot encoded time series.
        """
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=num_measurements, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # Calculate the size of the output from the last convolutional layer
        self.num_flat_features = 32 * (200 // 4)

        self.fc1 = nn.Linear(self.num_flat_features, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """
        Forward pass through the model.

        Parameters:
        - x (Tensor): The input tensor containing the 1-hot encoded time series data.

        Returns:
        - Tensor: The model's output.
        """
        # x shape is expected to be [batch_size, series_length, num_measurements]
        # Need to permute to match Conv1d input requirement: [batch_size, channels, length]
        x = x.permute(0, 2, 1)

        # Apply 1D convolutions and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return 3 * x

class HeavierCNN(nn.Module):
    def __init__(self):
        super(HeavierCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # After 3 sets of Conv2d and MaxPool2d, the feature map size is [128, 64, 25].
        self.fc1 = nn.Linear(3145728, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Assuming x has shape [N, 256, 100] initially
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 3145728)  # Flatten the output for the linear layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Assuming a continuous range output, no activation here
        return x
