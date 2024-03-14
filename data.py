import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from epsilon_machines.stochastic_epsilon import StochasticEpsilonMachine
from epsilon_machines.stateful_stochastic_epsilon import StatefulStochasticEpsilonMachine

class TimeSeriesDataset(Dataset):
    def __init__(self, series, labels, num_classes):
        """
        Initializes the dataset with 1-hot encoded time series.

        Parameters:
        series (np.ndarray): Array of shape (num_samples, series_length) with discrete states.
        labels (np.ndarray): Array of labels for the series.
        num_classes (int): The number of unique states or classes in the time series.
        """
        self.series = series
        self.labels = labels
        self.num_classes = num_classes

        # 1-hot encode the series
        self.series_encoded = self.one_hot_encode(series)

    def one_hot_encode(self, series):
        """
        1-hot encodes the input series.

        Parameters:
        series (np.ndarray): Array of shape (num_samples, series_length) with discrete states.

        Returns:
        np.ndarray: 1-hot encoded series of shape (num_samples, series_length, num_classes).
        """
        # Initialize an array of zeros for the 1-hot encoded series
        series_encoded = np.zeros((series.shape[0], series.shape[1], self.num_classes), dtype=np.float32)

        # Set the appropriate indices to 1
        for i, sequence in enumerate(series):
            series_encoded[i, np.arange(series.shape[1]), sequence] = 1

        return series_encoded

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        return torch.tensor(self.series_encoded[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

def generate_time_series(num_series, series_length, machine_type, state_range=(10, 50), measurements=100):
    """
    Generates time series data from epsilon machines.

    Parameters:
    num_series (int): Number of series to generate.
    series_length (int): Length of each series.
    machine_type (str): Type of machine ("stochastic" or "deterministic").
    state_range (tuple): Range of states for the epsilon machines.
    measurements (int): Number of possible measurements.

    Returns:
    np.array: Generated time series.
    np.array: Labels for the series.
    """
    series_list = []
    labels = []

    for _ in tqdm(range(num_series)):
        num_states = np.random.randint(state_range[0], state_range[1] + 1)
        possible_measurements = list(range(measurements))

        if machine_type == "stochastic":
            machine = StochasticEpsilonMachine(num_states, possible_measurements)
        elif machine_type == "stochastic_stateful":
            machine = StatefulStochasticEpsilonMachine(num_states, possible_measurements)
        series = machine.generate_series(series_length, 1) # Arbitrary starting measurement, might want to make random
        entropy = machine.entropy()
        if entropy != 0.0:
            labels.append(entropy)
            series_list.append(series)

    return np.array(series_list), np.array(labels)

