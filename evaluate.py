import numpy as np
import torch
from models import HeavyCNN
from torch.utils.data import DataLoader
from data import TimeSeriesDataset

def process_and_evaluate_time_series(model, time_series, n, num_bins=100):
    """
    Processes the time series and evaluates it using the provided model.
    Splits into sets of 100 points before discretizing each set separately.

    Parameters:
    - model: The trained neural network model for evaluation.
    - time_series (np.ndarray): A numpy array of floats representing the time series.
    - n (int): Resolution for sampling the time series.
    - num_bins (int): Number of bins to use for discretizing the time series values.

    Returns:
    - float: The average entropy reported by the model.
    """
    # Normalize the time series around 0
    time_series_normalized = (time_series - np.mean(time_series)) / np.std(time_series)

    # Sample the time series by averaging over windows of size n
    sampled_series = np.mean(time_series_normalized.reshape(-1, n), axis=1)

    # Split the sampled series into sets of 100 points
    num_sets = len(sampled_series) // 100
    sets = np.array_split(sampled_series, num_sets)

    entropies = []

    for set in sets:
        # Digitize each set separately
        bins = np.linspace(set.min(), set.max(), num=num_bins)
        digitized_set = np.digitize(set, bins) - 1

        # 1-hot encode the digitized set
        # Create TimeSeriesDataset for 1-hot encoding
        fake_labels = np.zeros(len(digitized_set))
        dataset = TimeSeriesDataset(digitized_set.reshape(1, -1), fake_labels.reshape(1, -1), num_bins)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)  # Batch size of 1 since each set is processed independently

        # Evaluate the model on the 1-hot encoded set
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                output = model(inputs)  # Assume model returns entropy directly
                entropies.append(output.cpu().numpy())

    # Calculate the average entropy from all sets
    average_entropy = np.mean(entropies)

    return average_entropy

# Load the model
model = HeavyCNN()
model.load_state_dict(torch.load("checkpoints/best_500k.ckpt"))

time_series_data = np.random.randn(1000)
resolution = 10
average_entropy = process_and_evaluate_time_series(model, time_series_data, resolution)
print("Average Entropy:", average_entropy)
