import numpy as np

class DeterministicEpsilonMachine:
    def __init__(self, num_states, measurements, transitions=None, measurement_indices=None):
        """
        Initializes the Deterministic Epsilon Machine.

        Parameters:
        num_states (int): Number of states in the machine.
        measurements (list): A list of possible measurements Q.
        transitions (list, optional): A list defining the next state for each current state.
        measurement_indices (list, optional): A list defining the index of the measurement associated with the transition from each state.
        """
        self.num_states = num_states
        self.measurements = measurements
        self.current_state = 0  # Initial state

        if transitions is None:
            # If transitions are not provided, initialize them randomly
            self.transitions = np.random.randint(num_states, size=num_states)
        else:
            self.transitions = np.array(transitions)

        if measurement_indices is None:
            # If measurement indices are not provided, initialize them randomly
            self.measurement_indices = np.random.randint(len(measurements), size=num_states)
        else:
            self.measurement_indices = np.array(measurement_indices)

    def step(self):
        """
        Performs a step by moving to the next state based on the deterministic transition
        and returns the associated measurement.

        Returns:
        The measurement associated with the transition to the new state.
        """
        self.current_state = self.transitions[self.current_state]
        measurement_index = self.measurement_indices[self.current_state]
        return self.measurements[measurement_index]

    def generate_series(self, n, initial_measurement):
        """
        Generates a series of n measurements based on the initial measurement.

        Parameters:
        n (int): The number of measurements to generate.
        initial_measurement: The initial measurement to start with.

        Returns:
        list: A list of generated measurements.
        """
        series = [initial_measurement]
        for _ in range(1, n):
            series.append(self.step())
        return series
