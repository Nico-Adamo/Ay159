import numpy as np

class StochasticEpsilonMachine:
    def __init__(self, n, Q, T=None, measurement_matrix=None):
        """
        Initializes the StochasticEpsilonMachine with the given parameters.

        Parameters:
        - n (int): Number of states in the machine.
        - Q (list): A list of possible measurements.
        - T (np.ndarray, optional): A Q x n x n transition tensor.
        - measurement_matrix (np.ndarray, optional): A matrix indicating the measurement outputs for state transitions.
        """
        self.n = n
        self.Q = Q
        self.T = T if T is not None else np.zeros((len(Q), n, n))
        self.measurement_matrix = measurement_matrix if measurement_matrix is not None else np.full((n, n), -1)
        self.current_state = 0  # Initial state
        self.last_measurement = None  # Placeholder for the last measurement

    def random(self, avg_transitions_per_state):
        """
        Initializes the transition tensor T and measurement matrix with random values,
        based on the average number of transitions per state.

        Parameters:
        - avg_transitions_per_state (float): The average number of transitions per state.
        """
        for q in range(len(self.Q)):
            for i in range(self.n):
                num_transitions = int(np.random.normal(avg_transitions_per_state, 1))
                # Ensure num_transitions is within bounds
                num_transitions = max(1, min(num_transitions, self.n))

                # Select random indices for transitions
                j_indices = np.random.choice(range(self.n), num_transitions, replace=False)

                # Assign random probabilities to these transitions
                for j in j_indices:
                    self.T[q, i, j] = np.random.random()

                    # Assign a random measurement if not already assigned
                    if self.measurement_matrix[i, j] == -1:
                        self.measurement_matrix[i, j] = np.random.randint(len(self.Q))

                # Normalize T[q, i] so that it sums to 1
                self.T[q, i, :] /= np.sum(self.T[q, i, :])

    def step(self):
        """
        Performs a single step update of the machine's state according to the transition tensor,
        and returns the corresponding measurement.

        Returns:
        - measurement: The measurement result of this step.
        """
        if self.last_measurement is None:
            raise ValueError("Initial measurement not set. Use generate_series to set an initial measurement.")

        # Convert the last measurement to its index in Q
        last_measurement_index = self.Q.index(self.last_measurement)

        # Choose the next state based on the current state and the transition probabilities.
        transition_probs = self.T[last_measurement_index, self.current_state, :]
        next_state = np.random.choice(range(self.n), p=transition_probs)

        # Determine the measurement associated with the state transition.
        measurement_index = self.measurement_matrix[self.current_state, next_state]
        self.last_measurement = self.Q[measurement_index]

        # Update the current state.
        self.current_state = next_state

        return self.last_measurement

    def generate_series(self, n, initial_measurement):
        """
        Generates a series of measurements over n steps, starting with an initial measurement.

        Parameters:
        - n (int): The number of steps to simulate.
        - initial_measurement: The measurement to use as the starting point.

        Returns:
        - series (list): A list of measurements observed over n steps.
        """
        if initial_measurement not in self.Q:
            raise ValueError("Initial measurement is not in the set of possible measurements Q.")

        self.last_measurement = initial_measurement
        return [self.step() for _ in range(n)]
