import numpy as np
import matplotlib.pyplot as plt

class StatefulStochasticEpsilonMachine:
    def __init__(self, num_states, measurements, transition_tensor=None, measurement_matrix=None, avg_transitions_per_state=2):
        """
        Initializes the Stochastic Epsilon Machine.

        Parameters:
        num_states (int): Number of states in the machine.
        measurements (list): A list of possible measurements Q.
        transition_tensor (numpy.ndarray, optional): An q x n x n transition matrix T.
        measurement_matrix (numpy.ndarray, optional): A matrix defining the measurement for transitions.
        """
        self.num_states = num_states
        self.measurements = measurements
        self.current_state = 0  # Initial state

        if transition_tensor is None or measurement_matrix is None:
            self.transition_tensor, self.measurement_matrix = self.random(avg_transitions_per_state=avg_transitions_per_state)
        else:
            self.transition_tensor = transition_tensor
            self.measurement_matrix = measurement_matrix

    def step(self, measurement_idx):
        """
        Performs a step by updating the machine's state based on the transition tensor
        and returns the corresponding measurement.

        Returns:
        The measurement corresponding to the new state.
        """
        old_state = self.current_state
        # Choose the next state based on the transition probabilities of the current state
        self.current_state = np.random.choice(
            self.num_states,
            p=self.transition_tensor[self.current_state, :, measurement_idx]
        )
        # Find the measurement index associated with the new state
        measurement_idx = self.measurement_matrix[old_state, self.current_state, measurement_idx]
        # Return the corresponding measurement
        return self.measurements[measurement_idx], measurement_idx

    def generate_series(self, n, initial_measurement_idx):
        """
        Generates a series of n measurements based on the initial measurement.

        Parameters:
        n (int): The number of measurements to generate.
        initial_measurement: The initial measurement to start with.

        Returns:
        list: A list of generated measurements.
        """
        state_counts = np.zeros(self.num_states)
        initial_measurement = self.measurements[initial_measurement_idx]
        series = [initial_measurement]
        measurement_idx = initial_measurement_idx
        for _ in range(n-1):
            measurement, measurement_idx = self.step(measurement_idx)
            series.append(measurement)
            state_counts[self.current_state] += 1
        state_probabilities = state_counts / (n-1)
        non_zero_probabilities = state_probabilities[state_probabilities > 0]

        # Calculate the entropy
        self.run_entropy = -np.sum(non_zero_probabilities * np.log2(non_zero_probabilities))

        return series

    def random(self, avg_transitions_per_state):
        """
        Initializes T and measurement_matrix with random values based on avg_transitions_per_state.

        Parameters:
        avg_transitions_per_state (float): The average number of transitions per state, used in a Gaussian distribution.

        Returns:
        A tuple containing the initialized transition_tensor and measurement_matrix.
        """
        Q = len(self.measurements)
        T = np.zeros((self.num_states, self.num_states, Q))
        measurement_matrix = np.full((self.num_states, self.num_states, Q), -1)  # Initialize with -1 to indicate unfilled entries

        for i in range(self.num_states):
            for q in range(Q):
                # We pick a random number of transitions for each state based on a Gaussian
                num_transitions = int(np.random.normal(avg_transitions_per_state, 1))
                num_transitions = max(2, min(num_transitions, self.num_states))  # Ensure num_transitions is within valid range
                                                                                # At least 2 transitions so no probability 1 cycles

                j_indices = np.random.choice(self.num_states, size=num_transitions, replace=False)

                for j in j_indices:
                    T[i, j, q] = np.random.rand()  # Assign a random probability
                    measurement_matrix[i, j, q] = np.random.randint(len(self.measurements))

                T[i, :, q] /= T[i, :, q].sum()  # Normalize

        return T, measurement_matrix

    def entropy(self):
        """
        Calculates the entropy of the system based on the empirical probabilities
        of states after simulating the machine (assumes generate_series has already been called)

        We'd like to do a direct calculation via the stationary distribution of the transition matrix,
        but the transition matrix is not guaranteed to be ergodic, and usually isn't :(

        Returns:
        float: The entropy of the system.
        """
        return self.run_entropy

    def visualize(self, filename='epsilon_machine_graph.png'):
        """
        Visualizes the machine by plotting its states as nodes, transitions as edges,
        and labels each edge with 'probability | measurement'.

        Parameters:
        filename (str): The name of the file to save the graph visualization.
        """
        G = nx.DiGraph()

        # Add nodes
        for i in range(self.num_states):
            G.add_node(i)

        # Add edges with labels
        edge_labels = {}
        for i in range(self.num_states):
            for j in range(self.num_states):
                if self.transition_tensor[i, j] > 0:
                    # Probability of transition
                    prob = self.transition_tensor[i, j]
                    # Measurement associated with transition
                    measurement = self.measurements[self.measurement_matrix[i, j]]
                    # Add edge with label
                    G.add_edge(i, j, weight=prob)
                    edge_labels[(i, j)] = f'{prob:.2f} | {measurement}'

        # Draw the graph
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='k', linewidths=1, font_size=15, arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

        plt.savefig(filename)
        plt.close()