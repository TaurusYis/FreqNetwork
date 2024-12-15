import numpy as np
from .plotting import plot_s_parameters, plot_mixed_mode_s_parameters
from .utils import calculate_mixed_mode_s_parameters

class Network:
    def __init__(self, frequency, s_parameters, characteristic_impedance, name=""):
        """
        Initialize the Network class.

        Parameters:
        frequency (array-like): Frequencies at which the network parameters are defined.
        s_parameters (array-like): S-parameters of the network. Should be a 3D array with shape (n_freq, n_ports, n_ports).
        characteristic_impedance (array-like): Characteristic impedance of the network. Should be a 2D array with shape (n_freq, n_ports).
        name (str): Name of the network.
        """
        self.frequency = np.array(frequency)
        self.s_parameters = np.array(s_parameters)
        self.characteristic_impedance = np.array(characteristic_impedance)
        self.name = name
        
        # Validate dimensions
        if self.s_parameters.shape[0] != len(self.frequency):
            raise ValueError("The number of frequency points must match the first dimension of s_parameters")
        if self.characteristic_impedance.shape != (len(self.frequency), self.s_parameters.shape[1]):
            raise ValueError("The shape of characteristic_impedance must be (n_freq, n_ports)")
        
    def __repr__(self):
        return (f"Network(name={self.name}, frequency={self.frequency}, s_parameters={self.s_parameters.shape}, "
                f"characteristic_impedance={self.characteristic_impedance.shape})")
    
    def plot_s_parameters(self, fig=None, ax=None):
        """Plot the S-parameters of the network on the provided figure and axes."""
        plot_s_parameters(self, fig, ax)
    
    def to_dict(self):
        """Return the network data as a dictionary."""
        return {
            'name': self.name,
            'frequency': self.frequency.tolist(),
            's_parameters': self.s_parameters.tolist(),
            'characteristic_impedance': self.characteristic_impedance.tolist()
        }
    
    @staticmethod
    def from_dict(data):
        """Create a Network instance from a dictionary."""
        return Network(
            name=data.get('name', ""),
            frequency=data['frequency'],
            s_parameters=data['s_parameters'],
            characteristic_impedance=data['characteristic_impedance']
        )
    
    def plot_mixed_mode_s_parameters(self, differential_pairs, fig=None, ax=None):
        """
        Compute and plot the mixed-mode S-parameters for the specified differential pairs.

        Parameters:
        differential_pairs (list of tuples): List of differential pairs, where each tuple contains the indices of the positive and negative ports.
        fig: Matplotlib figure object for plotting.
        ax: Matplotlib axes object for plotting.
        """
        plot_mixed_mode_s_parameters(self, differential_pairs, fig, ax)

    def mixed_mode_s_parameters(self, differential_pairs):
        """
        Compute the mixed-mode S-parameters for the specified differential pairs.

        Parameters:
        differential_pairs (list of tuples): List of differential pairs, where each tuple contains the indices of the positive and negative ports.

        Returns:
        numpy array: Mixed-mode S-parameters.
        """
        return calculate_mixed_mode_s_parameters(self, differential_pairs)
