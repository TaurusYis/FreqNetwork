import numpy as np
from .plotting import *
from .utils import *

class Network:
    def __init__(self, frequency, s_parameters, z0, name=""):
        """
        Initialize the Network class.

        Parameters:
        frequency (array-like): Frequencies at which the network parameters are defined.
        s_parameters (array-like): S-parameters of the network. Should be a 3D array with shape (n_freq, n_ports, n_ports).
        z0 (array-like): Characteristic impedance of the network. Should be a 2D array with shape (n_freq, n_ports).
        name (str): Name of the network.
        """
        self.frequency = np.array(frequency)
        self.s_parameters = np.array(s_parameters)
        self.z0 = np.array(z0)
        self.name = name
        
        # Validate dimensions
        if self.s_parameters.shape[0] != len(self.frequency):
            raise ValueError("The number of frequency points must match the first dimension of s_parameters")
        if self.z0.shape != (len(self.frequency), self.s_parameters.shape[1]):
            raise ValueError("The shape of z0 must be (n_freq, n_ports)")
        n_freq, n_ports, _ = self.s_parameters.shape
        self.n_freq = n_freq
        self.n_ports = n_ports
    def __repr__(self):
        return (f"Network(name={self.name}, frequency={self.frequency}, s_parameters={self.s_parameters.shape}, "
                f"z0={self.z0.shape})")
    
    def plot_s_parameters(self, fig=None, ax=None):
        """Plot the S-parameters of the network on the provided figure and axes."""
        plot_s_parameters(self, fig, ax)
    
    def to_dict(self):
        """Return the network data as a dictionary."""
        return {
            'name': self.name,
            'frequency': self.frequency.tolist(),
            's_parameters': self.s_parameters.tolist(),
            'z0': self.z0.tolist()
        }
    
    @staticmethod
    def from_dict(data):
        """Create a Network instance from a dictionary."""
        return Network(
            name=data.get('name', ""),
            frequency=data['frequency'],
            s_parameters=data['s_parameters'],
            z0=data['z0']
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

    def s_to_t(self):
        return s_to_t(self.s_parameters)

    def t_to_s(self, t_matrix):
        return t_to_s(t_matrix)

    def cascade_with(self, other_network):
        if not np.array_equal(self.frequency, other_network.frequency):
            raise ValueError("Frequency points of both networks must match.")
        return cascade_networks(self, other_network)
        
    def scale_s_parameters(self, new_z0):
        self.s_parameters = scale_s_parameters(self.s_parameters, self.z0, new_z0)
        self.z0 = new_z0

    def renormalize_s_parameters(self, new_z0):
        self.s_parameters = renormalize_s_parameters(self.s_parameters, self.z0, new_z0)
        self.z0 = new_z0

    @staticmethod
    def from_skrft(skrft_network):
        return from_skrft(skrft_network)

    def to_skrft(self):
        return to_skrft(self)