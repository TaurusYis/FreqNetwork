import numpy as np
from .plotting import plot_s_parameters, plot_mixed_mode_s_parameters
from .utils import (
    calculate_mixed_mode_s_parameters, 
    s_to_t, 
    t_to_s, 
    cascade_networks,
    from_skrft,
    to_skrft
)

class Network:
    def __init__(self, frequency, s_parameters, z0, name=""):
        self.frequency = np.array(frequency)
        self.s_parameters = np.array(s_parameters)
        self.z0 = np.array(z0)
        self.name = name
        
        if self.s_parameters.shape[0] != len(self.frequency):
            raise ValueError("The number of frequency points must match the first dimension of s_parameters")
        if self.z0.shape != (len(self.frequency), self.s_parameters.shape[1]):
            raise ValueError("The shape of z0 must be (n_freq, n_ports)")
        
    def __repr__(self):
        return (f"Network(name={self.name}, frequency={self.frequency}, s_parameters={self.s_parameters.shape}, "
                f"z0={self.z0.shape})")
    
    def plot_s_parameters(self, fig=None, ax=None):
        plot_s_parameters(self, fig, ax)
    
    def to_dict(self):
        return {
            'name': self.name,
            'frequency': self.frequency.tolist(),
            's_parameters': self.s_parameters.tolist(),
            'z0': self.z0.tolist()
        }
    
    @staticmethod
    def from_dict(data):
        return Network(
            name=data.get('name', ""),
            frequency=data['frequency'],
            s_parameters=data['s_parameters'],
            z0=data['z0']
        )
    
    def plot_mixed_mode_s_parameters(self, differential_pairs, fig=None, ax=None):
        plot_mixed_mode_s_parameters(self, differential_pairs, fig, ax)

    def mixed_mode_s_parameters(self, differential_pairs):
        return calculate_mixed_mode_s_parameters(self, differential_pairs)
    
    def s_to_t(self):
        return s_to_t(self.s_parameters)
    
    def t_to_s(self, t_matrix):
        return t_to_s(t_matrix)
    
    def cascade_with(self, other_network):
        if not np.array_equal(self.frequency, other_network.frequency):
            raise ValueError("Frequency points of both networks must match.")
        return cascade_networks(self, other_network)
    
    @staticmethod
    def from_skrft(skrft_network):
        return from_skrft(skrft_network)
    
    def to_skrft(self):
        return to_skrft(self)
