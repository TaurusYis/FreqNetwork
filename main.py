import numpy as np
import matplotlib.pyplot as plt
from self_network import Network

if __name__ == "__main__":
    # Sample frequency and S-parameters data
    frequency = np.linspace(1e9, 10e9, 101)
    s_parameters = np.random.rand(101, 4, 4) + 1j * np.random.rand(101, 4, 4)
    characteristic_impedance = np.ones((101, 4)) * (50 + 1j * 5)  # Example complex impedance for each port
    
    network = Network(frequency, s_parameters, characteristic_impedance, name="Example Network")
    print(network)
    
    # Plot S-parameters
    fig, ax = plt.subplots(figsize=(10, 6))
    network.plot_s_parameters(fig=fig, ax=ax)
    
    # Compute and plot mixed-mode S-parameters for differential pairs (0,1) and (2,3)
    fig, ax = plt.subplots(figsize=(10, 6))
    network.plot_mixed_mode_s_parameters(differential_pairs=[(0, 1), (2, 3)], fig=fig, ax=ax)
    
    # Convert to dictionary and back
    network_dict = network.to_dict()
    new_network = Network.from_dict(network_dict)
    print(new_network)
