import numpy as np
import matplotlib.pyplot as plt
from network import Network

if __name__ == "__main__":
    # Sample frequency and S-parameters data
    frequency = np.linspace(1e9, 10e9, 101)
    s_parameters = np.random.rand(101, 4, 4) + 1j * np.random.rand(101, 4, 4)
    z0 = np.ones((101, 4)) * (50 + 1j * 5)  # Example complex impedance for each port
    
    network1 = Network(frequency, s_parameters, z0, name="Network 1")
    network2 = Network(frequency, s_parameters, z0, name="Network 2")
    print(network1)
    
    # Plot S-parameters
    fig, ax = plt.subplots(figsize=(10, 6))
    network1.plot_s_parameters(fig=fig, ax=ax)
    
    # Compute and plot mixed-mode S-parameters for differential pairs (0,1) and (2,3)
    fig, ax = plt.subplots(figsize=(10, 6))
    network1.plot_mixed_mode_s_parameters(differential_pairs=[(0, 1), (2, 3)], fig=fig, ax=ax)
    
    # Convert to dictionary and back
    network_dict = network1.to_dict()
    new_network = Network.from_dict(network_dict)
    print(new_network)
    
    # Cascade networks
    cascaded_network = network1.cascade_with(network2)
    print(cascaded_network)
    
    # Plot cascaded network's S-parameters
    fig, ax = plt.subplots(figsize=(10, 6))
    cascaded_network.plot_s_parameters(fig=fig, ax=ax)
