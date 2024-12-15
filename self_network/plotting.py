import numpy as np
import matplotlib.pyplot as plt

def plot_s_parameters(network, fig=None, ax=None):
    """
    Plot the S-parameters of the network on the provided figure and axes.

    Parameters:
    network (Network): The Network object containing frequency, S-parameters, and characteristic impedance.
    fig: Matplotlib figure object for plotting.
    ax: Matplotlib axes object for plotting.
    """
    if ax is None:
        # Create a grid of subplots if no axes are provided
        fig, axes = plt.subplots(nrows=network.s_parameters.shape[1], ncols=network.s_parameters.shape[2], figsize=(15, 10))
        for i in range(network.s_parameters.shape[1]):
            for j in range(network.s_parameters.shape[2]):
                axes[i, j].plot(network.frequency, 20 * np.log10(np.abs(network.s_parameters[:, i, j])), label=f"S{i+1}{j+1}")
                axes[i, j].set_xlabel('Frequency (Hz)')
                axes[i, j].set_ylabel('Magnitude (dB)')
                axes[i, j].set_title(f'S{i+1}{j+1}')
                axes[i, j].grid(True)
        fig.suptitle(f'S-parameters: {network.name}')
        plt.tight_layout()
        plt.show()
    else:
        # Plot on the provided axes
        for i in range(network.s_parameters.shape[1]):
            for j in range(network.s_parameters.shape[2]):
                ax.plot(network.frequency, 20 * np.log10(np.abs(network.s_parameters[:, i, j])), label=f"{network.name} S{i+1}{j+1}")
        ax.legend()

def plot_mixed_mode_s_parameters(network, differential_pairs, fig=None, ax=None):
    """
    Compute and plot the mixed-mode S-parameters for the specified differential pairs.

    Parameters:
    network (Network): The Network object containing frequency, S-parameters, and characteristic impedance.
    differential_pairs (list of tuples): List of differential pairs, where each tuple contains the indices of the positive and negative ports.
    fig: Matplotlib figure object for plotting.
    ax: Matplotlib axes object for plotting.
    """
    mixed_s_parameters = network.mixed_mode_s_parameters(differential_pairs)
    labels = ['dd', 'dc', 'cd', 'cc']
    if ax is None:
        # Create a grid of subplots if no axes are provided
        fig, axes = plt.subplots(nrows=2 * len(differential_pairs), ncols=2 * len(differential_pairs), figsize=(15, 10))
        for i in range(2 * len(differential_pairs)):
            for j in range(2 * len(differential_pairs)):
                label = labels[i % 2] + labels[j % 2] + f"{(i//2)+1}{(j//2)+1}"
                axes[i, j].plot(network.frequency, 20 * np.log10(np.abs(mixed_s_parameters[:, i, j])), label=label)
                axes[i, j].set_xlabel('Frequency (Hz)')
                axes[i, j].set_ylabel('Magnitude (dB)')
                axes[i, j].set_title(label)
                axes[i, j].grid(True)
        fig.suptitle(f'Mixed-mode S-parameters: {network.name}')
        plt.tight_layout()
        plt.show()
    else:
        # Plot on the provided axes
        for i in range(2 * len(differential_pairs)):
            for j in range(2 * len(differential_pairs)):
                label = labels[i % 2] + labels[j % 2] + f"{(i//2)+1}{(j//2)+1}"
                ax.plot(network.frequency, 20 * np.log10(np.abs(mixed_s_parameters[:, i, j])), label=f"{network.name} {label}")
        ax.legend()
