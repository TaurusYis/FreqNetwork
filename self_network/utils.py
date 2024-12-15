import numpy as np

def calculate_mixed_mode_s_parameters(network, differential_pairs):
    """
    Compute the mixed-mode S-parameters for the specified differential pairs.

    Parameters:
    network (Network): The Network object containing frequency, S-parameters, and characteristic impedance.
    differential_pairs (list of tuples): List of differential pairs, where each tuple contains the indices of the positive and negative ports.

    Returns:
    numpy array: Mixed-mode S-parameters.
    """
    n_ports = network.s_parameters.shape[1]
    if any(max(pair) >= n_ports for pair in differential_pairs):
        raise ValueError("Differential pair indices must be within the number of ports")
    
    # Create the transformation matrix T for mixed-mode conversion
    T = np.zeros((2 * len(differential_pairs), n_ports))
    for i, (p, n) in enumerate(differential_pairs):
        T[2*i, p] = 1
        T[2*i, n] = 1
        T[2*i+1, p] = 1
        T[2*i+1, n] = -1
    T *= 0.5
    
    # Calculate the pseudo-inverse of the transformation matrix
    T_inv = np.linalg.pinv(T)
    
    # Compute mixed-mode S-parameters for each frequency point
    s_mm = np.empty((len(network.frequency), 2 * len(differential_pairs), 2 * len(differential_pairs)), dtype=complex)
    for k in range(len(network.frequency)):
        s_mm[k] = T @ network.s_parameters[k] @ T_inv
    
    return s_mm
