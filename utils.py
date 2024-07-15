import numpy as np
import skrf as rf

def calculate_mixed_mode_s_parameters(network, differential_pairs):
    n_ports = network.s_parameters.shape[1]
    if any(max(pair) >= n_ports for pair in differential_pairs):
        raise ValueError("Differential pair indices must be within the number of ports")
    
    T = np.zeros((2 * len(differential_pairs), n_ports))
    for i, (p, n) in enumerate(differential_pairs):
        T[2*i, p] = 1
        T[2*i, n] = 1
        T[2*i+1, p] = 1
        T[2*i+1, n] = -1
    T *= 0.5
    
    T_inv = np.linalg.pinv(T)
    
    s_mm = np.empty((len(network.frequency), 2 * len(differential_pairs), 2 * len(differential_pairs)), dtype=complex)
    for k in range(len(network.frequency)):
        s_mm[k] = T @ network.s_parameters[k] @ T_inv
    
    return s_mm

def s_to_t(s_matrix):
    n_ports = s_matrix.shape[1]
    I = np.eye(n_ports)
    t_matrix = np.empty_like(s_matrix, dtype=complex)
    for k in range(s_matrix.shape[0]):
        s11 = s_matrix[k, :n_ports//2, :n_ports//2]
        s12 = s_matrix[k, :n_ports//2, n_ports//2:]
        s21 = s_matrix[k, n_ports//2:, :n_ports//2]
        s22 = s_matrix[k, n_ports//2:, n_ports//2:]
        
        t_matrix[k, :n_ports//2, :n_ports//2] = np.linalg.inv(I - s22) @ s21
        t_matrix[k, :n_ports//2, n_ports//2:] = np.linalg.inv(I - s22) @ s11
        t_matrix[k, n_ports//2:, :n_ports//2] = s12 @ np.linalg.inv(I - s22) @ s21 - s22
        t_matrix[k, n_ports//2:, n_ports//2:] = s12 @ np.linalg.inv(I - s22) @ s11 - I
    
    return t_matrix

def t_to_s(t_matrix):
    n_ports = t_matrix.shape[1]
    I = np.eye(n_ports)
    s_matrix = np.empty_like(t_matrix, dtype=complex)
    for k in range(t_matrix.shape[0]):
        t11 = t_matrix[k, :n_ports//2, :n_ports//2]
        t12 = t_matrix[k, :n_ports//2, n_ports//2:]
        t21 = t_matrix[k, n_ports//2:, :n_ports//2]
        t22 = t_matrix[k, n_ports//2:, n_ports//2:]
        
        s_matrix[k, :n_ports//2, :n_ports//2] = t12 @ np.linalg.inv(I + t22)
        s_matrix[k, :n_ports//2, n_ports//2:] = t11 - t12 @ np.linalg.inv(I + t22) @ t21
        s_matrix[k, n_ports//2:, :n_ports//2] = np.linalg.inv(I + t22) @ t21
        s_matrix[k, n_ports//2:, n_ports//2:] = np.linalg.inv(I + t22) @ t22
    
    return s_matrix

def cascade_networks(network1, network2):
    t1 = network1.s_to_t()
    t2 = network2.s_to_t()
    t_cascaded = np.empty_like(t1, dtype=complex)
    
    for k in range(t1.shape[0]):
        t_cascaded[k] = t1[k] @ t2[k]
    
    s_cascaded = network1.t_to_s(t_cascaded)
    
    return Network(
        frequency=network1.frequency,
        s_parameters=s_cascaded,
        z0=network1.z0,  # Assuming same characteristic impedance for simplicity
        name=f"{network1.name}_cascaded_{network2.name}"
    )

def from_skrft(skrft_network):
    frequency = skrft_network.f
    s_parameters = skrft_network.s
    z0 = skrft_network.z0
    name = skrft_network.name
    return Network(frequency, s_parameters, z0, name)

def to_skrft(network):
    return rf.Network(
        f=network.frequency,
        s=network.s_parameters,
        z0=network.z0,
        name=network.name
    )
