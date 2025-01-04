import numpy as np
import skrf as rf

def from_skrft(skrft_network):
    frequency = skrft_network.f
    s_parameters = skrft_network.s
    z0 = skrft_network.z0
    name = skrft_network.name
    from .network import Network
    return Network(frequency, s_parameters, z0, name)

def to_skrft(network):
    return rf.Network(
        f=network.frequency,
        s=network.s_parameters,
        z0=network.z0,
        name=network.name
    )

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
    """
    Generalized conversion of S-parameters to T-parameters for N-port networks.

    Parameters:
    s_matrix : ndarray
        S-parameter matrix of shape (n_freq, n_ports, n_ports).

    Returns:
    ndarray
        T-parameter matrix of shape (n_freq, n_ports, n_ports).
    """
    n_freq, n_ports, _ = s_matrix.shape
    t_matrix = np.zeros_like(s_matrix, dtype=complex)
    z_matrix = s_to_z(s_matrix)  # Convert S-parameters to Z-parameters

    for f in range(n_freq):
        n = n_ports // 2

        # s = s_matrix[f]
        # # Split S-matrix into quadrants
        # s11, s12, s21, s22 = s[:n, :n], s[:n, n:], s[n:, :n], s[n:, n:]

        # # Check invertibility of S12
        # if np.linalg.det(s12) == 0:
        #     raise ValueError("S12 matrix is not invertible.")

        # # Calculate T-matrix quadrants
        # t11 = np.linalg.inv(s21)
        # t12 = -t11 @ s22
        # t21 = s11 @ t11
        # t22 = s12 + s11 @ t12

        # # Calculate the T-matrix quadrants
        # t11 = -np.linalg.inv(s12) @ s11
        # t12 = np.linalg.inv(s12)
        # t21 = s22 - s21 @ np.linalg.inv(s12) @ s11
        # t22 = s21 @ np.linalg.inv(s12)

        z = z_matrix[f]

        # Split Z-matrix into quadrants
        z11, z12, z21, z22 = z[:n, :n], z[:n, n:], z[n:, :n], z[n:, n:]
        # Check invertibility of S12
        if np.linalg.det(z21) == 0:
            raise ValueError("z21 matrix is not invertible.")
        z21_inv = np.linalg.inv(z21)
        # Calculate T-matrix quadrants
        t11 = z11 @ z21_inv
        t12 = z11 @ z22 @ z21_inv - z12
        t21 = z21_inv
        t22 = z22 @ z21_inv

        # Combine T-matrix quadrants
        t_matrix[f] = np.block([[t11, t12], [t21, t22]])

    return t_matrix

def t_to_s(t_matrix):
    """
    Generalized conversion of T-parameters to S-parameters for N-port networks.

    Parameters:
    t_matrix : ndarray
        T-parameter matrix of shape (n_freq, n_ports, n_ports).

    Returns:
    ndarray
        S-parameter matrix of shape (n_freq, n_ports, n_ports).
    """
    n_freq, n_ports, _ = t_matrix.shape
    # s_matrix = np.zeros_like(t_matrix, dtype=complex)
    z_matrix = np.zeros_like(t_matrix, dtype=complex)
    for f in range(n_freq):
        t = t_matrix[f]
        n = n_ports // 2

        # Split T-matrix into quadrants
        t11, t12, t21, t22 = t[:n, :n], t[:n, n:], t[n:, :n], t[n:, n:]

        # # Check invertibility of T12
        # if np.linalg.det(t12) == 0:
        #     raise ValueError("T12 matrix is not invertible.")

        # # Calculate S-matrix quadrants
        # s11 = np.linalg.inv(t12) @ t11
        # s12 = np.linalg.inv(t12)
        # s21 = t21 - t22 @ np.linalg.inv(t12) @ t11
        # s22 = t22 @ np.linalg.inv(t12)

        # # Combine S-matrix quadrants
        # s_matrix[f] = np.block([[s11, s12], [s21, s22]])

        # Check invertibility of T21
        if np.linalg.det(t21) == 0:
            raise ValueError("T21 matrix is not invertible.")
        # calculate Z-matrix quadrants
        t21_inv = np.linalg.inv(t21)
        z11 = t11 @ t21_inv
        z12 = t11 @ t22 @ t21_inv - t12
        z21 = t21_inv
        z22 = t22 @ t21_inv
        z_matrix[f] = np.block([[z11, z12], [z21, z22]])
    
    # convert Z-parameters to S-parameters    
    s_matrix = z_to_s(z_matrix)

    return s_matrix

def cascade_networks(network1, network2):
    """
    Cascades two networks by converting their S-parameters to T-parameters,
    performing the cascade operation, and converting back to S-parameters.

    Parameters:
    network1 : Network
        The first Network object to cascade.
    network2 : Network
        The second Network object to cascade.

    Returns:
    Network
        A new Network object representing the cascaded system.
    """
    # if network1.n_ports != 2 or network2.n_ports != 2:
    #     raise ValueError("Cascading is only supported for 2-port networks.")

    if not np.allclose(network1.frequency, network2.frequency):
        raise ValueError("Networks must have the same frequency points to cascade.")

    # Convert S-parameters to T-parameters
    t1 = s_to_t(network1.s_parameters)
    t2 = s_to_t(network2.s_parameters)

    # Perform the cascade operation (matrix multiplication)
    t_cascaded = np.zeros_like(t1, dtype=complex)
    for f in range(t1.shape[0]):
        t_cascaded[f] = np.matmul(t1[f], t2[f])

    # Convert back to S-parameters
    s_cascaded = t_to_s(t_cascaded)

    # Create the cascaded network
    from .network import Network
    cascaded_network = Network(
        frequency=network1.frequency,
        s_parameters=s_cascaded,
        z0=network1.z0,  # Assuming identical characteristic impedance
        name=f"Cascade({network1.name}, {network2.name})"
    )

    return cascaded_network


def scale_s_parameters(s_parameters, old_z0, new_z0):
    scaled_s = np.empty_like(s_parameters, dtype=complex)
    for f in range(len(s_parameters)):
        gamma_old = (old_z0[f] - 50) / (old_z0[f] + 50)
        gamma_new = (new_z0[f] - 50) / (new_z0[f] + 50)
        scaling_matrix = np.diag(gamma_new / gamma_old)
        scaled_s[f] = scaling_matrix @ s_parameters[f] @ np.linalg.inv(scaling_matrix)
    return scaled_s

def renormalize_s_parameters(s_matrix, z_old, z_new):
    """
    Renormalizes S-parameters to a new characteristic impedance.

    Parameters:
    s_matrix : ndarray
        Original S-parameters of shape (n_freq, n_ports, n_ports).
    z_old : ndarray
        Original characteristic impedance matrix of shape (n_freq, n_ports).
    z_new : ndarray
        New characteristic impedance matrix of shape (n_freq, n_ports).

    Returns:
    ndarray
        Renormalized S-parameters of shape (n_freq, n_ports, n_ports).
    """
    z_old_matrix = s_to_z(s_matrix, z_old)
    renormalized_s = z_to_s(z_old_matrix, z_new)

    return renormalized_s

def s_to_z(s_matrix, z0):
    """
    Converts S-parameters to Z-parameters.

    Parameters:
    s_matrix : ndarray
        S-parameter matrix of shape (n_freq, n_ports, n_ports).
    z0 : scalar, array, or ndarray
        Characteristic impedance, can be:
        - Scalar: common impedance for all ports
        - Array: shape (n_freq,) for a common impedance per frequency
        - Matrix: shape (n_freq, n_ports) for per-port impedance per frequency

    Returns:
    ndarray
        Z-parameter matrix of shape (n_freq, n_ports, n_ports).
    """
    n_freq, n_ports, _ = s_matrix.shape
    z_matrix = np.zeros_like(s_matrix, dtype=complex)

    for f in range(n_freq):
        s = s_matrix[f]
        if np.isscalar(z0):
            z0_diag = np.eye(n_ports) * z0
        elif z0.ndim == 1:
            z0_diag = np.eye(n_ports) * z0[f]
        else:
            z0_diag = np.diag(z0[f])

        identity = np.eye(n_ports)
        z_matrix[f] = z0_diag @ (identity + s) @ np.linalg.inv(identity - s)

    return z_matrix

def z_to_s(z_matrix, z0):
    """
    Converts Z-parameters to S-parameters.

    Parameters:
    z_matrix : ndarray
        Z-parameter matrix of shape (n_freq, n_ports, n_ports).
    z0 : scalar, array, or ndarray
        Characteristic impedance, can be:
        - Scalar: common impedance for all ports
        - Array: shape (n_freq,) for a common impedance per frequency
        - Matrix: shape (n_freq, n_ports) for per-port impedance per frequency

    Returns:
    ndarray
        S-parameter matrix of shape (n_freq, n_ports, n_ports).
    """
    n_freq, n_ports, _ = z_matrix.shape
    s_matrix = np.zeros_like(z_matrix, dtype=complex)

    for f in range(n_freq):
        z = z_matrix[f]
        if np.isscalar(z0):
            z0_diag = np.eye(n_ports) * z0
        elif z0.ndim == 1:
            z0_diag = np.eye(n_ports) * z0[f]
        else:
            z0_diag = np.diag(z0[f])

        s_matrix[f] = (z - z0_diag) @ np.linalg.inv(z + z0_diag)

    return s_matrix

if __name__ == "__main__":
    # Validate functions with scikit-rf
    freq = rf.Frequency(1, 10, 10, unit='ghz')
    network = rf.Network(frequency=freq, s=np.random.rand(10, 2, 2) + 1j * np.random.rand(10, 2, 2), z0=[50, 50])

    # Convert S to T and back to S
    # t_params = s_to_t(network.s)
    # recovered_s = t_to_s(t_params)
    # print("Original S-parameters:")
    # print(network.s)
    # print("Recovered S-parameters:")
    # print(recovered_s)

    # Convert S to Z and back to S
    z_params = s_to_z(network.s, 50)
    z_ref = rf.s2z(network.s, 50)
    assert np.allclose(z_params, z_ref, atol=1e-6), \
        "s_to_z does not match scikit-rf's s2z function!"
    print("s_to_z matches scikit-rf's s2z function.")   
    recovered_s = z_to_s(z_ref, 50)
    assert np.allclose(recovered_s, network.s, atol=1e-6), \
        "z_to_s does not match the original S-parameters!"
    print("z_to_s matches the original S-parameters.") 
    # Example of cascading networks