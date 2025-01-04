# %%
import numpy as np
import matplotlib.pyplot as plt
import self_network
from self_network import Network
import skrf as rf
from skrf.media import Coaxial, MLine, CPW

def generate_transmission_line(length=0.1, z0=50, frequency_range=(1e9, 10e9), n_points=501, line_type="coaxial"):
    """
    Generate a transmission line network using scikit-rf.

    Parameters:
    length : float
        Length of the transmission line in meters.
    z0 : float
        Characteristic impedance of the transmission line in ohms.
    frequency_range : tuple
        Frequency range as (start, stop) in Hz.
    n_points : int
        Number of frequency points.
    line_type : str
        Type of transmission line. Options: "coaxial", "microstrip".

    Returns:
    Network
        Custom Network class object representing the transmission line.
    """
    # Generate frequency points
    frequency = rf.Frequency(start=frequency_range[0], stop=frequency_range[1], npoints=n_points, unit='hz')
    
    # Create a transmission line
    if line_type == "coaxial":
        medium = Coaxial(frequency, z0=z0)
    elif line_type == "microstrip":
        medium = MLine(frequency, z0=z0)
    elif line_type == "cpw":
        ep_r = 4.421
        tanD = 0.0167
        medium = CPW(frequency=frequency, w = 1.7e-3, s = 0.5e-3, t = 50e-6, h = 1.55e-3,
                ep_r = ep_r, tand = tanD, rho = 1.7e-8, z0_port = z0, has_metal_backside = True)
    else:
        raise ValueError(f"Unsupported line_type '{line_type}'. Use 'coaxial' or 'microstrip'.")
    
    # Create a transmission line object
    tline = medium.line(length, unit='m')
    
    # Convert to custom Network class
    custom_network = Network(
        frequency=tline.frequency.f,
        s_parameters=tline.s,
        z0=np.full((n_points, 2), z0),
        name=f"{line_type}_line_{length}m"
    )
    
    return custom_network, tline

if __name__ == "__main__":
    # Generate example transmission line networks
    coax_tline, coax_tline_rf = generate_transmission_line(length=0.1, z0=50, line_type="coaxial")
    microstrip_tline, microstrip_tline_rf = generate_transmission_line(length=0.05, z0=40, line_type="microstrip")
    
    # Plot the S-parameters of the coaxial line
    # fig, ax = plt.subplots(figsize=(10, 6))
    # coax_tline.plot_s_parameters(fig=fig, ax=ax)
    coax_tline.plot_s_parameters()
    # plt.show()
    cpw_line_1, cpw_line_1_rf = generate_transmission_line(length=0.05, z0=50, line_type="cpw")
    cpw_line_2, cpw_line_2_rf = generate_transmission_line(length=0.05, z0=50, line_type="cpw")

    cpw_line_1.plot_s_parameters()
    cpw_line_2.plot_s_parameters()
    cascaded_network = cpw_line_1.cascade_with(cpw_line_2)
    cascaded_network.plot_s_parameters()

    # Validate with scikit-rf's cascade function
    rf_cascaded = cpw_line_1_rf ** cpw_line_2_rf
    rf_cascaded.plot_s_db()
    # Ensure consistency
    assert np.allclose(cascaded_network.s_parameters, rf_cascaded.s, atol=1e-6), \
        "Cascaded S-parameters do not match!"
    print("Cascading works as expected!")
    # %%
    # Sample frequency and S-parameters data
    frequency = np.linspace(1e9, 10e9, 101)
    s_parameters = np.random.rand(101, 4, 4) + 1j * np.random.rand(101, 4, 4)
    Z0 = np.ones((101, 4)) * (50 + 1j * 5)  # Example complex impedance for each port
    
    network = Network(frequency, s_parameters, Z0, name="Example Network")
    
    # Plot S-parameters
    fig, ax = plt.subplots(figsize=(10, 6))
    network.plot_s_parameters(fig=fig, ax=ax)
    
    # Compute and plot mixed-mode S-parameters for differential pairs (0,1) and (2,3)
    fig, ax = plt.subplots(figsize=(10, 6))
    network.plot_mixed_mode_s_parameters(differential_pairs=[(0, 1), (2, 3)], fig=fig, ax=ax)
    
    # Convert to dictionary and back
    network_dict = network.to_dict()
    new_network = Network.from_dict(network_dict)
    # print(new_network)
