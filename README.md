# Network Processing and Visualization

## Project Overview

A simplified class to process touchstone files (.snp) representing frequency responses. This project targets the processing and visualization of frequency responses, unlike some more powerful tools, like ADS Python and scikit-rf.

This class is designed to perform simple processes such as computing mixed-mode S-parameters, de-embedding using T-matrix inversion, etc. The main feature of this project is its ability to execute with an Excel file acting as a control panel to allow users to automate the process.

## Directory Structure

network/
init.py
network.py
utils.py
plotting.py
main.py


## Features

- Compute mixed-mode S-parameters for differential pairs
- De-embed networks using T-matrix inversion
- Cascade networks
- Plot S-parameters and mixed-mode S-parameters
- Convert to and from skrf.Network objects

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/your_username/your_repository.git
    ```

2. Navigate to the project directory:
    ```
    cd your_repository
    ```

3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage

1. Create a `Network` object by providing frequency, S-parameters, and characteristic impedance data:
    ```python
    from network import Network
    import numpy as np

    frequency = np.linspace(1e9, 10e9, 101)
    s_parameters = np.random.rand(101, 4, 4) + 1j * np.random.rand(101, 4, 4)
    z0 = np.ones((101, 4)) * (50 + 1j * 5)

    network = Network(frequency, s_parameters, z0, name="Example Network")
    ```

2. Plot S-parameters:
    ```python
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    network.plot_s_parameters(fig=fig, ax=ax)
    ```

3. Compute and plot mixed-mode S-parameters:
    ```python
    fig, ax = plt.subplots(figsize=(10, 6))
    network.plot_mixed_mode_s_parameters(differential_pairs=[(0, 1), (2, 3)], fig=fig, ax=ax)
    ```

4. Cascade networks:
    ```python
    network1 = Network(frequency, s_parameters, z0, name="Network 1")
    network2 = Network(frequency, s_parameters, z0, name="Network 2")

    cascaded_network = network1.cascade_with(network2)
    ```

5. Convert to and from `skrf.Network`:
    ```python
    import skrf as rf

    skrft_network = rf.Network(f=frequency, s=s_parameters, z0=z0, name="skrf Network")
    network_from_skrft = Network.from_skrft(skrft_network)
    skrft_network_converted_back = network_from_skrft.to_skrft()
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
