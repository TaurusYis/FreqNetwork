# __init__.py to make 'network' a package and facilitate imports
from .network import Network
from .utils import calculate_mixed_mode_s_parameters
from .plotting import plot_s_parameters, plot_mixed_mode_s_parameters

# list the class and all other functions defined outside the class
__all__ = ['Network', 
           'calculate_mixed_mode_s_parameters', 
           'plot_s_parameters', 
           'plot_mixed_mode_s_parameters']
