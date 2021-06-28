#######
#  ______   ________       _       ____    ____   ______   
# |_   _ \ |_   __  |     / \     |_   \  /   _|.' ____ \  
#   | |_) |  | |_ \_|    / _ \      |   \/   |  | (___ \_| 
#   |  __'.  |  _| _    / ___ \     | |\  /| |   _.____`.  
#  _| |__) |_| |__/ | _/ /   \ \_  _| |_\/_| |_ | \____) | 
# |_______/|________||____| |____||_____||_____| \______.' 
#                                                         
#######
# Project Numerics, COSSE Programme 2021
# Carsten van de Kamp, Sergi Andreu,
# Sebastian Myrbäck, Dylan Everingham
# 27.06.21
#
# Constants.py
# File containing constants used as default problem parameters.
#######

# For FEM solvers (static and dynamic), i.e. classes in
# NumericalSolutions.py and DynamicSolutions.py
DEFAULT_a = 0
DEFAULT_b = 0
DEFAULT_QL = 0
DEFAULT_M0 = 0
DEFAULT_ML = 0
DEFAULT_a0 = 0
DEFAULT_aL = 0
DEFAULT_E = 1
DEFAULT_I = 1
DEFAULT_N = 25
DEFAULT_L = 1
DEFAULT_q = lambda x: x

# Additional parameters for analytical soolutions, i.e. classes in
# AnalyticalSolutions.py. Also use those above
DEFAULT_case = 'arbitrary function'
DEFAULT_c = -0.01
DEFAULT_x0 = 0

# For dynamic case FEM solvers only, i.e. classes in
# DynamicSolutions.py

DEFAULT_u1 = 0
DEFAULT_up1 = 0
DEFAULT_upp1 = 0
DEFAULT_h = 0.1
DEFAULT_beta = 1.0/4
DEFAULT_gamma = 1.0/2
DEFAULT_Me = 1
DEFAULT_Se = 1
DEFAULT_f = 1