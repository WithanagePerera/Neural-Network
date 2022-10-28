# Libraries for math and graphing
import numpy as np
import matplotlib.pyplot as pyplot

# Function for initializing dimensional parameters
# which will ebb stored in a dictionary called params
# W1 will represent the weight matrix for layer 1
initialize_parameters(dimensions):
    np.random.seed(3)
    params = {}
    
    # Takes length of parameter
    L = len(dimensions)

    for l in range(1, L):
        params['W' + str(1)] = np.random.randn(layer_dims[1], layer_dims[l-1])*0.01

