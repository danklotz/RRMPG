# -*- coding: utf-8 -*-
# This file is part of RRMPG.
#
# RRMPG is free software with the aim to provide a playground for experiments
# with hydrological rainfall-runoff-models while achieving competitive
# performance results.
#
# You should have received a copy of the MIT License along with RRMPG. If not,
# see <https://opensource.org/licenses/MIT>

import numpy as np

from numba import njit

@njit
def run_linres(prec, initial_state, params):
    """Implementation of a linear reservoir.

    This function should be called via the .simulate() function of the linres
    class and not directly! It is kept in a separate file to avoid confusion
    in case anyone wants to inspect the actual model routine.

    Args:
        prec: Numpy [t] array, which contains the precipitation input.
        initial_state: Scalar for the intial state of the storage.
        params: Numpy array of custom dtype, which contains the model parameter.

    Returns:
        qsim: Numpy [t] array with the simulated streamflow.
        storage: Numpy [t] array with the state of the storage of each timestep.

    """
    # Number of simulation timesteps
    num_timesteps = len(prec)

    # Unpack model parameters
    k = params['k']

    # Initialize array for the simulated stream flow and the storage
    qsim = np.zeros(num_timesteps, np.float64)
    storage = np.zeros(num_timesteps, np.float64)

    # Set the initial storage value
    storage[0] = initial_state

    # Model simulation
    for t in range(1, num_timesteps):

        # Calculate the streamflow
        qsim[t] = prec[t] - (k*prec[t]-storage[t-1])*(1.0-np.exp(-k))
        # Update the storage
        storage[t] = qsim[t] - qsim[t-1]

    return qsim, storage
