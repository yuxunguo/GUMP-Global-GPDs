"""
The minimizer using iMinuit, which takes 1-D array for the input parameters only.

Extra efforts needed to convert the form of the parameters.

"""
# Number of GPD species, 4 leading-twist GPDs including H, E Ht, Et are needed.
NumofGPDSpecies = 4
# Number of flavor factor, Flavor_Factor = 2 * nf + 1 needed including 2 * nf quark (antiquark) and one gluon
Flavor_Factor = 2 * 2 + 1
# Number of ansatz, 1 set of (N, alpha, beta, alphap) will be used to start with
init_NumofAnsatz = 1
# Size of one parameter set, a set of parameters (N, alpha, beta, alphap) contain 4 parameters
Single_Param_Size = 4
# A factor of 2 including the xi^0 and xi^2 terms
xi2_Factor = 2
# Total number of parameters 
Tot_param_Size = NumofGPDSpecies * xi2_Factor * Flavor_Factor *  init_NumofAnsatz * Single_Param_Size

"""
The parameters will form a 5-dimensional matrix such that each para[#1,#2,#3,#4,#5] is a real number.
#1 = [0,1,2,3] corresponds to [H, E, Ht, Et]
#2 = [0,1] corresponds to [xi^0 terms, xi^2 terms]
#3 = [0,1,2,3,4] corresponds to [u - ubar, ubar, d - dbar, dbar, g]
#4 = [0,1,...,init_NumofAnsatz] corresponds to different set of parameters
#5 = [0,1,2,3] correspond to [norm, alpha, beta, alphap] as a set of parameters
"""

from Observables import GPDobserv
import numpy as np


def ParaManager(Paralst: np.array):

    """
        Here is the parameters manager, as there are over 100 free parameters. Therefore not all of them can be set free.
        Each element F_{q} is a two-dimensional matrix with init_NumofAnsatz = 1 row and Single_Param_Size = 4 columns
    """

    # Initial forward parameters for the H of (uV, ubar, dV, dbar,g) distributions
    H_uV = [[1, 0.2 , 3, 1]] 
    H_ubar = [[1, 0.8 , 8, 1]] 
    H_dV = [[1, 0.2 , 3, 1]] 
    H_dbar = [[1, 0.8 , 8, 1]] 
    H_g = [[10, 0.8 , 8, 1]] 

    # Initial forward parameters for the E of (uV, ubar, dV, dbar,g) distributions
    E_uV = [[1, 0.2 , 3, 1]]
    E_ubar = [[1, 0.8 , 8, 1]] 
    E_dV = [[1, 0.2 , 3, 1]] 
    E_dbar = [[1, 0.8 , 8, 1]] 
    E_g = [[10, 0.8 , 8, 1]] 

    # Initial forward parameters for the Ht of (uV, ubar, dV, dbar,g) distributions
    Ht_uV = [[1, 0.2 , 3, 1]] 
    Ht_ubar = [[1, 0.8 , 8, 1]] 
    Ht_dV = [[1, 0.2 , 3, 1]]
    Ht_dbar = [[1, 0.8 , 8, 1]] 
    Ht_g = [[10, 0.8 , 8, 1]] 

    # Initial forward parameters for the Et of (uV, ubar, dV, dbar,g) distributions
    Et_uV = [[1, 0.2 , 3, 1]] 
    Et_ubar = [[1, 0.8 , 8, 1]] 
    Et_dV = [[1, 0.2 , 3, 1]] 
    Et_dbar = [[1, 0.8 , 8, 1]] 
    Et_g = [[10, 0.8 , 8, 1]] 

    # Initial xi^2 parameters for the H of (uV, ubar, dV, dbar,g) distributions
    H_uV_xi2 = [[1, 0.2 , 3, 1]] 
    H_ubar_xi2 = [[1, 0.8 , 8, 1]] 
    H_dV_xi2 = [[1, 0.2 , 3, 1]] 
    H_dbar_xi2 = [[1, 0.8 , 8, 1]] 
    H_g_xi2 = [[10, 0.8 , 8, 1]] 

    # Initial xi^2 parameters for the E of (uV, ubar, dV, dbar,g) distributions
    E_uV_xi2 = [[1, 0.2 , 3, 1]] 
    E_ubar_xi2 = [[1, 0.8 , 8, 1]] 
    E_dV_xi2 = [[1, 0.2 , 3, 1]] 
    E_dbar_xi2 = [[1, 0.8 , 8, 1]] 
    E_g_xi2 = [[10, 0.8 , 8, 1]] 

    # Initial xi^2 parameters for the Ht of (uV, ubar, dV, dbar,g) distributions
    Ht_uV_xi2 = [[1, 0.2 , 3, 1]] 
    Ht_ubar_xi2 = [[1, 0.8 , 8, 1]] 
    Ht_dV_xi2 = [[1, 0.2 , 3, 1]] 
    Ht_dbar_xi2 = [[1, 0.8 , 8, 1]] 
    Ht_g_xi2 = [[10, 0.8 , 8, 1]] 

    # Initial xi^2 parameters for the Et of (uV, ubar, dV, dbar,g) distributions
    Et_uV_xi2 = [[1, 0.2 , 3, 1]] 
    Et_ubar_xi2 = [[1, 0.8 , 8, 1]] 
    Et_dV_xi2 = [[1, 0.2 , 3, 1]] 
    Et_dbar_xi2 = [[1, 0.8 , 8, 1]] 
    Et_g_xi2 = [[10, 0.8 , 8, 1]] 

    Hlst = np.array([[H_uV,     H_ubar,     H_dV,     H_dbar,     H_g],
                      [H_uV_xi2, H_ubar_xi2, H_dV_xi2, H_dbar_xi2, H_g_xi2]])
    
    Elst = np.array([[E_uV,     E_ubar,     E_dV,     E_dbar,     E_g],
                      [E_uV_xi2, E_ubar_xi2, E_dV_xi2, E_dbar_xi2, E_g_xi2]])

    Htlst = np.array([[Ht_uV,     Ht_ubar,     Ht_dV,     Ht_dbar,     Ht_g],
                       [Ht_uV_xi2, Ht_ubar_xi2, Ht_dV_xi2, Ht_dbar_xi2, Ht_g_xi2]])
    
    Etlst = np.array([[Et_uV,     Et_ubar,     Et_dV,     Et_dbar,     Et_g],
                       [Et_uV_xi2, Et_ubar_xi2, Et_dV_xi2, Et_dbar_xi2, Et_g_xi2]])

    return [Hlst, Elst, Htlst, Etlst]

