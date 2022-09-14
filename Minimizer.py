"""
The minimizer using iMinuit, which takes 1-D array for the input parameters only.

Extra efforts needed to convert the form of the parameters.

"""
# Number of GPD species, 4 leading-twist GPDs including H, E Ht, Et are needed.
NumofGPDSpecies = 4
# Number of Fermion flavor
NF = 2
# Number of flavor factor, Flavor_Factor = 2 * nf + 1 needed including 2 * nf quark (antiquark) and one gluon
Flavor_Factor = 2 * NF + 1
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
import sys

def ParaConvert(Allparameter: np.array):
    if(len(Allparameter) != Tot_param_Size):
        sys.exit("Number of parameters do not match!") 
    nDParam = np.reshape(Allparameter, (NumofGPDSpecies, xi2_Factor, Flavor_Factor, init_NumofAnsatz, Single_Param_Size))
    return nDParam

# Initial parameters for the non-singlet valence and sea quark distributions
Init_Para_uV = [[1, 0.2 , 3, 1]] * init_NumofAnsatz
Init_Para_ubar = [[1, 0.8 , 8, 1]] * init_NumofAnsatz


# Initial parameters for the singlet quark distributions
Init_Para_dV = [[1, 0.2 , 3, 1]] * init_NumofAnsatz
Init_Para_dbar = [[1, 0.8 , 8, 1]] * init_NumofAnsatz

# Initial parameters for the gluon distributions (only sea distributions)
Init_Para_g = [[10, 0.8 , 8, 1]] * init_NumofAnsatz

# Initial parameters for the non-singlet valence and sea quark distributions for the xi^2 terms
Init_Para_uV_xi2 = [[1, 0.2 , 3, 1]] * init_NumofAnsatz
Init_Para_ubar_xi2 = [[1, 0.8 , 8, 1]] * init_NumofAnsatz

# Initial parameters for the singlet quark distributions for the xi^2 terms
Init_Para_dV_xi2 = [[1, 0.2 , 3, 1]] * init_NumofAnsatz
Init_Para_dbar_xi2 = [[1, 0.8 , 8, 1]] * init_NumofAnsatz

# Initial parameters for the gluon distributions for the xi^2 terms
Init_Para_g_xi2 = [[10, 0.8 , 8, 1]] * init_NumofAnsatz

TestGPD = GPDobserv(0.1, 0.1, 0, 5, 1)

import time

Init_Para_forward = [Init_Para_uV, Init_Para_ubar, Init_Para_dV, Init_Para_dbar,Init_Para_g ]

Init_Para_xi2 = [Init_Para_uV_xi2, Init_Para_ubar_xi2, Init_Para_dV_xi2, Init_Para_dbar_xi2,Init_Para_g_xi2 ]

Init_Para_All = [Init_Para_forward, Init_Para_xi2]

start = time.time()
print(TestGPD.CFF(Init_Para_All))
end = time.time()
print(end - start)

start = time.time()
print(TestGPD.GPD(Init_Para_All) * np.pi)
end = time.time()
print(end - start)

start = time.time()
print(TestGPD.tPDF(Init_Para_All) * np.pi)
end = time.time()
print(end - start)


