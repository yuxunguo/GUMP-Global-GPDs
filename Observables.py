"""
Here we define the GPD ansatz based on the moment space expression norm * x^ alpha  * (1-x)^ beta.
With the GPDs ansatz, observables with LO evolution are calculated
"""

import scipy as sp
import numpy as np
from scipy.integrate import quad
"""
***********************GPD moments***************************************
"""
#intercept for inverse Mellin transformation
inv_Mellin_intercept = 0.35
#Cutoff for inverse Mellin transformation
inv_Mellin_cutoff = 10

# Euler beta function B(a,b) with complex arguments
def beta_loggamma(a, b):
    return np.exp(sp.special.loggamma(a) + sp.special.loggamma(b)-sp.special.loggamma(a + b))

# Conformal moment in j space F(j)
def ConfMoment(norm, alpha, beta, alphap, j, k, t):
    return norm * beta_loggamma (j + 1 - alpha, 1 + beta) * (j + 1 - k - alpha)/ (j + 1 - k - alpha - alphap * t)

# Complex contour integral with scipy quad function
def complex_quad(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    return real_integral[0]

"""
***********************Observables***************************************
"""
# Number of ansatz,  combinations of multiple ansatz x^alpha (1-x)^beta might be needed for more flexiblity but 1 should be good to start with
init_NumofAnsatz = 1

# Number of xi^2 terms, the max power of xi will be xi^(2 Max_k)
Max_k = 2

# Initial parameters for the non-singlet valence and sea quark distributions
Para_NS_Valence = [[1, 0.2 , 3, 1]] * init_NumofAnsatz
Para_NS_Sea = [[1, 0.8 , 8, 1]] * init_NumofAnsatz

# Initial parameters for the singlet quark distributions
Para_S_Valence = [[1, 0.2 , 3, 1]] * init_NumofAnsatz
Para_S_Sea = [[1, 0.8 , 8, 1]] * init_NumofAnsatz

# Initial parameters for the gluon distributions (only sea distributions)
Para_G = [[1, 0.8 , 8, 1]] * init_NumofAnsatz

# Class for observables
class GPDobserv (object) :
    #Initialization of observables. Each is a function of (x, xi ,t, Q) 
    def __init__(self, init_x , init_xi, init_t, init_Q) -> None:
        self.x = init_x
        self.xi = init_xi
        self.t = init_t
        self.Q = init_Q

    def tPDF(self, Para_NSV, Para_NSS, Para_SV, Para_SS, Para_G):
        return 

