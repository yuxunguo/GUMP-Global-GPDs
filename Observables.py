"""
Here we define the GPD ansatz based on the moment space expression norm * x^ alpha  * (1-x)^ beta.
With the GPDs ansatz, observables with LO evolution are calculated
"""

import scipy as sp
import numpy as np
from scipy.integrate import quad_vec
from Evolution import evolop
"""
***********************GPD moments***************************************
"""
#intercept for inverse Mellin transformation
inv_Mellin_intercept = 0.35

#Cutoff for inverse Mellin transformation
inv_Mellin_cutoff = 50

#Number of effective fermions
NFEFF = 5

# Euler Beta function B(a,b) with complex arguments
def beta_loggamma(a: complex, b: complex) -> complex:
    return np.exp(sp.special.loggamma(a) + sp.special.loggamma(b)-sp.special.loggamma(a + b))

# Conformal moment in j space F(j)
def ConfMoment(ParaSet: np.ndarray, j: complex, k: int, t: float) -> complex:
    """
    Conformal moment in j space F(j)

    Args:
        ParaSet is the array of parameters in the form of (norm, alpha, beta, alphap)
            norm = ParaSet[0]: overall normalization constant
            alpha = ParaSet[1], beta = ParaSet[2]: the two parameters corresponding to x ^ (-alpha) * (1 - x) ^ beta
            alphap = ParaSet[3]: regge trajectory alpha(t) = alpha + alphap * t
        j: conformal moment j
        k: xi powers xi ^ (2 * k)
        t: momentum transfer squared t

    Returns:
        Conformal moment in j space F(j,k,t)
    """
    norm = ParaSet[0]
    alpha = ParaSet[1]
    beta = ParaSet[2]
    alphap = ParaSet[3]
    return norm * beta_loggamma (j + 1 - alpha, 1 + beta) * (j + 1 - k - alpha)/ (j + 1 - k - alpha - alphap * t)

def Moment_Sum(ParaSets: np.ndarray, j: complex, k: int, t: float) -> complex:
    return sum(list(map(lambda paraset: ConfMoment(paraset, j, k, t), ParaSets)))

"""
***********************Observables***************************************
"""
# Number of ansatz,  combinations of multiple ansatz x^alpha (1-x)^beta might be needed for more flexiblity but 1 should be good to start with
init_NumofAnsatz = 1

# Number of xi^2 terms, the max power of xi will be xi^(2 Max_k)
Max_k = 1

# Initial parameters for the non-singlet valence and sea quark distributions
Init_Para_NS_Valence = [[1, 0.2 , 3, 1]] * init_NumofAnsatz
Init_Para_NS_Sea = [[1, 0.8 , 8, 1]] * init_NumofAnsatz

# Initial parameters for the singlet quark distributions
Init_Para_S_Valence = [[1, 0.2 , 3, 1]] * init_NumofAnsatz
Init_Para_S_Sea = [[1, 0.8 , 8, 1]] * init_NumofAnsatz

# Initial parameters for the gluon distributions (only sea distributions)
Init_Para_G = [[1, 0.8 , 8, 1]] * init_NumofAnsatz

# Class for observables
class GPDobserv (object) :
    #Initialization of observables. Each is a function of (x, xi ,t, Q) 
    def __init__(self, init_x: float, init_xi: float, init_t: float, init_Q: float) -> None:
        self.x = init_x
        self.xi = init_xi
        self.t = init_t
        self.Q = init_Q

    def tPDF(self, Para_NSV, Para_NSS, Para_SV, Para_SS, Para_G):
        """
        t-denpendent PDF in flavor space (non-singlet, singlet, gluon)
        Args:
            Para_NSV: parameter set for non-singlet valence quark
            Para_NSS: parameter set for non-singlet sea quark
            Para_SV: parameter set for singlet valence quark
            Para_SS: parameter set for singlet sea quark
            Para_G: parameter set for gluon

        Returns:
            f(x,t) in (non-singlet, singlet, gluon) space
        """
        def Integrand_Inv_Mellin(j):
            """
                define the integrand function for the inverse Mellin transform
            """
            ConfNS = Moment_Sum(Para_NSV, j, 0, self.t) + Moment_Sum(Para_NSS, j, 0, self.t)
            ConfS = Moment_Sum(Para_SV, j, 0, self.t) + Moment_Sum(Para_SS, j, 0, self.t)
            ConfG = Moment_Sum(Para_G, j, 0, self.t)
            Evo = evolop(j, NFEFF, self.Q)
            return np.einsum('...i,i->...', Evo, [ConfNS, ConfS, ConfG])
        
        # The contour for inverse Meliin transform. Note that S here is the analytically continued n which is j + 1 not j !
        reS = inv_Mellin_intercept + 1
        Max_ImS = inv_Mellin_cutoff 
        return quad_vec(lambda imS : ( np.real(self.x ** (- reS - 1j * imS) * Integrand_Inv_Mellin(reS - 1 + 1j * imS)) / (2 * np.pi) ), - Max_ImS , + Max_ImS )[0]




TestGPD = GPDobserv(0.1, 0, 0, 5)

import time
start_time = time.time()
print(TestGPD.tPDF(Init_Para_NS_Valence,Init_Para_NS_Sea, Init_Para_S_Valence, Init_Para_S_Sea, Init_Para_G))
print("--- %s seconds ---" % (time.time() - start_time))



