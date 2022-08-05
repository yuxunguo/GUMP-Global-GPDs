"""
Here we define the GPD ansatz based on the moment space expression norm * x^ alpha  * (1-x)^ beta.
With the GPDs ansatz, observables with LO evolution are calculated
"""

from platform import python_branch
import scipy as sp
import numpy as np
from mpmath import mp, hyp2f1
from scipy.integrate import quad_vec
from scipy.special import gamma
from Evolution import evolop

"""
***********************GPD moments***************************************
"""
#intercept for inverse Mellin transformation
inv_Mellin_intercept = 0.35

#Cutoff for inverse Mellin transformation
inv_Mellin_cutoff = 50

#Cutoff for Mellin Barnes integral
Mellin_Barnes_intercept = 0.35

#Cutoff for Mellin Barnes integral
Mellin_Barnes_cutoff = 20

#Number of effective fermions
NFEFF = 5

# Euler Beta function B(a,b) with complex arguments
def beta_loggamma(a: complex, b: complex) -> complex:
    return np.exp(sp.special.loggamma(a) + sp.special.loggamma(b)-sp.special.loggamma(a + b))

# Conformal moment in j space F(j)
def ConfMoment(ParaSet: np.ndarray, j: complex, t: float) -> complex:
    """
    Conformal moment in j space F(j)

    Args:
        ParaSet is the array of parameters in the form of (norm, alpha, beta, alphap)
            norm = ParaSet[0]: overall normalization constant
            alpha = ParaSet[1], beta = ParaSet[2]: the two parameters corresponding to x ^ (-alpha) * (1 - x) ^ beta
            alphap = ParaSet[3]: regge trajectory alpha(t) = alpha + alphap * t
        j: conformal spin j (conformal spin is actually j+2 but anyway)
        t: momentum transfer squared t

    Returns:
        Conformal moment in j space F(j,t)
    """
    norm = ParaSet[0]
    alpha = ParaSet[1]
    beta = ParaSet[2]
    alphap = ParaSet[3]
    return norm * beta_loggamma (j + 1 - alpha, 1 + beta) * (j + 1  - alpha)/ (j + 1 - alpha - alphap * t)

def Moment_Sum(ParaSets: np.ndarray, j: complex, t: float) -> complex:
    """
    Sum of the conformal moments when the ParaSets contain more than just one set of parameters 

    Args:
        ParaSets: contains [ParaSet1, ParaSet0, ParaSet2,...] with each ParaSet = [norm, alpha, beta ,alphap]
        j: conformal spin j (conformal spin is actually j+2 but anyway)
        t: momentum transfer squared t

    Returns:
        sum of conformal moments over all the ParaSet
    """

    return sum(list(map(lambda paraset: ConfMoment(paraset, j, t), ParaSets)))

def Moment_Evo(j: complex, t: float, nf: int, Q: float, ParasSet_NS, ParaSet_S, ParaSet_G):
    """
    Evolution of moments in the flavor space (non-singlet, singlet, gluon)

    Args:
        j: conformal spin j (conformal spin is actually j+2 but anyway)
        t: momentum transfer squared
        nf: number of effective fermions
        Q: final evolution scale
        ParasSet_NS: parameter sets for non-singlet quark (both sea and valence)
        ParasSet_S: parameter sets for singlet quark (both sea and valence)
        ParasSet_G: parameter sets for gluon

    Returns:
        Evolved conformal moments in flavor space (non-singlet, singlet, gluon)
    """
    ConfNS = Moment_Sum(ParasSet_NS, j, t)
    ConfS = Moment_Sum(ParaSet_S, j, t)
    ConfG = Moment_Sum(ParaSet_G, j, t)
    Evo = evolop(j, nf, Q)
    return np.einsum('...i,i->...', Evo, [ConfNS, ConfS, ConfG])
# precision for the hypergeometric function
mp.dps = 25

def ConfWaveFunc(j: complex, x: float, xi: float) -> complex:
    """ 
    conformal wave function p_j(x,xi)

    Args:
        j: conformal spin j (conformal spin is actually j+2 but anyway)
        x: momentum fraction x
        xi: skewness parameter xi

    Returns:
        conformal wave function p_j(x,xi)
    """  
    if(x > xi):
        pDGLAP = np.sin(np.pi * (j+1))/ np.pi * x**(-j-1) * complex(hyp2f1( (j+1)/2, (j+2)/2, j+5/2, (xi/x) ** 2)) 
        return pDGLAP

    if(x > -xi):
        pERBL = 2 ** (1+j) * gamma(5/2+j) / (gamma(1/2) * gamma(1+j)) * xi ** (-j-1) * (1+x/xi) * complex(hyp2f1(-1-j,j+2,2, (x+xi)/(2*xi)))
        return pERBL
    
    return 0

def CWilson(j: complex) -> complex:
    return 2 ** (1+j) * gamma(5/2+j) / (gamma(3/2) * gamma(3+j))


"""
***********************Observables***************************************
"""
# Number of ansatz,  combinations of multiple ansatz x^alpha (1-x)^beta might be needed for more flexiblity but 1 should be good to start with
init_NumofAnsatz = 1

# Initial parameters for the non-singlet valence and sea quark distributions
Init_Para_NS_Valence = [[1, 0.2 , 3, 1]] * init_NumofAnsatz
Init_Para_NS_Sea = [[1, 0.8 , 8, 1]] * init_NumofAnsatz

# Initial parameters for the non-singlet valence and sea quark distributions for the xi^2 terms
Init_Para_NS_Valence_xi_2 = [[1, 0.2 , 3, 1]] * init_NumofAnsatz
Init_Para_NS_Sea_xi_2 = [[1, 0.8 , 8, 1]] * init_NumofAnsatz

# Initial parameters for the singlet quark distributions
Init_Para_S_Valence = [[1, 0.2 , 3, 1]] * init_NumofAnsatz
Init_Para_S_Sea = [[1, 0.8 , 8, 1]] * init_NumofAnsatz

# Initial parameters for the singlet quark distributions for the xi^2 terms
Init_Para_S_Valence_xi_2 = [[1, 0.2 , 3, 1]] * init_NumofAnsatz
Init_Para_S_Sea_xi_2 = [[1, 0.8 , 8, 1]] * init_NumofAnsatz

# Initial parameters for the gluon distributions (only sea distributions)
Init_Para_G = [[1, 0.8 , 8, 1]] * init_NumofAnsatz

# Initial parameters for the gluon distributions for the xi^2 terms
Init_Para_G_xi_2 = [[1, 0.8 , 8, 1]] * init_NumofAnsatz

# Class for observables
class GPDobserv (object) :
    #Initialization of observables. Each is a function of (x, xi ,t, Q) 
    def __init__(self, init_x: float, init_xi: float, init_t: float, init_Q: float) -> None:
        self.x = init_x
        self.xi = init_xi
        self.t = init_t
        self.Q = init_Q

    def tPDF(self, Para_NS, Para_S, Para_G):
        """
        t-denpendent PDF in flavor space (non-singlet, singlet, gluon)
        Args:
            Para_NS: parameter set for non-singlet quark (both sea and valence)
            Para_S: parameter set for singlet quark (both sea and valence)
            Para_G: parameter set for gluon

        Returns:
            f(x,t) in (non-singlet, singlet, gluon) space
        """

        # The contour for inverse Meliin transform. Note that S here is the analytically continued n which is j + 1 not j !
        reS = inv_Mellin_intercept + 1
        Max_imS = inv_Mellin_cutoff 

        def Integrand_inv_Mellin(s: complex):
            return self.x ** (-s) * Moment_Evo(s - 1, self.t, NFEFF, self.Q, Para_NS, Para_S, Para_G)/(2 * np.pi)

        return quad_vec(lambda imS : np.real(Integrand_inv_Mellin(reS + 1j * imS)) , - Max_imS, + Max_imS)[0]

    def GPD(self, Para_NS, Para_S, Para_G, Para_NS_xi_2, Para_S_xi_2, Para_G_xi_2):
        """
        GPD F(x, xi, t) in flavor space (non-singlet, singlet, gluon)
        Args:
            Para_NS: parameter set for non-singlet quark (both sea and valence)
            Para_S: parameter set for singlet quark (both sea and valence)
            Para_G: parameter set for gluon
            Para_NS_2: parameter set for non-singlet quark (both sea and valence) for the xi^2 terms
            Para_S_2: parameter set for singlet quark (both sea and valence) for the xi^2 terms
            Para_G_2: parameter set for gluon for the xi^2 terms

        Returns:
            f(x,xi,t) in (non-singlet, singlet, gluon) space
        """
        # The contour for Mellin-Barnes integral in terms of j not n.
        reJ = Mellin_Barnes_intercept 
        Max_imJ = Mellin_Barnes_cutoff 

        def Integrand_Mellin_Barnes(j: complex):
            return ConfWaveFunc(j, self.x, self.xi) * Moment_Evo(j, self.t, NFEFF, self.Q, Para_NS, Para_S, Para_G) + self.xi ** 2 * ConfWaveFunc(j + 2, self.x, self.xi) * Moment_Evo(j, self.t, NFEFF, self.Q, Para_NS_xi_2, Para_S_xi_2, Para_G_xi_2)

        return quad_vec(lambda imJ : np.real(Integrand_Mellin_Barnes(reJ + 1j* imJ) / (2 * np.sin((reJ + 1j * imJ+1) * np.pi)) ), - Max_imJ, + Max_imJ)[0]

    def CFF(self, Para_NS, Para_S, Para_G, Para_NS_xi_2, Para_S_xi_2, Para_G_xi_2):
        """
        CFF \mathcal{F}(xi, t) in flavor space (non-singlet, singlet, gluon)
        Args:
            Para_NS: parameter set for non-singlet quark (both sea and valence)
            Para_S: parameter set for singlet quark (both sea and valence)
            Para_G: parameter set for gluon

        Returns:
            CFF \mathcal{F}(xi, t) in (non-singlet, singlet, gluon) space
        """
        # The contour for Mellin-Barnes integral in terms of j not n.
        reJ = Mellin_Barnes_intercept 
        Max_imJ = Mellin_Barnes_cutoff 

        def Integrand_Mellin_Barnes_CFF(j: complex):
            return CWilson(j) * Moment_Evo(j, self.t, NFEFF, self.Q, Para_NS, Para_S, Para_G) + CWilson(j+2) * Moment_Evo(j, self.t, NFEFF, self.Q, Para_NS_xi_2, Para_S_xi_2, Para_G_xi_2)

        return quad_vec(lambda imJ : self.xi ** (-reJ - 1j * imJ - 1) * (1j + np.tan((reJ + 1j * imJ) * np.pi / 2)) *Integrand_Mellin_Barnes_CFF(reJ + 1j * imJ) / 2, - Max_imJ, + Max_imJ)[0]

TestGPD = GPDobserv(0.1, 0.1, 0, 2.5)
import time
Init_Para_NS = np.vstack([Init_Para_NS_Valence, Init_Para_NS_Sea])
Init_Para_S =  np.vstack([Init_Para_S_Valence, Init_Para_S_Sea])
Init_Para_NS_xi_2 = np.vstack([Init_Para_NS_Valence_xi_2, Init_Para_NS_Sea_xi_2])
Init_Para_S_xi_2 =  np.vstack([Init_Para_S_Valence_xi_2, Init_Para_S_Sea_xi_2])

start_time = time.time()
print(TestGPD.tPDF(Init_Para_NS, Init_Para_S, Init_Para_G))
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
print(TestGPD.GPD(Init_Para_NS, Init_Para_S, Init_Para_G, Init_Para_NS_xi_2, Init_Para_S_xi_2, Init_Para_G_xi_2)*np.pi)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
print(TestGPD.CFF(Init_Para_NS, Init_Para_S, Init_Para_G, Init_Para_NS_xi_2, Init_Para_S_xi_2, Init_Para_G_xi_2))
print("--- %s seconds ---" % (time.time() - start_time))