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
    [norm, alpha, beta, alphap] = ParaSet
    return norm * beta_loggamma (j + 1 - alpha, 1 + beta) * (j + 1  - alpha)/ (j + 1 - alpha - alphap * t)

def Moment_Sum(ParaSets: np.ndarray, j: complex, t: float) -> complex:
    """
    Sum of the conformal moments when the ParaSets contain more than just one set of parameters 

    Args:
        ParaSets = [ParaSea, ParaValence]        
        ParaValence/ParaSea : contains [ParaSet1, ParaSet0, ParaSet2,...] with each ParaSet = [norm, alpha, beta ,alphap] for valence and sea distributions repsectively.        
        j: conformal spin j (conformal spin is actually j+2 but anyway)
        t: momentum transfer squared t

    Returns:
        sum of conformal moments over all the ParaSet
    """
    
    [ParaSea, ParaValence] = ParaSets
    return sum(list(map(lambda paraset: ConfMoment(paraset, j, t), ParaValence))) + sum(list(map(lambda paraset: ConfMoment(paraset, j, t),ParaSea)))

def Moment_Evo(j: complex, t: float, nf: int, Q: float, ParasSet_Flavor):
    """
    Evolution of moments in the flavor space (non-singlet, singlet, gluon)

    Args:
        ParasSet_Flavor = [ParasSet_NS, ParasSet_S, ParasSet_G] 
        ParaSets_NS: parameter sets for non-singlet quark (both sea and valence)
        ParaSets_S: parameter sets for singlet quark (both sea and valence)
        ParaSets_G: parameter sets for gluon
        j: conformal spin j (conformal spin is actually j+2 but anyway)
        t: momentum transfer squared
        nf: number of effective fermions
        Q: final evolution scale
        
    Returns:
        Evolved conformal moments in flavor space (non-singlet, singlet, gluon)
    """
    [ParaSets_NS, ParaSets_S, ParaSets_G] = ParasSet_Flavor 

    ConfNS = Moment_Sum(ParaSets_NS, j, t)
    ConfS = Moment_Sum(ParaSets_S, j, t)
    ConfG = Moment_Sum(ParaSets_G, j, t)
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

# Class for observables
class GPDobserv (object) :
    #Initialization of observables. Each is a function of (x, xi ,t, Q) 
    def __init__(self, init_x: float, init_xi: float, init_t: float, init_Q: float) -> None:
        self.x = init_x
        self.xi = init_xi
        self.t = init_t
        self.Q = init_Q

    def tPDF(self, ParaAll):
        """
        t-denpendent PDF in flavor space (non-singlet, singlet, gluon)
        Args:
            ParaAll = [Para_All_Forward, Para_All_xi2]
            Para_All_Forward = [Para_NS, Para_S, Para_G]
            Para_NS: parameter set for non-singlet quark (both sea and valence) in the forward limit 
            Para_S: parameter set for singlet quark (both sea and valence) in the forward limit
            Para_G: parameter set for gluon in the forward limit
            Para_All_xi2: only matter for non-zero xi (NOT needed here but the parameters are passed for consistency with GPDs)

        Returns:
            f(x,t) in (non-singlet, singlet, gluon) space
        """

        # The contour for inverse Meliin transform. Note that S here is the analytically continued n which is j + 1 not j !
        reS = inv_Mellin_intercept + 1
        Max_imS = inv_Mellin_cutoff 

        def Integrand_inv_Mellin(s: complex):
            return self.x ** (-s) * Moment_Evo(s - 1, self.t, NFEFF, self.Q, ParaAll[0])/(2 * np.pi)

        return quad_vec(lambda imS : np.real(Integrand_inv_Mellin(reS + 1j * imS)) , - Max_imS, + Max_imS)[0]

    def GPD(self, ParaAll):
        """
        GPD F(x, xi, t) in flavor space (non-singlet, singlet, gluon)
        Args:
            ParaAll = [Para_All_Forward, Para_All_xi2]
            Para_All_Forward = [Para_NS, Para_S, Para_G]
            Para_NS: parameter set for non-singlet quark (both sea and valence) in the forward limit 
            Para_S: parameter set for singlet quark (both sea and valence) in the forward limit
            Para_G: parameter set for gluon in the forward limit
            Para_All_xi2 = [Para_NS_2, Para_S_2, Para_G_2]
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
            return ConfWaveFunc(j, self.x, self.xi) * Moment_Evo(j, self.t, NFEFF, self.Q, ParaAll[0]) + self.xi ** 2 * ConfWaveFunc(j + 2, self.x, self.xi) * Moment_Evo(j, self.t, NFEFF, self.Q, ParaAll[1])

        return quad_vec(lambda imJ : np.real(Integrand_Mellin_Barnes(reJ + 1j* imJ) / (2 * np.sin((reJ + 1j * imJ+1) * np.pi)) ), - Max_imJ, + Max_imJ)[0]

    def CFF(self, ParaAll):
        """
        CFF \mathcal{F}(xi, t) in flavor space (non-singlet, singlet, gluon)
        Args:
            ParaAll = [Para_All_Forward, Para_All_xi2]
            Para_All_Forward = [Para_NS, Para_S, Para_G]
            Para_NS: parameter set for non-singlet quark (both sea and valence) in the forward limit 
            Para_S: parameter set for singlet quark (both sea and valence) in the forward limit
            Para_G: parameter set for gluon in the forward limit
            Para_All_xi2 = [Para_NS_2, Para_S_2, Para_G_2]
            Para_NS_2: parameter set for non-singlet quark (both sea and valence) for the xi^2 terms
            Para_S_2: parameter set for singlet quark (both sea and valence) for the xi^2 terms
            Para_G_2: parameter set for gluon for the xi^2 terms

        Returns:
            CFF \mathcal{F}(xi, t) in (non-singlet, singlet, gluon) space
        """

        # The contour for Mellin-Barnes integral in terms of j not n.
        reJ = Mellin_Barnes_intercept 
        Max_imJ = Mellin_Barnes_cutoff 

        def Integrand_Mellin_Barnes_CFF(j: complex):
            return CWilson(j) * Moment_Evo(j, self.t, NFEFF, self.Q, ParaAll[0]) + CWilson(j+2) * Moment_Evo(j, self.t, NFEFF, self.Q,  ParaAll[1])

        return quad_vec(lambda imJ : self.xi ** (-reJ - 1j * imJ - 1) * (1j + np.tan((reJ + 1j * imJ) * np.pi / 2)) *Integrand_Mellin_Barnes_CFF(reJ + 1j * imJ) / 2, - Max_imJ, + Max_imJ)[0]