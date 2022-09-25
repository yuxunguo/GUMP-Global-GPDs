"""
Here we define the GPD ansatz based on the moment space expression norm * x^ alpha  * (1-x)^ beta.
With the GPDs ansatz, observables with LO evolution are calculated
"""
import scipy as sp
import numpy as np
from mpmath import mp, hyp2f1
from scipy.integrate import quad, quad_vec
from scipy.special import gamma
from Evolution import Moment_Evo

CFF_trans =np.array([1*(2/3)**2, 2*(2/3)**2, 1*(1/3)**2, 2*(1/3)**2, 0])

"""
***********************GPD moments***************************************
"""
#intercept for inverse Mellin transformation
inv_Mellin_intercept = 0.25

#Cutoff for inverse Mellin transformation
inv_Mellin_cutoff = 20

#Cutoff for Mellin Barnes integral
Mellin_Barnes_intercept = 0.25

#Cutoff for Mellin Barnes integral
Mellin_Barnes_cutoff = 20

#Number of effective fermions
NFEFF = 2

#Relative precision Goal of quad set to be 1e-3
Prec_Goal = 1e-3

#The flavor interpreter to return the corresponding flavor combination 
def Flv_Intp(Flv_array: np.array, flv):
    if(flv == "u"):
        return Flv_array[0]
    if(flv == "d"):
        return Flv_array[1]
    if(flv == "g"):
        return Flv_array[2]
    if(flv == "NS"):
        return Flv_array[0] - Flv_array[1]
    if(flv == "S"):
        return Flv_array[0] + Flv_array[1]
    
# Euler Beta function B(a,b) with complex arguments
def beta_loggamma(a: complex, b: complex) -> complex:
    return np.exp(sp.special.loggamma(a) + sp.special.loggamma(b)-sp.special.loggamma(a + b))

# Conformal moment in j space F(j)
def ConfMoment(j: complex, t: float, ParaSet: np.ndarray):
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

def Moment_Sum(j: complex, t: float, ParaSets: np.ndarray) -> complex:
    """
    Sum of the conformal moments when the ParaSets contain more than just one set of parameters 

    Args:
        ParaSets : contains [ParaSet1, ParaSet0, ParaSet2,...] with each ParaSet = [norm, alpha, beta ,alphap] for valence and sea distributions repsectively.        
        j: conformal spin j (or j+2 but anyway)
        t: momentum transfer squared t

    Returns:
        sum of conformal moments over all the ParaSet
    """

    return np.sum(np.array( list(map(lambda paraset: ConfMoment(j, t, paraset), ParaSets)) ))


# precision for the hypergeometric function
mp.dps = 25

def InvMellinWaveFuncQ(s: complex, x: float) -> complex:
    """ 
    Quark wave function for inverse Mellin transformation: x^(-s) for x>0 and 0 for x<0

    Args:
        s: Mellin moment s (= n)
        x: momentum fraction x

    Returns:
        Wave function for inverse Mellin transformation: x^(-s) for x>0 and 0 for x<0
    """  
    if(x > 0):
        return x ** (-s)
    
    return 0

def InvMellinWaveFuncG(s: complex, x: float) -> complex:
    """ 
    Gluon wave function for inverse Mellin transformation: x^(-s+1) for x>0 and 0 for x<0

    Args:
        s: Mellin moment s (= n)
        x: momentum fraction x

    Returns:
        Gluon wave function for inverse Mellin transformation: x^(-s+1) for x>0 and 0 for x<0
    """  
    if(x > 0):
        return x ** (-s+1)
    
    return 0


def ConfWaveFuncQ(j: complex, x: float, xi: float) -> complex:
    """ 
    Quark conformal wave function p_j(x,xi) check e.g. https://arxiv.org/pdf/hep-ph/0509204.pdf

    Args:
        j: conformal spin j (conformal spin is actually j+2 but anyway)
        x: momentum fraction x
        xi: skewness parameter xi

    Returns:
        quark conformal wave function p_j(x,xi)
    """  
    if(x > xi):
        pDGLAP = np.sin(np.pi * (j+1))/ np.pi * x**(-j-1) * complex(hyp2f1( (j+1)/2, (j+2)/2, j+5/2, (xi/x) ** 2)) 
        return pDGLAP

    if(x > -xi):
        pERBL = 2 ** (1+j) * gamma(5/2+j) / (gamma(1/2) * gamma(1+j)) * xi ** (-j-1) * (1+x/xi) * complex(hyp2f1(-1-j,j+2,2, (x+xi)/(2*xi)))
        return pERBL
    
    return 0

def ConfWaveFuncG(j: complex, x: float, xi: float) -> complex:
    """ 
    Gluon conformal wave function p_j(x,xi) check e.g. https://arxiv.org/pdf/hep-ph/0509204.pdf

    Args:
        j: conformal spin j (actually conformal spin is j+2 but anyway)
        x: momentum fraction x
        xi: skewness parameter xi

    Returns:
        gluon conformal wave function p_j(x,xi)
    """ 
    # An extra minus sign defined different from the orginal definition to absorb the extra minus sign of MB integral for gluon
    
    Minus = -1
    if(x > xi):
        pDGLAP = np.sin(np.pi * j)/ np.pi * x**(-j) * complex(hyp2f1( j/2, (j+1)/2, j+5/2, (xi/x) ** 2)) 
        return Minus * pDGLAP

    if(x > -xi):
        pERBL = 2 ** j * gamma(5/2+j) / (gamma(1/2) * gamma(j)) * xi ** (-j) * (1+x/xi) ** 2 * complex(hyp2f1(-1-j,j+2,3, (x+xi)/(2*xi)))
        return Minus * pERBL
    
    return 0

def CWilson(j: complex) -> complex:
    return 2 ** (1+j) * gamma(5/2+j) / (gamma(3/2) * gamma(3+j))


"""
***********************Observables***************************************
"""

# Class for observables
class GPDobserv (object) :
    #Initialization of observables. Each is a function of (x, xi ,t, Q), p for parity: p = 1 for vector GPDs (H, E) and p = -1 for axial-vector GPDs (Ht, Et)
    def __init__(self, init_x: float, init_xi: float, init_t: float, init_Q: float, p: int) -> None:
        self.x = init_x
        self.xi = init_xi
        self.t = init_t
        self.Q = init_Q
        self.p = p

    def tPDF(self, flv, ParaAll):
        """
        t-denpendent PDF for given flavor (flv = "u", "d", "S", "NS" or "g")
        Args:
            ParaAll = [Para_Forward, Para_xi2]
            Para_Forward = [Para_Forward_uV, Para_Forward_ubar, Para_Forward_dV, Para_Forward_dbar, Para_Forward_g]
            Para_Forward_i: parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
            Para_xi2: only matter for non-zero xi (NOT needed here but the parameters are passed for consistency with GPDs)

        Returns:
            f(x,t) in for the given flavor
        """
        
        Para_Forward = ParaAll[0]

        # The contour for inverse Meliin transform. Note that S here is the analytically continued n which is j + 1 not j !
        reS = inv_Mellin_intercept + 1
        Max_imS = inv_Mellin_cutoff 

        def InvMellinWaveConf(s: complex):

            InvMellinWaveC = np.array([[InvMellinWaveFuncQ(s, self.x), InvMellinWaveFuncQ(s, self.x) - self.p * InvMellinWaveFuncQ(s, -self.x),0,0,0],
                                       [0,0,InvMellinWaveFuncQ(s, self.x), InvMellinWaveFuncQ(s, self.x) - self.p * InvMellinWaveFuncQ(s, -self.x),0],
                                       [0,0,0,0,(InvMellinWaveFuncG(s, self.x)+ self.p * InvMellinWaveFuncG(s, -self.x))]])

            return InvMellinWaveC

        def Integrand_inv_Mellin(s: complex):
            # Calculate the unevolved moments in the orginal flavor basis
            ConfFlav = np.array( list(map(lambda paraset: Moment_Sum(s - 1, self.t, paraset), Para_Forward)) )
            # Return the evolved moments with x^(-s) for quark or x^(-s+1) for gluon for the given flavor flv = "u", "d", "S", "NS" or "g"
            return Flv_Intp(np.einsum('...j,j', InvMellinWaveConf(s), Moment_Evo(s - 1, NFEFF, self.p, self.Q, ConfFlav)), flv)

        return quad(lambda imS : np.real(Integrand_inv_Mellin(reS + 1j * imS)/(2 * np.pi)) , - Max_imS, + Max_imS, epsrel = Prec_Goal)[0]

    def CFF(self, ParaAll):
        """
        Charge averged CFF \mathcal{F}(xi, t) (\mathcal{F} = Q_u^2 F_u + Q_d^2 F_d)
        Args:
            ParaAll = [Para_Forward, Para_xi2]

            Para_Forward = [Para_Forward_uV, Para_Forward_ubar, Para_Forward_dV, Para_Forward_dbar, Para_Forward_g]
            Para_Forward_i: forward parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
            Para_xi2 = [Para_xi2_uV, Para_xi2_ubar, Para_xi2_dV, Para_xi2_dbar, Para_xi2_g]
            Para_xi2_i: xi^2 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)

        Returns:
            CFF \mathcal{F}(xi, t) = Q_u^2 F_u + Q_d^2 F_d
        """
        [Para_Forward, Para_xi2] = ParaAll

        # The contour for Mellin-Barnes integral in terms of j not n.
        reJ = Mellin_Barnes_intercept 
        Max_imJ = Mellin_Barnes_cutoff 

        def Integrand_Mellin_Barnes_CFF(j: complex):
            ConfFlav = np.array( list(map(lambda paraset: Moment_Sum(j, self.t, paraset), Para_Forward)) )
            ConfFlav_xi2 = np.array( list(map(lambda paraset: Moment_Sum(j+2, self.t, paraset), Para_xi2)) )

            EvoConf_Wilson = (CWilson(j) * Moment_Evo(j, NFEFF, self.p, self.Q, ConfFlav) + CWilson(j+2) * Moment_Evo(j+2, NFEFF, self.p, self.Q, ConfFlav_xi2))

            return np.einsum('j,j', CFF_trans, EvoConf_Wilson)

        if (self.p == 1):
            return quad_vec(lambda imJ : self.xi ** (-reJ - 1j * imJ - 1) * (1j + np.tan((reJ + 1j * imJ) * np.pi / 2)) *Integrand_Mellin_Barnes_CFF(reJ + 1j * imJ) / 2, - Max_imJ, + Max_imJ, epsrel = Prec_Goal)[0]
        
        if (self.p == -1):
            return quad_vec(lambda imJ : self.xi ** (-reJ - 1j * imJ - 1) * (1j - 1/np.tan((reJ + 1j * imJ) * np.pi / 2)) *Integrand_Mellin_Barnes_CFF(reJ + 1j * imJ) / 2, - Max_imJ, + Max_imJ, epsrel = Prec_Goal)[0]

    def GPD(self, flv, ParaAll):
        """
        GPD F(x, xi, t) in flavor space (uV, ubar, dV, dbar, gluon)
        Args:
            ParaAll = [Para_Forward, Para_xi2]

            Para_Forward = [Para_Forward_uV, Para_Forward_ubar, Para_Forward_dV, Para_Forward_dbar, Para_Forward_g]
            Para_Forward_i: forward parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
            Para_xi2 = [Para_xi2_uV, Para_xi2_ubar, Para_xi2_dV, Para_xi2_dbar, Para_xi2_g]
            Para_xi2_i: xi^2 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)

        Returns:
            f(x,xi,t) in (uV, ubar, dV, dbar, gluon) space
        """
        [Para_Forward, Para_xi2] = ParaAll

        # The contour for Mellin-Barnes integral in terms of j not n.        
        reJ = Mellin_Barnes_intercept 
        Max_imJ = Mellin_Barnes_cutoff 
        def ConfWaveConv(j: complex):

            ConfWaveC = np.array([[ConfWaveFuncQ(j, self.x, self.xi), ConfWaveFuncQ(j, self.x, self.xi) - self.p * ConfWaveFuncQ(j, -self.x, self.xi),0,0,0],
                                  [0,0,ConfWaveFuncQ(j, self.x, self.xi), ConfWaveFuncQ(j, self.x, self.xi) - self.p * ConfWaveFuncQ(j, -self.x, self.xi),0],
                                  [0,0,0,0,ConfWaveFuncG(j, self.x, self.xi)+ self.p * ConfWaveFuncG(j, -self.x, self.xi)]])

            return ConfWaveC
    
        def Integrand_Mellin_Barnes(j: complex):

            ConfFlav = np.array( list(map(lambda paraset: Moment_Sum(j, self.t, paraset), Para_Forward)) )
            ConfFlav_xi2 = np.array( list(map(lambda paraset: Moment_Sum(j+2, self.t, paraset), Para_xi2)) )
             
            return Flv_Intp(np.einsum('...j,j', ConfWaveConv(j), Moment_Evo(j, NFEFF, self.p, self.Q, ConfFlav))  +  self.xi ** 2 * np.einsum('...j,j', ConfWaveConv(j+2), Moment_Evo(j+2, NFEFF, self.p, self.Q, ConfFlav_xi2)), flv)
        
        # Adding a j = 0 term because the contour do not enclose the j = 0 pole which should be the 0th conformal moment.
        # We cannot change the Mellin_Barnes_intercept > 0 to enclose the j = 0 pole only, due to the pomeron pole around j = 0.
        def GPD0():
            if(self.p == -1):
                return Integrand_Mellin_Barnes(0)

            if(self.p == 1):
                
                ConfFlav = np.array( list(map(lambda paraset: Moment_Sum(0, self.t, paraset), Para_Forward)) )
                ConfFlav_xi2 = np.array( list(map(lambda paraset: Moment_Sum(2, self.t, paraset), Para_xi2)) )

                return Flv_Intp(np.einsum('...j,j', ConfWaveConv(0),ConfFlav) +  self.xi ** 2 * np.einsum('...j,j', ConfWaveConv(2), Moment_Evo(2, NFEFF, self.p, self.Q, ConfFlav_xi2)), flv)

        return np.real(GPD0()) + quad(lambda imJ : np.real(Integrand_Mellin_Barnes(reJ + 1j* imJ) / (2 * np.sin((reJ + 1j * imJ+1) * np.pi)) ), - Max_imJ, + Max_imJ, epsrel = Prec_Goal)[0]

    def GFFj0(self, j: int, flv, ParaAll):
        """
            Generalized Form Factors A_{j0}(t) which is the xi^0 term of the nth (n= j+1) Mellin moment of GPD int dx x^j F(x,xi,t) for quark and int dx x^(j-1) F(x,xi,t) for gluon
            Note for gluon, GPD reduce to x*g(x), not g(x) so the Mellin moment will have a mismatch
        """
        Para_Forward = ParaAll[0]
        
        GFF_trans = np.array([[1,1 - self.p * (-1) ** j,0,0,0],
                              [0,0,1,1 - self.p * (-1) ** j,0],
                              [0,0,0,0,(1 - self.p * (-1) ** j)/2]])

        ConfFlav = np.array( list(map(lambda paraset: Moment_Sum(j, self.t, paraset), Para_Forward)) )

        if (j == 0):
            if(self.p == 1):
                return Flv_Intp( np.array([ConfFlav[0],ConfFlav[2],ConfFlav[4]]) , flv)
        
        return Flv_Intp(np.einsum('...j,j', GFF_trans, Moment_Evo(j, NFEFF, self.p, self.Q, ConfFlav)), flv)