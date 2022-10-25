"""
LO QCD evolution of moment space GPD. Credits to K. Kumericki at https://github.com/kkumer/gepard.

We used RunDec for the running strong coupling constant alphaS instead and made slight modifications.

Note:
    Functions in this module have as first argument Mellin moment
    n = j + 1, where j is conformal moment used everywhere else.
    Thus, they should always be called as f(j+1, ...).

"""
# from cmath import exp
# from scipy.special import loggamma as clngamma
#from this import d

import numpy as np
import rundec
from scipy.special import psi
from typing import Tuple
import numba
from numba import vectorize, njit

"""
***********************QCD constants***************************************
Refer to the constants.py at https://github.com/kkumer/gepard.
"""

NC = 3
CF = (NC**2 - 1) / (2 * NC)
CA = NC
CG = CF - CA/2
TF = 0.5
Alpha_Mz = 0.1181
# All unit in GeV for dimensional quantities.
Mz = 91.1876
# Two loop accuracy for running strong coupling constant.
nloop_alphaS = 2
# Initial scale of distribution functions at 2 GeV.
Init_Scale_Q = 2

# Transform the original flavor basis to the evolution basis (qVal, q_du_plus, q_du_minus, qSigma, g)
# The same basis for PDF evolution are used, check references.

flav_trans =np.array([[1, 0, 1, 0, 0],
                     [-1, -2, 1, 2, 0],
                     [-1, 0, 1, 0 , 0],
                     [1, 2, 1, 2, 0],
                     [0, 0, 0, 0, 1]])

inv_flav_trans = np.linalg.inv(flav_trans)

"""
***********************pQCD running coupling constant***********************
Here rundec is used instead.
"""

B00 = 11./3. * CA
B01 = -4./3. * TF
B10 = 34./3. * CA**2
B11 = -20./3. * CA*TF - 4. * CF*TF

@njit(["float64(int32)"])
def beta0(nf: int) -> float:
    """ LO beta function of pQCD, will be used for LO GPD evolution. """
    return - B00 - B01 * nf

@njit(["float64(int32)"])
def beta1(nf):
    """ NLO beta function of pQCD """
    return - B10 - B11 * nf

@njit(["float64[:](float64[:], int32)", "float64(float64, int32)"])
def _fbeta1(a: float, nf: int) -> float:
    return a**2 * (beta0(nf) + a * beta1(nf))

@njit(["float64[:](int32, float64[:])", "float64(int32, float64)"])
def AlphaS0(nf: int, Q: float) -> float:
    return Alpha_Mz / (1 - Alpha_Mz/2/np.pi * beta0(nf) * np.log(Q/Mz))

@njit(["float64[:](int32, float64[:])", "float64(int32, float64)"])
def AlphaS1(nf: int, Q: float) -> float:
    NASTPS = 20
    
    # a below is as defined in 1/4pi expansion
    a = np.ones_like(Q) * Alpha_Mz / 4 / np.pi
    lrrat = 2 * np.log(Q/Mz)
    dlr = lrrat / NASTPS

   
    for k in range(1, NASTPS+1):
        xk0 = dlr * _fbeta1(a, nf)
        xk1 = dlr * _fbeta1(a + 0.5 * xk0, nf)
        xk2 = dlr * _fbeta1(a + 0.5 * xk1, nf)
        xk3 = dlr * _fbeta1(a + xk2, nf)
        a += (xk0 + 2 * xk1 + 2 * xk2 + xk3) / 6


    # Return to .../(2pi)  expansion
    a *= 4*np.pi
    return a

@njit(["float64[:](int32, int32, float64[:])", "float64(int32, int32, float64)"])
def AlphaS(nloop: int, nf: int, Q: float) -> float:
    if nloop==1:
        return AlphaS0(nf, Q)
    if nloop==2:
        return AlphaS1(nf, Q)
    raise ValueError('Only LO and NLO implemented!')


"""
***********************Anomalous dimensions of GPD in the moment space*****
Refer to the adim.py at https://github.com/kkumer/gepard.
"""

def S1(z: complex) -> complex:
    """ Harmonic sum S_1. """
    return np.euler_gamma + psi(z+1)

def non_singlet_LO(n: complex, nf: int, p: int, prty: int = 1) -> complex:
    """
    Non-singlet LO anomalous dimension.

    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et)
        prty (int): 1 for NS^{+}, -1 for NS^{-}, irrelevant at LO

    Returns:
        Non-singlet LO anomalous dimension.

    """
    return CF*(-3.0-2.0/(n*(1.0+n))+4.0*S1(n))



def singlet_LO(n: complex, nf: int, p: int, prty: int = 1) -> np.ndarray:
    """
    Singlet LO anomalous dimensions.

    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et)
        prty (int): C parity, irrelevant at LO

    Returns:
        2x2 complex matrix ((QQ, QG),
                            (GQ, GG))

    Here, n and nf are scalars.
    p is array of shape (N)
    However, it will still work if nf and n are arrays of shape (N)
    In short, this will work as long as n, nf, and p can be broadcasted together.

    """

    '''
    if(p == 1):
        qq0 = CF*(-3.0-2.0/(n*(1.0+n))+4.0*S1(n))
        qg0 = (-4.0*nf*TF*(2.0+n+n*n))/(n*(1.0+n)*(2.0+n))
        gq0 = (-2.0*CF*(2.0+n+n*n))/((-1.0+n)*n*(1.0+n))
        gg0 = -4.0*CA*(1/((-1.0+n)*n)+1/((1.0+n)*(2.0+n))-S1(n)) -11*CA/3. + 4*nf*TF/3.

        return np.array([[qq0, qg0],
                        [gq0, gg0]])
    
    if(p == -1):
        qq0 = CF*(-3.0-2.0/(n*(1.0+n))+4.0*S1(n))
        qg0 = (-4.0*nf*TF*(-1.0+n))/(n*(1.0+n))
        gq0 = (-2.0*CF*(2.0+n))/(n*(1.0+n))
        gg0 = -4.0*CA*(2/(n*(1.0+n))-S1(n)) -11*CA/3. + 4*nf*TF/3.

        return np.array([[qq0, qg0],
                        [gq0, gg0]])
    '''

    # Here, I am making the assumption that a is either 1 or -1
    qq0 = np.where(p>0,  CF*(-3.0-2.0/(n*(1.0+n))+4.0*S1(n)),           CF*(-3.0-2.0/(n*(1.0+n))+4.0*S1(n)))
    qg0 = np.where(p>0,  (-4.0*nf*TF*(2.0+n+n*n))/(n*(1.0+n)*(2.0+n)),  (-4.0*nf*TF*(-1.0+n))/(n*(1.0+n)) )
    gq0 = np.where(p>0,  (-2.0*CF*(2.0+n+n*n))/((-1.0+n)*n*(1.0+n)),    (-2.0*CF*(2.0+n))/(n*(1.0+n)))
    gg0 = np.where(p>0,  -4.0*CA*(1/((-1.0+n)*n)+1/((1.0+n)*(2.0+n))-S1(n)) -11*CA/3. + 4*nf*TF/3., \
        -4.0*CA*(2/(n*(1.0+n))-S1(n)) -11*CA/3. + 4*nf*TF/3. )

    # all of the four above have shape (N)
    # more generally, if p is a multi dimensional array, like (N1, N1, N2)... Then this could also work

    qq0_qg0 = np.stack((qq0, qg0), axis=-1)
    gq0_gg0 = np.stack((gq0, gg0), axis=-1)

    return np.stack((qq0_qg0, gq0_gg0), axis=-2)# (N, 2, 2)


"""
***********************Evolution operator of GPD in the moment space*******
Refer to the evolution.py at https://github.com/kkumer/gepard. Modifications are made.
"""

def lambdaf(n: complex, nf: int, p: int, prty: int = 1) -> np.ndarray:
    """
    Eigenvalues of the LO singlet anomalous dimensions matrix.

    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et)
        prty (int): 1 for NS^{+}, -1 for NS^{-}, irrelevant at LO

    Returns:
        lam[a, k]
        a in [+, -] and k is MB contour point index

    Normally, n and nf should be scalars. p should be (N)
    More generally, as long as they can be broadcasted, any shape is OK.

    """
    # To avoid crossing of the square root cut on the
    # negative real axis we use trick by Dieter Mueller
    gam0 = singlet_LO(n, nf, p, prty) # (N, 2, 2)
    aux = ((gam0[..., 0, 0] - gam0[..., 1, 1]) *
           np.sqrt(1. + 4.0 * gam0[..., 0, 1] * gam0[..., 1, 0] /
                   (gam0[..., 0, 0] - gam0[..., 1, 1])**2)) # (N)
    lam1 = 0.5 * (gam0[..., 0, 0] + gam0[..., 1, 1] - aux) # (N)
    lam2 = lam1 + aux  # (N)
    return np.stack([lam1, lam2], axis=-1) # shape (N, 2)


def projectors(n: complex, nf: int, p: int, prty: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projectors on evolution quark-gluon singlet eigenaxes.

    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et)
        prty (int): 1 for NS^{+}, -1 for NS^{-}, irrelevant at LO

    Returns:
         lam: eigenvalues of LO an. dimm matrix lam[a, k]  # Eq. (123)
          pr: Projector pr[k, a, i, j]  # Eq. (122)
               k is MB contour point index
               a in [+, -]
               i,j in {Q, G}

    n and nf will be scalars
    p will be shape (N)
    prty should be scalar (but maybe I can make it work with shape N)

    """
    gam0 = singlet_LO(n, nf, p, prty)    # (N, 2, 2)
    lam = lambdaf(n, nf, p, prty)        # (N, 2)
    den = 1. / (lam[..., 0] - lam[..., 1]) #(N)
    # P+ and P-
    ssm = gam0 - np.einsum('...,ij->...ij', lam[..., 1], np.identity(2)) #(N, 2, 2)
    ssp = gam0 - np.einsum('...,ij->...ij', lam[..., 0], np.identity(2)) #(N, 2, 2)
    prp = np.einsum('...,...ij->...ij', den, ssm) # (N, 2, 2)
    prm = np.einsum('...,...ij->...ij', -den, ssp) # (N, 2, 2)
    # We insert a-axis before i,j-axes, i.e. on -3rd place
    pr = np.stack([prp, prm], axis=-3) # (N, 2, 2, 2)
    return lam, pr # (N, 2) and (N, 2, 2, 2)

def evolop(j: complex, nf: int, p: int, Q: float):
    """
    GPD evolution operator E(j, nf, Q)[a,b].

    Args:
         j: MB contour points (Note: n = j + 1 !!)
         nf: number of effective fermion
         p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et)
         Q: final scale of evolution 

    Returns:
         Evolution operator E(j, nf, Q)[a,b] at given j nf and Q as 3-by-3 matrix
         - a and b are in the flavor space (non-singlet, singlet, gluon)

    In original evolop function, j, nf, p, and Q are all scalars.
    Here, j and nf will be scalars.
    p and Q will have shape (N)

    """
    #Alpha-strong ratio.
    R = AlphaS(nloop_alphaS, nf, Q)/AlphaS(nloop_alphaS, nf, Init_Scale_Q) # shape N
    R = np.array(R)

    #LO singlet anomalous dimensions and projectors
    lam, pr = projectors(j+1, nf, p)    # (N, 2) (N, 2, 2, 2)

    #LO pQCD beta function of GPD evolution
    b0 = beta0(nf) # scalar

    #Singlet LO evolution factor (alpha(mu)/alpha(mu0))^(-gamma/beta0) in (+,-) space
    Rfact = R[..., np.newaxis]**(-lam/b0) # (N, 2)     

    #Singlet LO evolution matrix in (u+d, g) space
    """
    # The Gepard code by K. Kumericki reads:
    evola0ab = np.einsum('kaij,ab->kabij', pr,  np.identity(2))
    evola0 = np.einsum('kabij,bk->kij', evola0ab, Rfact)
    # We use instead
    """ 
    evola0 = np.einsum('...aij,...a->...ij', pr, Rfact) # (N, 2, 2)

    #Non-singlet LO anomalous dimension
    gam0NS = non_singlet_LO(j+1, nf, p) #this function is already numpy compatible
    # shape (N)

    #Non-singlet evolution factor 
    evola0NS = R**(-gam0NS/b0) #(N)

    return [evola0NS, evola0] # (N) and (N, 2, 2)

def Moment_Evo(j: complex, nf: int, p: int, Q: float, ConfFlav: np.array) -> np.array:
    """
    Evolution of moments in the flavor space 

    Args:
        uneolved conformal moments in flavor space ConfFlav = [ConfMoment_uV, ConfMoment_ubar, ConfMoment_dV, ConfMoment_dbar, ConfMoment_g] 
        j: conformal spin j (conformal spin is actually j+2 but anyway): scalar
        t: momentum transfer squared
        nf: number of effective fermions; 
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et): array (N,)
        Q: final evolution scale: array(N,)

    Returns:
        Evolved conformal moments in flavor space (non-singlet, singlet, gluon)

        return shape (N, 5)
    """

    # flavor_trans (5, 5) ConfFlav (N, 5)

    # Transform the unevolved moments to evolution basis
    # ConfEvoBasis = np.einsum('...j,j', flav_trans, ConfFlav) # originally, output will be (5), I want it to be (N, 5)
    ConfEvoBasis = np.einsum('ij, ...j->...i', flav_trans, ConfFlav) # shape (N, 5)


    # Taking the non-singlet and singlet parts of the conformal moments
    ConfNS = ConfEvoBasis[..., :3] # (N, 3)
    ConfS = ConfEvoBasis[..., -2:] # (N, 2)

    # Calling evolution mulitiplier
    [evons, evoa] = evolop(j, nf, p, Q) # (N) and (N, 2, 2)

    # non-singlet part evolves multiplicatively
    EvoConfNS = evons[..., np.newaxis] * ConfNS # (N, 3)
    # singlet part mixes with the gluon
    EvoConfS = np.einsum('...ij, ...j->...i', evoa, ConfS) # (N, 2)

    # Recombing the non-singlet and singlet parts
    EvoConf = np.concatenate((EvoConfNS, EvoConfS), axis=-1) # (N, 5)
    # Inverse transform the evolved moments back to the flavor basis
    EvoConfFlav = np.einsum('...ij, ...j->...i', inv_flav_trans, EvoConf) #(N, 5)

    return EvoConfFlav