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

"""
***********************pQCD running coupling constant***********************
Here rundec is used instead.
"""

B00 = 11./3. * CA
B01 = -4./3. * TF

def beta0(nf: int) -> float:
    """ LO beta function of pQCD, will be used for LO GPD evolution. """
    return - B00 - B01 * nf

def AlphaS(nloop: int, nf: int, Q: float) -> float:
    """
    Alpha strong with initial scale set by Z boson mass Mz and alpha strong there alphaS = Alpha_Mz.

    Args:
        nloop: number of pQCD loop (order) of alpha strong
        nf: effective Fermion number
        Q: final scale
    
    Returns:
        single value of alphaS at final scale Q.

    """
    return rundec.CRunDec().AlphasExact(Alpha_Mz, Mz, Q, nf, nloop)

def beta_alphaS(nloop: int, nf: int, Q: float) -> float:
    rundec.CRunDec().SetBeta()

"""
***********************Anomalous dimensions of GPD in the moment space*****
Refer to the adim.py at https://github.com/kkumer/gepard.
"""

def S1(z: complex) -> complex:
    """ Harmonic sum S_1. """
    return np.euler_gamma + psi(z+1)

def non_singlet_LO(n: complex, nf: int, prty: int = 1) -> complex:
    """
    Non-singlet LO anomalous dimension.

    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        prty (int): 1 for NS^{+}, -1 for NS^{-}, irrelevant at LO

    Returns:
        Non-singlet LO anomalous dimension.

    """
    return CF*(-3.0-2.0/(n*(1.0+n))+4.0*S1(n))

def singlet_LO(n: complex, nf: int, prty: int = 1) -> np.ndarray:
    """
    Singlet LO anomalous dimensions.

    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        prty (int): C parity, irrelevant at LO

    Returns:
        2x2 complex matrix ((QQ, QG),
                            (GQ, GG))

    """
    qq0 = CF*(-3.0-2.0/(n*(1.0+n))+4.0*S1(n))
    qg0 = (-4.0*nf*TF*(2.0+n+n*n))/(n*(1.0+n)*(2.0+n))
    gq0 = (-2.0*CF*(2.0+n+n*n))/((-1.0+n)*n*(1.0+n))
    gg0 = (-22*CA/3.-8.0*CA*(1/((-1.0+n)*n)+1/((1.0+n)*(2.0+n))-S1(n))+8*nf*TF/3.)/2.

    return np.array([[qq0, qg0],
                     [gq0, gg0]])

"""
***********************Evolution operator of GPD in the moment space*******
Refer to the evolution.py at https://github.com/kkumer/gepard. Modifications are made.
"""

def lambdaf(n: complex, nf: int, prty: int = 1) -> np.ndarray:
    """
    Eigenvalues of the LO singlet anomalous dimensions matrix.

    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        prty (int): 1 for NS^{+}, -1 for NS^{-}, irrelevant at LO

    Returns:
        lam[a, k]
        a in [+, -] and k is MB contour point index

    """
    # To avoid crossing of the square root cut on the
    # negative real axis we use trick by Dieter Mueller
    gam0 = singlet_LO(n, nf, prty)
    aux = ((gam0[0, 0] - gam0[1, 1]) *
           np.sqrt(1. + 4.0 * gam0[0, 1] * gam0[1, 0] /
                   (gam0[0, 0] - gam0[1, 1])**2))
    lam1 = 0.5 * (gam0[0, 0] + gam0[1, 1] - aux)
    lam2 = lam1 + aux
    return np.stack([lam1, lam2])

def projectors(n: complex, nf: int, prty: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projectors on evolution quark-gluon singlet eigenaxes.

    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        prty (int): 1 for NS^{+}, -1 for NS^{-}, irrelevant at LO

    Returns:
         lam: eigenvalues of LO an. dimm matrix lam[a, k]  # Eq. (123)
          pr: Projector pr[k, a, i, j]  # Eq. (122)
               k is MB contour point index
               a in [+, -]
               i,j in {Q, G}

    """
    gam0 = singlet_LO(n, nf, prty)    
    lam = lambdaf(n, nf, prty)
    den = 1. / (lam[0, ...] - lam[1, ...])
    # P+ and P-
    ssm = gam0 - np.einsum('...,ij->...ij', lam[1, ...], np.identity(2))
    ssp = gam0 - np.einsum('...,ij->...ij', lam[0, ...], np.identity(2))
    prp = np.einsum('...,...ij->...ij', den, ssm)
    prm = np.einsum('...,...ij->...ij', -den, ssp)
    # We insert a-axis before i,j-axes, i.e. on -3rd place
    pr = np.stack([prp, prm], axis=-3)
    return lam, pr

def evolop(j: complex, nf: int, Q: float) -> np.ndarray:
    """
    GPD evolution operator E(j, nf, Q)[a,b].

    Args:
         j: MB contour points (Note: n = j + 1 !!)
         nf: number of effective fermion
         Q: final scale of evolution 

    Returns:
         Evolution operator E(j, nf, Q)[a,b] at given j nf and Q as 3-by-3 matrix
         - a and b are in the flavor space (non-singlet, singlet, gluon)

    """
    #Alpha-strong ratio.
    R = AlphaS(nloop_alphaS, nf, Q)/AlphaS(nloop_alphaS, nf, Init_Scale_Q)

    #LO singlet anomalous dimensions and projectors
    lam, pr = projectors(j+1, nf)    

    #LO pQCD beta function of GPD evolution
    b0 = beta0(nf)

    #Singlet LO evolution factor (alpha(mu)/alpha(mu0))^(-gamma/beta0) in (+,-) space
    Rfact = R**(-lam/b0)     

    #Singlet LO evolution matrix in (u+d, g) space
    """
    # The Gepard code by K. Kumericki reads:
    evola0ab = np.einsum('kaij,ab->kabij', pr,  np.identity(2))
    evola0 = np.einsum('kabij,bk->kij', evola0ab, Rfact)
    # We use instead
    """ 
    evola0 = np.einsum('aij,a->ij', pr, Rfact)

    #Non-singlet LO anomalous dimension
    gam0NS = non_singlet_LO(j+1, nf)

    #Non-singlet evolution factor (1 by 1 matrix)
    evola0NS =np.einsum('...,ij->...ij', R**(-gam0NS/b0), np.identity(1))

    # Direct sum of the 1-by-1 NS matrix and and the 2-by-2 Singlet matrix
    evo_dsum = np.zeros(np.add(evola0NS.shape, evola0.shape))
    evo_dsum[:evola0NS.shape[0],:evola0NS.shape[1]] = evola0NS
    evo_dsum[evola0NS.shape[0]:,evola0NS.shape[1]:] = evola0
    return evo_dsum

nftemp = 5

jtest = 0.5

evo = evolop(jtest, nftemp, 2* Init_Scale_Q)

print(evo)
