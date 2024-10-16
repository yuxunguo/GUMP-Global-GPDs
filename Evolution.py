"""
NLO QCD evolution of moment space GPD. Many credits to K. Kumericki at https://github.com/kkumer/gepard.

Note:
    Functions in this module have as first argument Mellin moment
    n = j + 1, where j is conformal moment used everywhere else.
    Thus, they should always be called as f(j+1, ...).

"""
# from cmath import exp
# from scipy.special import loggamma as clngamma
#from this import d

import numpy as np
from Parameters import Moment_Sum
from scipy.special import psi, zeta, gamma, orthogonal, loggamma
from math import factorial, log
from mpmath import mp, hyp2f1
from typing import Tuple, Union
from numba import vectorize, njit
import functools

"""
***********************QCD constants***************************************
Refer to the constants.py at https://github.com/kkumer/gepard.
"""

M_jpsi = 3.097
NC = 3
CF = (NC**2 - 1) / (2 * NC)
CA = NC
CG = CF - CA/2
TF = 0.5
Alpha_Mz = 0.1181
# All unit in GeV for dimensional quantities.
Mz = 91.1876
# One loop accuracy for running strong coupling constant. 
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

f_rho= 0.209 
f_phi = 0.221 # Change to 0.233
f_jpsi = 0.406

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

# Fixed quad function that allow more general function. The func here take input of shape (N,) and output (N,......) which doesn't have to be (N,)
def fixed_quadvec(func, a, b, n=100, args=()):
    rootsNLO, weightsNLO = orthogonal.p_roots(n)
    y = (b-a) * (rootsNLO + 1)/2.0 + a
    yfunc = func(y)
    return (b-a)/2.0*np.einsum('j,j...->...',weightsNLO,yfunc)
    
def pochhammer(z: Union[complex, np.ndarray], m: int) -> Union[complex, np.ndarray]:
    """Pochhammer symbol.

    Args:
        z: complex argument
        m: integer index

    Returns:
        complex: pochhammer(z,m)

    """
    p = z
    for k in range(1, m):
        p = p * (z + k)
    return p

poch = pochhammer  # just an abbreviation

def dpsi_one(z: complex, m: int) -> complex:
    """Polygamma - m'th derivative of Euler gamma at z."""
    # Algorithm from Vogt, cf. julia's implementation
    sub = 0j

    if z.imag < 10:
        subm = (-1/z)**(m+1) * factorial(m)
        while z.real < 10:
            sub += subm
            z += 1
            subm = (-1/z)**(m+1) * factorial(m)

    a1 = 1.
    a2 = 1./2.
    a3 = 1./6.
    a4 = -1./30.
    a5 = 1./42.
    a6 = -1./30.
    a7 = 5./66.

    if m != 1:
        for k2 in range(2, m+1):
            a1 = a1 * (k2-1)
            a2 = a2 * k2
            a3 = a3 * (k2+1)
            a4 = a4 * (k2+3)
            a5 = a5 * (k2+5)
            a6 = a6 * (k2+7)
            a7 = a7 * (k2+9)

    rz = 1. / z
    dz = rz * rz
    res = (sub + (-1)**(m+1) * rz**m *
           (a1 + rz * (a2 + rz * (a3 + dz *
            (a4 + dz * (a5 + dz * (a6 + a7 * dz)))))))
    return res

dpsi = np.vectorize(dpsi_one)

def S1(z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Harmonic sum S_1."""
    return np.euler_gamma + psi(z+1)

def S2(z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Harmonic sum S_2."""
    return zeta(2) - dpsi(z+1, 1)

def S3(z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Harmonic sum S_3."""
    return zeta(3) + dpsi(z+1, 2) / 2

def S4(z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Harmonic sum S_4."""
    return zeta(4) - dpsi(z+1, 3) / 6

def S2_prime(z: Union[complex, np.ndarray], prty: int) -> Union[complex, np.ndarray]:
    """Curci et al Eq. (5.25)."""
    # note this is related to delS2
    return (1+prty)*S2(z)/2 + (1-prty)*S2(z-1/2)/2

def S3_prime(z: Union[complex, np.ndarray], prty: int) -> Union[complex, np.ndarray]:
    """Curci et al Eq. (5.25)."""
    return (1+prty)*S3(z)/2 + (1-prty)*S3(z-1/2)/2

def delS2(z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Return Harmonic sum S_2 difference.

    Args:
        z: complex argument

    Returns:
        delS2((z+1)/2) From Eq. (4.13) of 1310.5394.
        Note halving of the argument.

    """
    return S2(z) - S2(z - 1/2)

def deldelS2(j: Union[complex, np.ndarray], k: int) -> Union[complex, np.ndarray]:
    """Return diference of harmonic sum S_2 differences.

    Args:
        j: complex argument
        k: integer index

    Returns:
        Equal to delS2((j+1)/2, (k+1)/2) From Eq. (4.38) of 1310.5394.
        Note halving of the argument.

    """
    return (delS2(j) - delS2(k)) / (4*(j-k)*(2*j+2*k+1))

def Sm1(z: Union[complex, np.ndarray], k: int) -> Union[complex, np.ndarray]:
    """Aux fun. FIXME: not tested."""
    return - log(2) + 0.5 * (1-2*(k % 2)) * (psi((z+2)/2) - psi((z+1)/2))

def MellinF2(n: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Return Mellin transform  i.e. x^(N-1) moment of Li2(x)/(1+x).

    Args:
        n: complex argument

    Returns:
        According to Eq. (33) in Bluemlein and Kurth, hep-ph/9708388

    """
    abk = np.array([0.9999964239, -0.4998741238,
                    0.3317990258, -0.2407338084, 0.1676540711,
                   -0.0953293897, 0.0360884937, -0.0064535442])
    psitmp = psi(n)
    mf2 = 0

    for k in range(1, 9):
        psitmp = psitmp + 1 / (n + k - 1)
        mf2 += (abk[k-1] *
                ((n - 1) * (zeta(2) / (n + k - 1) -
                 (psitmp + np.euler_gamma) / (n + k - 1)**2) +
                 (psitmp + np.euler_gamma) / (n + k - 1)))

    return zeta(2) * log(2) - mf2

def SB3(j: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Eq. (4.44e) of arXiv:1310.5394."""
    return 0.5*S1(j)*(-S2(-0.5+0.5*j)+S2(0.5*j))+0.125*(-S3(
             - 0.5 + 0.5 * j) + S3(0.5 * j)) - 2 * (0.8224670334241131 * (
                 -S1(0.5 * (-1 + j)) + S1(0.5 * j)) - MellinF2(1 + j))

def S2_tilde(n: Union[complex, np.ndarray], prty: int) -> Union[complex, np.ndarray]:
    """Eq. (30) of  Bluemlein and Kurth, hep-ph/9708388."""
    G = psi((n+1)/2) - psi(n/2)
    return -(5/8)*zeta(3) + prty*(S1(n)/n**2 - (zeta(2)/2)*G + MellinF2(n))

def lsum(m: Union[complex, np.ndarray], n: Union[complex, np.ndarray])-> Union[complex, np.ndarray]:
    
    return sum( (2*l+1)*(-1)**l * deldelS2((m+1)/2,l/2)/2 for l in range(1))

def lsumrev(m: Union[complex, np.ndarray], n: Union[complex, np.ndarray])-> Union[complex, np.ndarray]:
    
    return sum((2*l+1)*deldelS2((m+1)/2,l/2)/2 for l in range(1))










def non_singlet_LO(n:Union[complex, np.ndarray], nf: int, p: int, prty: int = 1) -> Union[complex, np.ndarray]:
    """Non-singlet LO anomalous dimension.

    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et)
        prty (int): 1 for NS^{+}, -1 for NS^{-}, irrelevant at LO

    Returns:
        Non-singlet LO anomalous dimension.
        
    It's an algebric equation, any shape of n should be fine.
    """
    return CF*(-3.0-2.0/(n*(1.0+n))+4.0*S1(n))

def singlet_LO(n: Union[complex, np.ndarray], nf: int, p: int, prty: int = 1) -> np.ndarray:
    """Singlet LO anomalous dimensions.

    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et)
        prty (int): C parity, irrelevant at LO

    Returns:
        2x2 complex matrix ((QQ, QG),
                            (GQ, GG))

    This will work as long as n, nf, and p can be broadcasted together.

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

    epsilon = 0.00001 * ( n == 1)

    # Here, I am making the assumption that a is either 1 or -1
    qq0 = np.where(p>0,  CF*(-3.0-2.0/(n*(1.0+n))+4.0*S1(n)),           CF*(-3.0-2.0/(n*(1.0+n))+4.0*S1(n)))
    qg0 = np.where(p>0,  (-4.0*nf*TF*(2.0+n+n*n))/(n*(1.0+n)*(2.0+n)),  (-4.0*nf*TF*(-1.0+n))/(n*(1.0+n)) )
    gq0 = np.where(p>0,  (-2.0*CF*(2.0+n+n*n))/((-1.0+n + epsilon)*n*(1.0+n)),    (-2.0*CF*(2.0+n))/(n*(1.0+n)))
    gg0 = np.where(p>0,  -4.0*CA*(1/((-1.0+n + epsilon)*n)+1/((1.0+n)*(2.0+n))-S1(n)) -11*CA/3. + 4*nf*TF/3., \
        -4.0*CA*(2/(n*(1.0+n))-S1(n)) -11*CA/3. + 4*nf*TF/3. )

    # all of the four above have shape (N)
    # more generally, if p is a multi dimensional array, like (N1, N1, N2)... Then this could also work

    qq0_qg0 = np.stack((qq0, qg0), axis=-1)
    gq0_gg0 = np.stack((gq0, gg0), axis=-1)

    return np.stack((qq0_qg0, gq0_gg0), axis=-2)# (N, 2, 2)

def non_singlet_NLO(n: complex, nf: int, prty: int) -> complex:
    """Non-singlet anomalous dimension.

    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        prty (int): 1 for NS^{+}, -1 for NS^{-}

    Returns:
        Non-singlet NLO anomalous dimension.
        
    This will work as long as n, nf, and prty can be broadcasted together.
    """
    # From Curci et al.
    nlo = (CF * CG * (
            16*S1(n)*(2*n+1)/poch(n, 2)**2 +
            16*(2*S1(n) - 1/poch(n, 2)) * (S2(n)-S2_prime(n/2, prty)) +
            64 * S2_tilde(n, prty) + 24*S2(n) - 3 - 8*S3_prime(n/2, prty) -
            8*(3*n**3 + n**2 - 1)/poch(n, 2)**3 -
            16*prty*(2*n**2 + 2*n + 1)/poch(n, 2)**3) +
           CF * CA * (S1(n)*(536/9 + 8*(2*n+1)/poch(n, 2)**2) - 16*S1(n)*S2(n) +
                      S2(n)*(-52/3 + 8/poch(n, 2)) - 43/6 -
                      4*(151*n**4 + 263*n**3 + 97*n**2 + 3*n + 9)/9/poch(n, 2)**3) +
           CF * nf * TF * (-(160/9)*S1(n) + (32/3)*S2(n) + 4/3 +
                           16*(11*n**2 + 5*n - 3)/9/poch(n, 2)**2)) / 4

    return nlo
    

def singlet_NLO(n: complex, nf: int, p: int, prty: int = 1) -> np.ndarray:
    """Singlet NLO anomalous dimensions matrix.
    
    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et)
        prty (int): C parity
            
    Returns:
        Matrix (LO, NLO) where each is in turn
        2x2 complex matrix
        ((QQ, QG),
        (GQ, GG))
        
    """
    
   # qq1 = non_singlet_NLO(n, nf, 1) + np.ones_like(p) * (-4*CF*TF*nf*((2 + n*(n + 5))*(4 + n*(4 + n*(7 + 5*n)))/(n-1)/n**3/(n+1)**3/(n+2)**2))
    
   # qg1 = np.ones_like(p) * 2*nf*TF*(S1(n)*(2*S1(n)*(-CF*(n-1)*(n+2)**2*(2+n+n**2)*n**2*(n+1)**2 + CA*n**2*(n+1)**2*(n+2)**2*(n**3 + n - 2)) - CF*(n-1)*(n+2)**2*(-4*n*(n+1)**3*(n+2)) - 2*CA*(4*(n-1)*n**3*(n+1)*(n+2)*(2*n + 3))) + S2_prime(n)*(2*CA*n**2*(n+1)**2*(n+2)**2*(n**3 + n - 2)) - CF*(n-1)*(n+1)**2*(64 + 160*(n-1) + 159*(n-1)**2 + 70*(n-1)**3 + 11*(n-1)**4 + n**2*(n+1)**2*5*(2 + n + n**2)) - 2*CA*(16 + 64*n + 104*n**2 + 128*n**3 + 85*n**4 + 36*n**5 + 25*n**6 + 15*n**7 + 6*n**8 + n**9))/(n-1)/n**3/(n+1)**3/(n+2)**3
    
  #  gq1 = np.ones_like(p) * (-CF)*(S1(n)*(S1(n)*(18*CA*n**2*(n+1)**2*(n+2)**2*(n**3 + n - 2) - 18*CF*(n-1)*n**2*(n+2)**2*(2 + 5*n + 5*n**2 + 3*n**3 + n**4)) + 18*CF*(n-1)*n**2*(n+2)**2*(10 + 27*n + 25*n**2 + 13*n**3 + 5*n**4) - 6*CA*n*(n+1)**2*(n+2)**2*(-12 + n*(-22 + 41*n + 17*n**3)) + 24*(n-1)*n**2*(n+1)**2*(n+2)**2*(n**2 + n + 2)*TF*nf) + S2(n)*(18*CA*n**2*(n+1)**2*(n+2)**2*(n**3 + n - 2) - 18*CF*(n-1)*n**2*(n+2)**2*(2 + 5*n + 5*n**2 + 3*n**3 + n**4)) + 9*CF*(n-1)*(n+2)**2*(-4 + n*(-12 + n*(-1 + 28*n + 43*n + 43*n**2 + 30*n**3 + 12*n**4))) + 2*CA*(2592 + 21384*(n-1) + 72582*(n-1)**2 + 128024*(n-1)**3 + 133818*(n-1)**4 + 88673*(n-1)**5 + 38822*(n-1)**6 + 10292*(n-1)**7 + 1662*(n-1)**8 + 109*(n-1)**9 - 9*n**2*(n+1)**2*(n+2)**2*(n**3 + n - 2)*S2_prime(n)) - 8*(n-1)*n**2*(n+1)**2*(n+2)**2*(16 + 27*n + 13*n**2 + 8*n**3)*TF*nf)/9/n**3/(n+1)**3/(n**2 + n - 2)**2
    
  #  gg1 = np.ones_like(p) * (S1(n)*(CA**2*(134/9 + 16*(2*n + 1)*(-2 + n*(n+1)*(n**2 + n + 2))/(n-1)**2*n**2*(n+1)**2*(n+2)**2 - 4*S2_prime(n)) - 40*CA*TF*nf/9) + CA**2*(-16/3 - (576 + n*(1488 + n*(560 + n*(-1632 + n*(-2344 + n*(1567 + n*(6098 + n*(6040 + 457*n*(6 + n)))))))))/9/(n-1)**2/n**3/(n+1)**3/(n+2)**3 + 8*(n*82 + n + 1)*S2_prime(n)/(n-1)*n*(n+1)*(n+2) - S3_prime(n) + 8*S2_tilde(n)) + 8*CF*TF*nf*(3 + (6+ n*(n+1)*(28 + 19*n*(n+1)))/n**2/(n+1)**2/(n**2 + n - 2))/9 + 2*CF*TF*nf*(-8 + n*(-8 + n*(-10 + n*(-22 + n*(-3 + n*(6 + n*(8 + n*(4 + n)))))))))
    
  #  qq1_qg1 = np.stack((qq1, qg1), axis=-1)
  #  gq1_gg1 = np.stack((gq1, gg1), axis=-1)

   # return np.stack((qq1_qg1, gq1_gg1), axis=-2)# (N, 2, 2)
    
    
   # qq1 = np.where(p>0, non_singlet_NLO(n, nf, 1) - 4*CF*TF*nf*(5*n**5+32*n**4+49*n**3+38*n**2 + 28*n+8)/((n-1)*n**3*(n+1)**3*(n+2)**2), non_singlet_NLO(n, nf, 1) - 4*CF*TF*nf*(5*n**5+32*n**4+49*n**3+38*n**2 + 28*n+8)/((n-1)*n**3*(n+1)**3*(n+2)**2))

   # qg1 = np.where(p>0,(-8*CF*nf*TF*((-4*S1(n))/n**2+(4+8*n + 26*n**3 + 11*n**4 + 15*(n*n))/(n**3*(1+n)**3*(2+n)) + ((2+n+n*n)*(5-2*S2(n) + 2*(S1(n)*S1(n))))/(n*(1+n)*(2+n))) - 8*CA*nf*TF*((8*(3+2*n)*S1(n))/((1+n)**2*(2+n)**2) + (2*(16+64*n+128*n**3+85*n**4+36*n**5+25*n**6 + 15*n**7+6*n**8+n**9+104*(n*n)))/( (-1+n)*n**3*(1+n)**3*(2+n)**3)+( (2+n+n*n)*(2*S2(n)-2*(S1(n)*S1(n))-2*S2(n/2)))/(n*(1+n)*(2+n))))/4, (-8*CF*nf*TF*((-4*S1(n))/n**2+(4+8*n + 26*n**3 + 11*n**4 + 15*(n*n))/(n**3*(1+n)**3*(2+n)) + ((2+n+n*n)*(5-2*S2(n) + 2*(S1(n)*S1(n))))/(n*(1+n)*(2+n))) - 8*CA*nf*TF*((8*(3+2*n)*S1(n))/((1+n)**2*(2+n)**2) + (2*(16+64*n+128*n**3+85*n**4+36*n**5+25*n**6 + 15*n**7+6*n**8+n**9+104*(n*n)))/( (-1+n)*n**3*(1+n)**3*(2+n)**3)+( (2+n+n*n)*(2*S2(n)-2*(S1(n)*S1(n))-2*S2(n/2)))/(n*(1+n)*(2+n))))/4)

   # gq1 = np.where(p>0, (-(32/3)*CF*nf*TF*((1+n)**(-2) + ((-(8/3)+S1(n))*(2+n+n*n))/((-1+n)*n*(1+n))) - 4*(CF*CF)*((-4*S1(n))/(1+n)**2-( -4-12*n+28*n**3+43*n**4 + 30*n**5+12*n**6-n*n)/((-1+n)*n**3*(1+n)**3) + ((2+n+n*n)*(10*S1(n)-2*S2(n)-2*(S1(n)*S1(n))))/((-1+n)*n*(1+n))) - 8*CF*CA*(((1/9)*(144+432*n-1304*n**3-1031*n**4 + 695*n**5+1678*n**6+1400*n**7+621*n**8+109*n**9 - 152*(n*n)))/((-1+n)**2*n**3*(1+n)**3*(2+n)**2) - ((1/3)*S1(n)*(-12-22*n+17*n**4 + 41*(n*n)))/((-1+n)**2*n**2*(1+n))+( (2+n+n*n)*(S2(n) + S1(n)*S1(n)-S2(n/2)))/((-1+n)*n*(1+n))))/4, (-(32/3)*CF*nf*TF*((1+n)**(-2) + ((-(8/3)+S1(n))*(2+n+n*n))/((-1+n)*n*(1+n))) - 4*(CF*CF)*((-4*S1(n))/(1+n)**2-( -4-12*n+28*n**3+43*n**4 + 30*n**5+12*n**6-n*n)/((-1+n)*n**3*(1+n)**3) + ((2+n+n*n)*(10*S1(n)-2*S2(n)-2*(S1(n)*S1(n))))/((-1+n)*n*(1+n))) - 8*CF*CA*(((1/9)*(144+432*n-1304*n**3-1031*n**4 + 695*n**5+1678*n**6+1400*n**7+621*n**8+109*n**9 - 152*(n*n)))/((-1+n)**2*n**3*(1+n)**3*(2+n)**2) - ((1/3)*S1(n)*(-12-22*n+17*n**4 + 41*(n*n)))/((-1+n)**2*n**2*(1+n))+( (2+n+n*n)*(S2(n) + S1(n)*S1(n)-S2(n/2)))/((-1+n)*n*(1+n))))/4)

   # gg1 = np.where(p>0, (CF*nf*TF*(8+(16*(-4-4*n-10*n**3+n**4+4*n**5+2*n**6 - 5*(n*n)))/((-1+n)*n**3*(1+n)**3*(2+n))) + CA*nf*TF*(32/3 - (160/9)*S1(n)+( (16/9)*(12+56*n+76*n**3+38*n**4+94*(n*n)))/((-1+n)*n**2*(1+n)**2*(2+n))) + CA*CA*(-64/3+(536/9)*S1(n)+(64*S1(n)*( -2-2*n+8*n**3+5*n**4+2*n**5+7*(n*n)))/((-1+n)**2*n**2*(1+n)**2*(2+n)**2) - ((4/9)*(576+1488*n-1632*n**3-2344*n**4+1567*n**5 + 6098*n**6+6040*n**7+2742*n**8+457*n**9+560*(n*n)))/( (-1+n)**2*n**3*(1+n)**3*(2+n)**3) - 16*S1(n)*S2(n/2)+(32*(1+n+n*n)*S2(n/2))/( (-1+n)*n*(1+n)*(2+n))-4*S3(n/2) + 32*(S1(n)/n**2-(5/8)*zeta(3)+MellinF2(n) - zeta(2)*(-psi(n/2)+psi((1+n)/2))/2)))/4, (CF*nf*TF*(8+(16*(-4-4*n-10*n**3+n**4+4*n**5+2*n**6 - 5*(n*n)))/((-1+n)*n**3*(1+n)**3*(2+n))) + CA*nf*TF*(32/3 - (160/9)*S1(n)+( (16/9)*(12+56*n+76*n**3+38*n**4+94*(n*n)))/((-1+n)*n**2*(1+n)**2*(2+n))) + CA*CA*(-64/3+(536/9)*S1(n)+(64*S1(n)*( -2-2*n+8*n**3+5*n**4+2*n**5+7*(n*n)))/((-1+n)**2*n**2*(1+n)**2*(2+n)**2) - ((4/9)*(576+1488*n-1632*n**3-2344*n**4+1567*n**5 + 6098*n**6+6040*n**7+2742*n**8+457*n**9+560*(n*n)))/( (-1+n)**2*n**3*(1+n)**3*(2+n)**3) - 16*S1(n)*S2(n/2)+(32*(1+n+n*n)*S2(n/2))/( (-1+n)*n*(1+n)*(2+n))-4*S3(n/2) + 32*(S1(n)/n**2-(5/8)*zeta(3)+MellinF2(n) - zeta(2)*(-psi(n/2)+psi((1+n)/2))/2)))/4)


    qq1 = non_singlet_NLO(n, nf, 1) + np.ones_like(p) * (-4*CF*TF*nf*(5*n**5+32*n**4+49*n**3+38*n**2 + 28*n+8)/((n-1)*n**3*(n+1)**3*(n+2)**2))
    
    qg1 = np.ones_like(p) * ((-8*CF*nf*TF*((-4*S1(n))/n**2+(4+8*n + 26*n**3 + 11*n**4 + 15*(n*n))/(n**3*(1+n)**3*(2+n)) + ((2+n+n*n)*(5-2*S2(n) + 2*(S1(n)*S1(n))))/(n*(1+n)*(2+n))) - 8*CA*nf*TF*((8*(3+2*n)*S1(n))/((1+n)**2*(2+n)**2) + (2*(16+64*n+128*n**3+85*n**4+36*n**5+25*n**6 + 15*n**7+6*n**8+n**9+104*(n*n)))/( (-1+n)*n**3*(1+n)**3*(2+n)**3)+( (2+n+n*n)*(2*S2(n)-2*(S1(n)*S1(n))-2*S2(n/2)))/(n*(1+n)*(2+n))))/4)
    
    gq1 = np.ones_like(p) * ((-(32/3)*CF*nf*TF*((1+n)**(-2) + ((-(8/3)+S1(n))*(2+n+n*n))/((-1+n)*n*(1+n))) - 4*(CF*CF)*((-4*S1(n))/(1+n)**2-( -4-12*n+28*n**3+43*n**4 + 30*n**5+12*n**6-n*n)/((-1+n)*n**3*(1+n)**3) + ((2+n+n*n)*(10*S1(n)-2*S2(n)-2*(S1(n)*S1(n))))/((-1+n)*n*(1+n))) - 8*CF*CA*(((1/9)*(144+432*n-1304*n**3-1031*n**4 + 695*n**5+1678*n**6+1400*n**7+621*n**8+109*n**9 - 152*(n*n)))/((-1+n)**2*n**3*(1+n)**3*(2+n)**2) - ((1/3)*S1(n)*(-12-22*n+17*n**4 + 41*(n*n)))/((-1+n)**2*n**2*(1+n))+( (2+n+n*n)*(S2(n) + S1(n)*S1(n)-S2(n/2)))/((-1+n)*n*(1+n))))/4)
    
    gg1 = np.ones_like(p) * ((CF*nf*TF*(8+(16*(-4-4*n-10*n**3+n**4+4*n**5+2*n**6 - 5*(n*n)))/((-1+n)*n**3*(1+n)**3*(2+n))) + CA*nf*TF*(32/3 - (160/9)*S1(n)+( (16/9)*(12+56*n+76*n**3+38*n**4+94*(n*n)))/((-1+n)*n**2*(1+n)**2*(2+n))) + CA*CA*(-64/3+(536/9)*S1(n)+(64*S1(n)*( -2-2*n+8*n**3+5*n**4+2*n**5+7*(n*n)))/((-1+n)**2*n**2*(1+n)**2*(2+n)**2) - ((4/9)*(576+1488*n-1632*n**3-2344*n**4+1567*n**5 + 6098*n**6+6040*n**7+2742*n**8+457*n**9+560*(n*n)))/( (-1+n)**2*n**3*(1+n)**3*(2+n)**3) - 16*S1(n)*S2(n/2)+(32*(1+n+n*n)*S2(n/2))/( (-1+n)*n*(1+n)*(2+n))-4*S3(n/2) + 32*(S1(n)/n**2-(5/8)*zeta(3)+MellinF2(n) - zeta(2)*(-psi(n/2)+psi((1+n)/2))/2)))/4)
                          
    #qq1_qg1 = np.stack((qq1, qg1), axis=0)
    #gq1_gg1 = np.stack((gq1, gg1), axis=0)
    
    n_tester = np.array([1.0])
    if type(n)==type(n_tester):
        qq1_qg1 = np.stack((qq1, qg1), axis=-1)
        gq1_gg1 = np.stack((gq1, gg1), axis=-1)
        result = np.stack((qq1_qg1,gq1_gg1),axis=-1)
    else:
        length = 1
        result = np.reshape(np.array([[qq1, qg1], [gq1, gg1]]),(length,2,2))

    return result# (N, 2, 2)    

    #return np.array([[qq1, qg1],
    #                 [gq1, gg1]])[np.newaxis,...]*np.ones_like(p)










"""
***********************Evolution operator of GPD in the moment space*******
Refer to the evolution.py at https://github.com/kkumer/gepard. Modifications are made.
"""

def lambdaf(n: complex, nf: int, p: int, prty: int = 1) -> np.ndarray:
    """Eigenvalues of the LO singlet anomalous dimensions matrix.

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
    """Projectors on evolution quark-gluon singlet eigenaxes.

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

 
def outer_subtract(arr1,arr2):   
    """Perform the outer product of two array at the last dimension, each has shape (N,..., m)
    
    | Generate shape (N,m,m), Here m = 2 for S/G 
    | result(i,j)=arr1(i)-arr2(j)

    Args:
        arr1 (np.array): 1st array in the outer subtract has shape (N,m)
        arr2 (np.array): 2nd array in the outer subtract has shape (N,m)

    Returns:
        result (np.ndarray): shape(N,m,m) given by result(i,j)=arr1(i)-arr2(j)
    """
    repeated_arr1 = np.repeat(arr1[..., np.newaxis], repeats=2, axis=-1)
    repeated_arr2 = np.repeat(arr2[..., np.newaxis], repeats=2, axis=-1)
    transposed_axes = list(range(repeated_arr1.ndim))
    transposed_axes[-2], transposed_axes[-1] = transposed_axes[-1], transposed_axes[-2]    
    return repeated_arr1-np.transpose(repeated_arr2, axes=transposed_axes)

def rmudep(nf, lamj, lamk, mu):
    """Scale dependent part of NLO evolution matrix 
    
    | Ref to the eq. (126) in hep-ph/0703179 
    | Here the expression is exactly the same as the ref, UNLIKE the Gepard with has an extra beta_0 to be canceled with amuindep

    Args:
        nf (int): number of effective fermions
        lamj (np.array): shape (N,2,2), each row is 2-by-2 matrix of anomalous dimension in the (S, G) basis
        lamk (np.array): shape (N,2,2), second row anomalous dimension for k
        mu (float): final scale to be evolved from inital scale Init_Scale_Q

    Returns:
        R_ij^ab(Q}|n=1) according to eq. (126) in hep-ph/0703179 
    """

    lamdif=outer_subtract(lamj,lamk)
        
    b0 = beta0(nf) # scalar
    b11 = b0 * np.ones_like(lamdif) + lamdif # shape (N,2,2)
    #print(b11)
    R = AlphaS(nloop_alphaS, nf, mu)/AlphaS(nloop_alphaS, nf, Init_Scale_Q) # shape N
    R = np.array(R)
    #print(R)
    Rpow = (1/R)[..., np.newaxis, np.newaxis, np.newaxis]**(b11/b0) # shape (N,2,2)
    #print((np.ones_like(Rpow) - Rpow) / b11)
    return (np.ones_like(Rpow) - Rpow) / b11 # shape (N,2,2)

def amuindep(j: complex, nf: int, p: int, prty: int = 1):
    """Result the P [gamma] P part of the diagonal evolution operator A.
    
    | Ref to eq. (124) in hep-ph/0703179 (the A operator are the same in both CSbar and MSbar scheme)
    | Here the expression is exactly the same as the ref, UNLIKE the Gepard with has an extra 1/beta_0 to be canceled with rmudep
    
    Args:
        j (complex): _description_
        nf (int): _description_
        p (int): _description_
        prty (int, optional): _description_. Defaults to 1.

    Returns:
        the P [gamma] P part of the diagonal evolution operator A.
    """
    lam, pr = projectors(j+1, nf, p, prty)
    
    gam0 = singlet_LO(j+1,nf,p)
    gam1 = singlet_NLO(j+1,nf,p)
    a1 = - gam1 + 0.5 * beta1(nf) * gam0 / beta0(nf)
    A = np.einsum('...aic,...cd,...bdj->...abij', pr, a1, pr)
   
    return A

'''
def amuindepNS(j: complex, nf: int, p: int, prty: int = 1):
    
    gam0NS = non_singlet_LO(j+1,nf,p)
    gam1NS = non_singlet_NLO(j+1,nf,p)
    a1 = - gam1NS + 0.5 * beta1(nf) * gam0NS / beta0(nf) 
    return a1
#print(amuindepNS(np.array([0.1,0.2]),2,1,1))
'''

def bmudep(mu, zn, zk, nf: int, p: int, NS: bool = False, prty: int = 1):
    """Return the off-diagonal part of the evolution operator B^{jk} combined with (alpha(Q)/alpha(mu_0)) ^ (-b/beta0)
    
    Check eq. (140) in hep-ph/0703179 for the expression of B^{jk}, and eq. (137) for the following factor

    Args:
        mu (float): scale evolved to from the initial scale Init_Scale_Q
        zn (np.array): shape (N,) moment j
        zk (np.array): shape (N,) moment k that differs from j for off-diagonal term
        nf (int): number of effective fermions
        p (int): parity of the GPDs 
        NS (bool, optional): True for non-singlet. NOT used here so set to default False.
        prty (int, optional): The party of the parton distributions +1 for S/G. Not used for NS so set to default 1.

    Returns:
       B^{jk}
    """
    R = AlphaS(nloop_alphaS, nf, mu)/AlphaS(nloop_alphaS, nf, Init_Scale_Q) # shape N
    R = np.array(R)
    b0 = beta0(nf)
    
    AAA = (S1((zn+zk+2)/2) -
           S1((zn-zk-2)/2) +
           2*S1(zn-zk-1) -
           S1(zn+1))
    # GOD = "G Over D" = g_jk / d_jk
    GOD_11 = 2 * CF * (
            2*AAA + (AAA - S1(zn+1))*(zn-zk)*(
                zn+zk + 3)/(zk+1)/(zk+2))
    nzero = np.zeros_like(GOD_11)
    GOD_12 = nzero
    GOD_21 = 2*CF*(zn-zk)*(zn+zk+3)/zn/(zk+1)/(zk+2)
    GOD_22 = 2 * CA * (2*AAA + (AAA - S1(
        zn + 1)) * (poch(zn, 4) / poch(zk, 4) -
                    1) + 2 * (zn-zk) * (
            zn+zk + 3) / poch(zk, 4)) * zk / zn
    god = np.array([[GOD_11, GOD_12], [GOD_21, GOD_22]])
    dm_22 = zk/zn
    dm_11 = np.ones_like(dm_22)
    dm_12 = np.zeros_like(dm_22)
    dm = np.array([[dm_11, dm_12], [dm_12, dm_22]])
    
    fac = 2**(zk-zn)*gamma(1+zn)/gamma(zn+3/2)*gamma(zk+3/2)/gamma(1+zk)*(2*zk+3)/(zn-zk)/(zn+zk+3) 

    #fac = (zk+1)*(zk+2)*(2*zn+3)/(zn+1)/(zn+2)/(zn-zk)/(zn+zk+3)
    
    lamn, pn = projectors(zn + 1, nf, p, prty)
    lamk, pk = projectors(zk + 1, nf, p, prty)

    proj_DM = np.einsum('...naif,fg...n,...nbgj->...nabij', pn, dm, pk)
    #print(proj_DM)
    proj_GOD = np.einsum('...naif,fg...n,...nbgj->...nabij', pn, god, pk)
    #print(proj_GOD)
     
    er1 = rmudep(nf, lamn, lamk, mu)
    
    bet_proj_DM = np.einsum('...nb,...nabij->...nabij', b0-lamk, proj_DM)      
    
    lamdif=outer_subtract(lamn,lamk)   

    Bjk = -np.einsum('...n,...nab,...nabij,...nb->...nij', fac,
                    er1*lamdif,
                    bet_proj_DM + proj_GOD,
                    R**(-lamk/b0)) 
    return Bjk

def evolop(j: complex, nf: int, p: int, mu: float):
    """Leading order GPD evolution operator E(j, nf, mu)[a,b].

    Args:
         j: MB contour points (Note: n = j + 1 !!)
         nf: number of effective fermion
         p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et)
         mu: final scale of evolution 

    Returns:
         Evolution operator E(j, nf, mu)[a,b] at given j nf and mu as 3-by-3 matrix
         - a and b are in the flavor space (non-singlet, singlet, gluon)

    In original evolop function, j, nf, p, and mu are all scalars.
    Here, j and nf will be scalars.
    p and mu will have shape (N)

    """
    #Alpha-strong ratio.
    R = AlphaS(nloop_alphaS, nf, mu)/AlphaS(nloop_alphaS, nf, Init_Scale_Q) # shape N
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


def Charge_Factor(particle:int):
    
    """The charge factors. For mesons it also multiplies with decay widths (f_m  is for meson m). Output is in evolution basis
    
    Args: 
        particle (integer): shape(5,)
    
    Returns:
        The charge factors are multiplied by decay constant of the m meson, f_m
        
     | particle=0 refers to DVCS
     | particle=1 refers to rho meson 
     | particle=3  refers to j/psi meson 
    """ 
    if (particle==0):
        return [0, -1/6, 0, 5/18, 5/18]
        
    if (particle==1):
        return f_rho *[1,0,0,1/np.sqrt(2),1/np.sqrt(2)]
    
    if (particle==3):
        return f_jpsi *[0,0,0,2/3,2/3]




# Need Wilson coefficients for evolution. Allows numerical pre-calculation of non-diagonal piece using Mellin-Barnes integral

def WilsonCoef(j: complex) -> complex:
    """Leading-order Wilson coefficient, elements used in both DVCS and DVMP

    Args:
        j (complex array): shape(N,) conformal spin j

    Returns:
        Leading-order Wilson coefficient (complex, could be an array)
    """
    return 2 ** (1+j) * gamma(5/2+j) / (gamma(3/2) * gamma(3+j))







def WilsonCoef_DVCS_LO(j: complex) -> complex:
    """LO Wilson coefficient of DVCS in the evolution basis (qVal, q_du_plus, q_du_minus, qSigma, g)
        
    Args:
        j (complex array): shape(N,) conformal spin j
        
    Returns:
        Wilson coefficient of shape (N,5) in the evolution basis (qVal, q_du_plus, q_du_minus, qSigma, g)
        
    | Charge factor are calculated such that the sum in the evolution basis are identical to the sum in the flavor basis
    | Gluon charge factor is the same as the singlet one, but the LO Wilson coefficient is zero in DVCS.
    """
    
    CWT = np.array([WilsonCoef(j), \
                    WilsonCoef(j), \
                    WilsonCoef(j), \
                    WilsonCoef(j),\
                    0 * j])
    return np.einsum('j, j...->j...', Charge_Factor(0), CWT)



def WilsonCoef_DVMP_LO(j: complex, nf: int, meson: int) -> complex:
    """LO Wilson coefficient of DVMP in the evolution basis (qVal, q_du_plus, q_du_minus, qSigma, g)

    Args:
        j (complex array): shape(N,) conformal spin j
        nf (int): number of effective fermions
        meson (int): 1 for rho, 2 for phi, and 3 for Jpsi
        
    Returns:
        Wilson coefficient of shape (N,5) in the evolution basis (qVal, q_du_plus, q_du_minus, qSigma, g)
        
    | Charge factor are calculated such that the sum in the evolution basis are identical to the sum in the flavor basis
    | Gluon charge factor is the same as the singlet one.
    | The meson decay constant, CF/NC, and eq are included in the prefactor of Wilson coefficient. 
    """
    # A factor of 3 coming from the asymptotic DA. The factors for S/G can be found in eq. (22b) of 2310.13837
    CWT = 3* np.array([WilsonCoef(j), \
                       WilsonCoef(j), \
                       WilsonCoef(j), \
                       1/ nf * WilsonCoef(j),\
                       2 /CF/ (j+3) * WilsonCoef(j)])
    
    '''
    if the meson is jpsi we are setting all the quark parts to zero
    '''
                      
    if(meson== 3):
    
     CWT = 3* np.array([0*j, \
                        0*j, \
                        0*j, \
                        0*j, \
                       2 /CF/ (j+3) * WilsonCoef(j)])

    return np.einsum('j, j...->j...', Charge_Factor(meson), CWT) 

  


def WilsonCoef_DVMP_NLO(j: complex, k: complex, nf: int, Q: float, muf: float, meson: int, p:int):
    """NLO Wilson coefficient of DVMP in the evolution basis (qVal, q_du_plus, q_du_minus, qSigma, g)
  
    | currently setting factorization scale and renormalization scale to the same as muf
    | Only singlet at this point.

    Args:
        j (complex array): shape(N,) conformal spin j of the GPD moment
        k (complex array): shape(N,) conformal spin k of the meson DA
        nf (int): number of effective fermions
        Q (float): the photon virtuality 
        mufact (float): the factorization scale mu_fact
        meson (int): 1 for rho, 2 for phi, and 3 for Jpsi
        p(int):parity 1 for vector -1 for axial. 
        
    Returns:
        Wilson coefficient of shape (N,5) in the evolution basis (qVal, q_du_plus, q_du_minus, qSigma, g)
    
    | Charge factor are calculated such that the sum in the evolution basis are identical to the sum in the flavor basis
    | Gluon charge factor is the same as the singlet one.
    | The meson decay constant, CF/NC, and eq are included in the prefactor of Wilson coefficient. 
    |Parity was called ptyk in GEPARD not with p. So we are setting  ptyk=p
    """
    
    mufact2 = muf ** 2
    mures2 =  muf ** 2
    muphi2 =  muf ** 2

   
    ptyk=p
    ptyk = 1
 
    
        
 
    
 
    MCQ1CF = -np.log(Q**2 / mufact2)-23/3+(0.5*(1.+3.*(1.+j)*(2.+j)))/((
             1 + j)**2*(2.+j)**2)+(0.5*(1.+3.*(1.+k)*(2.+k)))/((1.+k)**2
             *(2.+k)**2)+0.5*(-3.-2./((1.+j)*(2.+j))+4.*S1(1.+j))*((-
             0.5*(1.+(1.+j)*(2.+j)))/((1.+j)*(2.+j))-(0.5*(1.+(1.+k)*(
             2.+k)))/((1.+k)*(2.+k))-0.+S1(1.+j)+S1(1.+k))+0.5 * \
             ((-0.5*(1.+(1.+j)*(2.+j)))/((1.+j)*(2.+j))-(0.5*(1.+(1.+k
             )*(2.+k)))/((1.+k)*(2.+k))-0.+S1(1.+j)+S1(1.+k))*(-
             3.-2./((1.+k)*(2.+k))+4.*S1(1.+k))

    MCQ1BET0 = np.log(Q**2/mures2)/4-5/6+0.5/((1.+j)*(2.+j))+0.5/((
               1 + k)*(2.+k))+0.5*0.-S1(1.+j)-S1(1.+k)

    SUMA = 0j
    SUMB = 0j

    for LI in range(0, 1):
        SUMANDB = (0.125*(1.+2.*LI)*(S2(0.5*(1.+j))-S2(-0.5+0.5*(
           1.+j))+S2(-0.5+0.5*LI)-S2(0.5*LI)))/((0.5*(1.+j)-0.5*LI)*(2.+j+LI))
        SUMB += SUMANDB
        SUMA += (-1)**LI * SUMANDB
        
       

    DELc1aGJK = (-2.*ptyk)/((1.+j)*(2.+j)*(1.+k)*(2.+k))+(0.5
       *(-1.+(1.+j)*(2.+j))*(-2.+(1.+k)*(2.+k)*(S2(0.5*(1.+k)) -
       S2(-0.5+0.5*(1.+k))))*ptyk)/((1.+j)*(2.+j))+(1.+k)*(2.
       +k)*(2.+0.5*(1.+k)*(2.+k))*((0.25*k*(2.+(1.+k)**2)*(S2(
       0.5*(1.+j))-S2(-0.5+0.5*(1.+j))+S2(-0.5+0.5*k)-S2(0.5*k
       )))/((0.5*(1.+j)-0.5*k)*(2.+j+k)*(3.+2.*k)*(4.+(1.+k)*(2.
       +k)))-(0.25*(S2(0.5*(1.+j))-S2(-0.5+0.5*(1.+j))-S2(0.5
       *(1.+k))+S2(-0.5+0.5*(1.+k))))/((0.5*(1.+j)+0.5*(-1.-k))
       *(3.+j+k))+(0.25*(3.+k)*(2.+(2.+k)**2)*(S2(0.5*(1.+j)) -
       S2(-0.5+0.5*(1.+j))-S2(0.5*(2.+k))+S2(-0.5+0.5*(2.+k)))
       )/((0.5*(1.+j)+0.5*(-2.-k))*(4.+j+k)*(3.+2.*k)*(4.+(1.+k)
       *(2.+k))))*ptyk+2.*(j-k)*(3.+j+k)*(-SUMA-zeta(3)+S3(1.+j
       )+(0.125*(1.+k)*(S2(0.5*(1.+j))-S2(-0.5+0.5*(1.+j))-S2
       (0.5*(1.+k))+S2(-0.5+0.5*(1.+k)))*ptyk)/((0.5*(1.+j)+0.5
       *(-1.-k))*(3.+j+k)))

    
  
                                                 
    DELc1bGKJ = 1/((1.+k)*(2.+k))+0.5*(-2.-(1.+k)**2-((1.+j)*(2
       +j))/((1.+k)*(2.+k)))*(S2(0.5*(1.+j))-S2(-0.5+0.5*(1. +
       j)))-0.5*(1.+k)*(S2(0.5*(1.+k))-S2(-0.5+0.5*(1.+k)))-(0.125
       *(1.+k)*(2.+k)*(4.+(1.+k)*(2.+k))*(S2(0.5*(1.+j)) -
       S2(-0.5+0.5*(1.+j))-S2(0.5*(1.+k))+S2(-0.5+0.5*(1.+k)))) \
       / ((0.5*(1.+j)+0.5*(-1.-k))*(3.+j+k))-(0.5*(1.+k)*(2.+k)*(
          (0.25*k*(2.+(1.+k)**2)*(S2(0.5*(1.+j))-S2(-0.5+0.5*(1. +
       j))+S2(-0.5+0.5*k)-S2(0.5*k)))/((0.5*(1.+j)-0.5*k)*(2. +
       j+k))+(0.25*(3.+k)*(2.+(2.+k)**2)*(S2(0.5*(1.+j))-S2(-0.5
       +0.5*(1.+j))-S2(0.5*(2.+k))+S2(-0.5+0.5*(2.+k))))/((0.5
       *(1.+j)+0.5*(-2.-k))*(4.+j+k))))/(3.+2.*k)+2.*(-j+k)*(3
       +j+k)*(-SUMB-0.5*S1(1.+k)*(S2(0.5*(1.+j))-S2(-0.5+0.5
       *(1.+j)))+SB3(1+j))

    def MCQ1CG(sgntr:int):
        """Nonsinglet moment 
        
        Args:
            sgntr(integer): shape() charge parity or signature of the moment
            
        Returns: One of the non singlet moments
        """
        return 0.9565348003631189+DELc1aGJK-(2.*(1.+(1.+j)*(2.+j)
       )*(1.-sgntr))/((1.+j)**2*(2.+j)**2)-DELc1bGKJ*sgntr+(-(1 /
       ((1.+k)*(2.+k)))+2.*S1(1.+k))*(1.-sgntr+0.5*(1.+j)*(2.+j
       )*sgntr*(-S2(0.5*j)+S2(0.5*(1.+j)))) + (2.*sgntr*ptyk)/(
               (1.+j)*(2.+j)*(1.+k)*(2.+k))-(
               2.*(1.+(1.+k)*(2.+k))*(1.+
       ptyk))/((1.+k)**2*(2.+k)**2)+(-(1/((1.+j)*(2.+j))) +
       2*S1(1.+j))*(1.+ptyk-0.5*(1.+k)*(2.+k)*(-S2(0.5*k) +
       S2(0.5*(1.+k)))*ptyk)+2.*(1.+j)*(2.+j)*((-0.5*(-1.+(1.+j)*(
       2.+j)))/((1.+j)**2*(2.+j)**2)+zeta(3)-(0.5*sgntr*(-S2(0.5 *
       j)+S2(0.5*(1.+j))))/((1.+j)*(2.+j))-S3(1.+j)-sgntr*SB3(
       1+j))+2.*(1.+k)*(2.+k)*((-0.5*(-1.+(1.+k)*(2.+k)))/((1. +
       k)**2*(2.+k)**2)+zeta(3)-S3(1.+k)+(0.5*(-S2(0.5*k) +
           S2(0.5*(1.+k)))*ptyk)/((1.+k)*(2.+k))+ptyk*SB3(1+k))

    def CQNS(sgntr:int):
        """The nonsinglet amplitude
        
        Args:
            sgntr(integer): shape() charge parity or signature of the moment
            
        Returns: One of the non singlet moments
        """
       
        return AlphaS(nloop_alphaS, nf, Init_Scale_Q) / 2 / np.pi * (CF * MCQ1CF + (CF - NC/2) * MCQ1CG(sgntr) + beta0(nf) * MCQ1BET0)
    
    CQPS = AlphaS(nloop_alphaS, nf, Init_Scale_Q) / 2 / np.pi * ((-np.log(Q**2/mufact2) - 1 + 2*S1(j+1) + 2*S1(k+1) - 1)*(-2*(4 + 3*j + j**2)/j/(j+1)/(j+2)/(j+3)) - (1/2 + 1/(j+1)/(j+2) + 1/(k+1)/(k+2))*2/(j+1)/(j+2) + k*(k+1)*(k+2)*(k+3)*(deldelS2((j+1)/2,k/2) - deldelS2((j+1)/2,(k+2)/2))/2/(2*k+3) )
    
    CGNC =  AlphaS(nloop_alphaS, nf, Init_Scale_Q) / 2 / np.pi * ((-np.log(Q**2 / mufact2) + S1(j+1) + 3*S1(k+1)/2 + 0.5 + 1/(j+1)/(j+2))*(4*S1(j+1) + 4/(j+1)/(j+2) - 12/j/(j+3)) * 0.5 - (3 * (2*S1(j+1) + S1(k+1) - 6) / j / (j+3)) + (8 + 4*np.pi**2 / 6 - (k+1)*(k+2)*delS2((k+1)/2))/8 - delS2((j+1)/2)/2 - (10*(j+1)*(j+2) + 4)/(j+1)**2 / (j+2)**2 - (delS2((j+1)/2) / 2 / (k+1) / (k+2) - (k-1)*deldelS2((j+1)/2,k/2)/(2*k+3) + (k+4)*deldelS2((j+1)/2,(k+2)/2)/(2*k+3))*k*(k+1)*(k+2)*(k+3)/4 + ((k+1)*(k+2)*S1(k+1)-2)/(j+1)/(j+2)/(k+1)/(k+2))
    
    CGCF = AlphaS(nloop_alphaS, nf, Init_Scale_Q) / 2 / np.pi * ((-np.log(Q**2 / muphi2) + S1(j+1) + S1(k+1) - 3/4 - 1/2/(k+1)/(k+2) - 1/(j+1)/(j+2))*(4*S1(k+1) - 3 - 2/(k+1)/(k+2))/2 + (-np.log(Q**2 / mufact2) + 1 + 3*S1(j+1) - 0.5 + (2*S1(j+1)-1)/(k+1)/(k+2) -1/(j+1)/(j+2))*(-(4+2*(j+1)*(j+2))/(j+1)/(j+2)/(j+3))*(j+3)/4 - (35 - ((k+1)*(k+2) + 2)*delS2((k+1)/2) + 4/(k+1)**2/(k+2)**2)/8 + (((k+1)*(k+2) + 2)*S1(j+1)/(k+1)/(k+2) +1)/(j+1)/(j+2) + (delS2((j+1)/2)/2/(k+1)/(k+2) - ((k-1)*k*deldelS2((j+1)/2,k/2) - (k+3)*(k+4)*deldelS2((j+1)/2,(k+2)/2))/2/(2*k+3))*((k+1)*(k+2)*((k+1)*(k+2) +2)/4) - ((k+1)*(k+2) + 2)/2/(j+1)/(j+2)/(k+1)/(k+2))
    
   
  

    CWT= np.array([CQNS(1)*(3 * 2 ** (1+j) * gamma(5/2+j) / (gamma(3/2) * gamma(3+j))), CQNS(-1)*(3 * 2 ** (1+j) * gamma(5/2+j) / (gamma(3/2) * gamma(3+j))),  CQNS(-1)*(3 * 2 ** (1+j) * gamma(5/2+j) / (gamma(3/2) * gamma(3+j))), \
                   CQNS(1)*(3 * 2 ** (1+j) * gamma(5/2+j) / (gamma(3/2) * gamma(3+j)))/nf + CQPS * (3 * 2 ** (1+j) * gamma(5/2+j) / (gamma(3/2) * gamma(3+j))), \
                   2/ CF / (j + 3) * 3 * 2 ** (1+j) * gamma(5/2+j) / (gamma(3/2) * gamma(3+j)) * (NC*CGNC + CF*CGCF + beta0(nf)*np.log(mufact2/mures2)/2)],dtype=complex) #+ beta0(nf)*np.log(mufact/mures)

    if (meson==3):
        
        """
        |if the meson is jpsi we are setting all the quark parts to zero except the pure singlet
        """ 

        CWT= np.array([0*j, 0*j,  0*j, \
                  CQPS * (3 * 2 ** (1+j) * gamma(5/2+j) / (gamma(3/2) * gamma(3+j))), \
                2/ CF / (j + 3) * 3 * 2 ** (1+j) * gamma(5/2+j) / (gamma(3/2) * gamma(3+j)) * (NC*CGNC + CF*CGCF + beta0(nf)*np.log(mufact2/mures2)/2)],dtype=complex) #+ beta0(nf)*np.log(mufact/mures)

    return  np.einsum('j, j...->j...', Charge_Factor(meson), CWT)    
      
   

def np_cache(function):
    @functools.cache
    def cached_wrapper(tupled_arr):
        [arr, nf, p, Q, meson, muset] = np.array(tupled_arr[0]), int(tupled_arr[1]), int(tupled_arr[2]), float(tupled_arr[3]), int(tupled_arr[4]), float(tupled_arr[5])
        return function(arr, nf, p, Q, meson, muset)

    @functools.wraps(function)
    def wrapper(arr, nf, p, Q, meson, muset):
        return cached_wrapper(tuple((tuple(arr), nf, p, Q, meson, muset)))

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper

def np_cache_moment(function):
    
    def totuple(a):
        try:
            return tuple(totuple(i) for i in a)
        except TypeError:
            return a
    
    @functools.cache
    def cached_wrapper(tupled_arr):
        [arr, nf, p, mu, t, xi, Para, momshift] = np.array(tupled_arr[0]), int(tupled_arr[1]), int(tupled_arr[2]), float(tupled_arr[3]), float(tupled_arr[4]), float(tupled_arr[5]), np.array(tupled_arr[6]) ,int(tupled_arr[7])
        return function(arr, nf, p, mu, t, xi,Para, momshift)

    @functools.wraps(function)
    def wrapper(arr, nf, p, mu, t, xi,Para, momshift):
        return cached_wrapper(tuple((tuple(arr), nf, p, mu,t,xi,totuple(Para), momshift)))

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper
    
def Moment_Evo_LO(j: np.array, nf: int, p: int, mu: float, ConfFlav: np.array) -> np.array:
    """Leading order evolution of moments in the flavor space 

    Args:
        j: conformal spin j (conformal spin is actually j+2 but anyway): shape (N,)
        nf: number of effective fermions, scalar
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et), scalars
        mu: final evolution scale: array(N,), scalar
        ConfFlav: uneolved conformal moments in flavor space ConfFlav = [ConfMoment_uV, ConfMoment_ubar, ConfMoment_dV, ConfMoment_dbar, ConfMoment_g] shape (N,5)

    Returns:
        Evolved conformal moments in the evolution basis (qVal, q_du_plus, q_du_minus, qSigma, g)
        return shape (N, 5)
        
    | The j here accept array input and preferrably just 1D for the contour integral in j. 
    | Therefore, j has shape (N,) where N is the interpolating order of the fixed quad.
    | Other quantities must be broadcastable with j and thus they should be preferrably scalar.
    """
    assert j.ndim == 1, "Check dimension of j, must be 1D array" # shape (N,)
    
    # flavor_trans (5, 5) ConfFlav (N, 5)

    # Transform the unevolved moments to evolution basis
    # ConfEvoBasis = np.einsum('...j,j', flav_trans, ConfFlav) # originally, output will be (5), I want it to be (N, 5)
    ConfEvoBasis = np.einsum('ij, ...j->...i', flav_trans, ConfFlav) # shape (N, 5)

    # Taking the non-singlet and singlet parts of the conformal moments
    ConfNS = ConfEvoBasis[..., :3] # (N, 3)
    ConfS = ConfEvoBasis[..., -2:] # (N, 2)

    # Calling evolution mulitiplier
    [evons, evoa] = evolop(j, nf, p, mu) # (N) and (N, 2, 2)

    # non-singlet part evolves multiplicatively
    EvoConfNS = evons[..., np.newaxis] * ConfNS # (N, 3)
    # singlet part mixes with the gluon
    EvoConfS = np.einsum('...ij, ...j->...i', evoa, ConfS) # (N, 2)

    # Recombing the non-singlet and singlet parts
    EvoConf = np.concatenate((EvoConfNS, EvoConfS), axis=-1) # (N, 5)
        
    return EvoConf

def CFF_Evo_LO(j: np.array, nf: int, p: int, mu: float, ConfFlav: np.array) -> np.array:
    """Leading order evolution of the combination of DVCS Wilson coefficient and conformal moments

    Args:
        j: conformal spin j (conformal spin is actually j+2 but anyway): array (N,)
        nf: number of effective fermions; scalar
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et); scalars
        mu: final evolution scale; scalar
        ConfFlav: uneolved conformal moments in flavor space ConfFlav = [ConfMoment_uV, ConfMoment_ubar, ConfMoment_dV, ConfMoment_dbar, ConfMoment_g]; shape (N,5)

    Returns:
        Evolved conformal moments times the DVCS Wilson coefficients in the evolution basis (qVal, q_du_plus, q_du_minus, qSigma, g);
        ingredients for the Mellin-Barnes integral for CFF 
        return shape (N, 5)
        
    | The j here accept array input and preferrably just 1D for the contour integral in j. 
    | Therefore, j has shape (N,) where N is the interpolating order of the fixed quad.
    | Other quantities must be broadcastable with j and thus they should be preferrably scalar
    """
    assert j.ndim == 1, "Check dimension of j, must be 1D array" # shape (N,)
    # flavor_trans (5, 5) ConfFlav (N, 5)

    # Transform the unevolved moments to evolution basis
    # ConfEvoBasis = np.einsum('...j,j', flav_trans, ConfFlav) # originally, output will be (5), I want it to be (N, 5)
    ConfEvoBasis = np.einsum('ij, ...j->...i', flav_trans, ConfFlav) # shape (N, 5)

    # Taking the non-singlet and singlet parts of the conformal moments
    ConfNS = ConfEvoBasis[..., :3] # (N, 3)
    ConfS = ConfEvoBasis[..., -2:] # (N, 2)
    
    # Calling evolution mulitiplier
    [evons, evoa] = evolop(j, nf, p, mu) # (N) and (N, 2, 2)
    
    # Combine with the corresponding wilson coefficient in the evolution basis
    EvoWCNS = np.einsum('i...,...->...i', WilsonCoef_DVCS_LO(j)[:3,...], evons) # shape (3,N), (N) -> (N,3)
    EvoWCS = np.einsum('ik,kij->kij', WilsonCoef_DVCS_LO(j)[-2:,...], evoa) # Shape (2, N), (N,2,2) ->(N,2,2)
  
    # Non-singlet part evolves multiplicatively
    EvoConfNS = np.einsum('...j,...j->...j',EvoWCNS, ConfNS)
    # Singlet part mixes with the gluon
    EvoConfS = np.einsum('kij,kj->ki', EvoWCS, ConfS)
    
    # Recombing the non-singlet and singlet parts
    EvoConf = np.concatenate((EvoConfNS,EvoConfS),axis=-1)

    return EvoConf

def TFF_Evo_LO(j: np.array, nf: int, p: int, mu: float, ConfFlav: np.array, meson: int) -> np.array:
    """Leading order evolution of the combination of DVMP Wilson coefficient and conformal moments

    Args:
        j: conformal spin j (conformal spin is actually j+2 but anyway): array (N,)
        nf: number of effective fermions; scalar
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et); scalar
        mu: final evolution scale; scalar
        ConfFlav: uneolved conformal moments in flavor space ConfFlav = [ConfMoment_uV, ConfMoment_ubar, ConfMoment_dV, ConfMoment_dbar, ConfMoment_g]; shape (N,5)
        meson (int): 1 for rho, 2 for phi, 3 for Jpsi

    Returns:
        Evolved conformal moments times the DVMP Wilson coefficients in the evolution basis (qVal, q_du_plus, q_du_minus, qSigma, g);
        Ingredients for the Mellin-Barnes integral for TFF 
        return shape (N, 5)
        
    | The j here accept array input and preferrably just 1D for the contour integral in j. 
    | Therefore, j has shape (N,) where N is the interpolating order of the fixed quad.
    | Other quantities must be broadcastable with j and thus they should be preferrably scalar
    """
    
    assert j.ndim == 1, "Check dimension of j, must be 1D array" # shape (N,)
    
    # flavor_trans (5, 5) ConfFlav (N, 5)
    # Transform the unevolved moments to evolution basis
    ConfEvoBasis = np.einsum('ij, ...j->...i', flav_trans, ConfFlav) # shape (N, 5)

    # Taking the non-singlet and singlet parts of the conformal moments
    ConfNS = ConfEvoBasis[..., :3] # (N, 3)
    ConfS = ConfEvoBasis[..., -2:] # (N, 2)

    # Calling evolution mulitiplier
    [evons, evoa] = evolop(j, nf, p, mu) # (N) and (N, 2, 2)
    
    # Combine with the corresponding wilson coefficient in the evolution basis
    EvoWCNS = np.einsum('i...,...->...i',WilsonCoef_DVMP_LO(j,nf, meson)[:3,...],evons)
    EvoWCS = np.einsum('ik,kij->kij', WilsonCoef_DVMP_LO(j,nf, meson)[-2:,...], evoa)

    # Non-singlet part evolves multiplicatively
    EvoConfNS = np.einsum('...j,...j->...j',EvoWCNS, ConfNS)
    # Singlet part mixes with the gluon
    EvoConfS = np.einsum('kij,kj->ki', EvoWCS, ConfS)
    
    # Recombing the non-singlet and singlet parts
    EvoConf = np.concatenate((EvoConfNS,EvoConfS),axis=-1)

    return EvoConf

@np_cache
def WCoef_Evo_NLO(j: np.array, nf: int, p: int, Q: float, meson: int, muf: float) -> Tuple[np.ndarray, np.ndarray]:
    """Next-to-leading order evolution of the DVMP Wilson coefficient (Evolved Wilson coefficient method)

    Args:
        j: conformal spin j (conformal spin is actually j+2 but anyway): array (N,)
        nf: number of effective fermions; scalars
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et); scalars
        Q (float): photon virtuality. At NLO the scale Q enters the amplitudes through logs.
        ConfFlav: uneolved conformal moments in flavor space ConfFlav = [ConfMoment_uV, ConfMoment_ubar, ConfMoment_dV, ConfMoment_dbar, ConfMoment_g]; shape (N,5)
        meson (int): 1 for rho, 2 for phi, 3 for Jpsi
        muf: factorization evolution scale; scalar

    Returns:
        Evolved DVMP Wilson coefficients in the evolution basis;
        CWilsonT_ev_NS_tot: Non-singlet evolved Wilson coefficient with shape (N,3)
        CWilsonT_ev_SG_tot Singlet/Gluon evolved Wilson coefficient with shape (N,2)

    | The j here accept array input and preferrably just 1D for the contour integral in j. 
    | Therefore, j has shape (N,) where N is the interpolating order of the fixed quad.
    | Other quantities must be broadcastable with j and thus they should be preferrably scalar
    """
    
    assert j.ndim == 1, "Check dimension of j, must be 1D array" # shape (N,)
    
    # Separate out NS and S/G Wilson coefficients
    CWNS = WilsonCoef_DVMP_LO(j, nf, meson)[:3]
    CWSG = WilsonCoef_DVMP_LO(j, nf, meson)[-2:]
     
    #Set up evolution operator for WCs
    Alphafact = np.array(AlphaS(nloop_alphaS, nf, muf)) / np.pi / 2
    R = AlphaS(nloop_alphaS, nf, muf)/AlphaS(nloop_alphaS, nf, Init_Scale_Q) # shape N
    R = np.array(R)

    b0 = beta0(nf)
    lam, pr = projectors(j+1, nf, p)
    pproj = amuindep(j, nf, p)
     
    rmu1 = rmudep(nf, lam, lam, muf)
    Rfact = R[...,np.newaxis]**(-lam/b0)  # LO evolution (alpha(mu)/alpha(mu0))^(-gamma/beta0)

    # S/G LO evolution operator
    evola0 = np.einsum('...aij,...a->...ij', pr, Rfact)
    # NS LO evolution operator
    gam0NS = non_singlet_LO(j+1, nf, p)
    evola0NS = R**(-gam0NS/b0)
    
    # LO evolved singlet and non-singlet WCs
    CWNS_ev0 = np.einsum('...,i...->...i',evola0NS,CWNS)
    CWSG_ev0 = np.einsum('...ij,i...->...ij',evola0,CWSG)
    
    # S/G diagonal NLO evolution operator     
    evola1_diag_ab = np.einsum('kab,kabij->kabij', rmu1, pproj)
    evola1_diag = np.einsum('...abij,...b,...->...ij', evola1_diag_ab, Rfact,Alphafact)

    # NLO NS evolutioon set to zero
    CWNS_ev1 = np.zeros_like(CWNS_ev0)
    # S/G diagonal NLO evolution operator
    CWSG_ev1_diag = np.einsum('...ij,i...->...ij',evola1_diag,CWSG)
        
    # Following are the second integral resumming the off diagonal pieces, note that (j,k) meshgrid is used for vectorized j and k input. Check the paper for expression
    reK = -0.8
    Max_imK = 150
     
    def non_diag_integrand_mesh(k):
        
        jmesh, kmesh= np.meshgrid(j,k)        
        meshshape=jmesh.shape

        jmesh=jmesh.reshape(-1)
        kmesh=kmesh.reshape(-1)
        
        CWk = WilsonCoef_DVMP_LO(jmesh+kmesh+1, nf, meson)[-2:]        
        Bjk = np.array(bmudep(muf, np.array(jmesh+kmesh+1,dtype=complex), np.array(jmesh,dtype=complex), nf,p))*Alphafact
        out = np.einsum('...ij,...->...ij',np.einsum('...ij,i...->...ij',Bjk,CWk), 1/4*np.tan(np.pi * kmesh / 2))  

        outorishape=out.shape
        out=out.reshape(meshshape[0],meshshape[1],*outorishape[1:])

        return out
    
    intd_non_diag_ev1_vec=fixed_quadvec(lambda imK:non_diag_integrand_mesh(reK+1j*imK)+non_diag_integrand_mesh(reK-1j*imK),0,Max_imK,300)
    
    # Off-diagonal piece for the NS evolution
    CWSG_ev1_non_diag = intd_non_diag_ev1_vec     
    # Combine the diagonal and off-diagonal pieces
    CWSG_ev1 = CWSG_ev1_diag + CWSG_ev1_non_diag
     
    # NLO Wilson coefficient combined with leading-order evolved conformal moment
    CWilsonT_1_SG = WilsonCoef_DVMP_NLO(j,0,nf,Q, muf, meson)[-2:]     
    CWilsonT_1_SG_ev0 = np.einsum('i...,...ij->...ij',CWilsonT_1_SG,evola0)
     
    # LO plus NLO evolution    
    CWilsonT_ev_NS_tot = CWNS_ev0 + CWNS_ev1
    CWilsonT_ev_SG_tot = CWSG_ev0 + CWSG_ev1 + CWilsonT_1_SG_ev0
    
    return CWilsonT_ev_NS_tot, CWilsonT_ev_SG_tot

def TFF_Evo_NLO_evWC(j: np.array, nf: int, p: int, Q: float, ConfFlav: np.array, meson: int, muf: float) -> np.array:
    """Next-to-leading order evolved DVMP Wilson coefficients in the flavor space combined with the conformal moments (Evolved Wilson coefficient method)
    
    Args:
        j: conformal spin j (conformal spin is actually j+2 but anyway): array (N,)
        nf: number of effective fermions; scalars
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et); scalars
        Q (float): photon virtuality. At NLO the scale Q enters the amplitudes through logs.
        ConfFlav: uneolved conformal moments in flavor space ConfFlav = [ConfMoment_uV, ConfMoment_ubar, ConfMoment_dV, ConfMoment_dbar, ConfMoment_g]; shape (N,5)
        meson (int): 1 for rho, 2 for phi, 3 for Jpsi
        muf: factorization evolution scale; scalar

    Returns:
        Evolved conformal moments times the DVMP Wilson coefficients in the evolution basis (qVal, q_du_plus, q_du_minus, qSigma, g);
        Ingredients for the Mellin-Barnes integral for TFF 
        return shape (N, 5)
        
    | The j here accept array input and preferrably just 1D for the contour integral in j. 
    | Therefore, j has shape (N,) where N is the interpolating order of the fixed quad.
    | Other quantities must be broadcastable with j and thus they should be preferrably scalar
    """
    
    assert j.ndim == 1, "Check dimension of j, must be 1D array" # shape (N,)
    
    # Retrive the evolved Wilson coefficient from WCoef_Evo_NLO()
    [CWilsonT_ev_NS_tot, CWilsonT_ev_SG_tot] = WCoef_Evo_NLO(j, nf, p, Q, meson, muf)

    # Transform the unevolved moments to evolution basis
    ConfEvoBasis = np.einsum('ij, ...j->...i', flav_trans, ConfFlav) # shape (N, 5)   
    
    # Taking the non-singlet and singlet parts of the conformal moments
    ConfNS = ConfEvoBasis[..., :3] # (N, 3)
    ConfSG = ConfEvoBasis[..., -2:] # (N, 2)
    
    # Combine the evolved Wilson coefficient with conformal moments
    EvoConfNS = np.einsum('...j,...j->...j', CWilsonT_ev_NS_tot, ConfNS)
    EvoConfSG = np.einsum('...ij,...j->...i', CWilsonT_ev_SG_tot, ConfSG)

    # Recombing the non-singlet and singlet parts
    EvoConf = np.concatenate((EvoConfNS, EvoConfSG), axis=-1) # (N, 5)
    
    return EvoConf

# Turn off the cache to reduce hashing time if only one evolved moment is calculated for a set of parameters at a given kinematics. Otherwise cache it.
@np_cache_moment
def Moment_Evo_NLO(j: np.array, nf: int, p: int, mu: float, t: float, xi: float, Para: np.array, momshift: int) -> np.array:
    """Next-to-leading order evolution of the conformal moments (Evolved moment method)

    Args:
        j: conformal spin j (conformal spin is actually j+2 but anyway): array (N,)
        nf: number of effective fermions; scalars
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et); scalars
        mu: factorization evolution scale; scalar
        t: momentum transfer squared
        xi: skewness
        Para: parameters of the conformal moments
        momshift: shift of moments
        
        Note: xi,t are need when for the evolution of moments:
            We need Para for the parameters rather than just the unevolved moment ConFlav;
            Because the off-diagonal terms requires shift of moments.
            
        momshift (CAUTION!):
            When we shift the moment by j -> j+2 for xi^2 terms, only the evolution kernel E_{jk} should be shifted into E_{j+2,k+2}, whereas terms like xi^{-j-1} should not be shifted
            therefore, we should write the expression as E_{j,k+momshift} and x^{-j+momshift-1} to compensate/cancel the shift of j when j->j+2 is performed (momshift=2 in this case)
            Only relevant in the off-diagonal part

    Returns:
        Evolved conformal moment in evolution basis (qVal, q_du_plus, q_du_minus, qSigma, g);
        To be combined with the corresponding Wilson coefficient (for TFF/CFF) or conformal wave function (for GPD)
        return shape (N, 5)
        
    | The j here accept array input and preferrably just 1D for the contour integral in j. 
    | Therefore, j has shape (N,) where N is the interpolating order of the fixed quad.
    | Other quantities must be broadcastable with j and thus they should be preferrably scalar
    """
    
    assert j.ndim == 1, "Check dimension of j, must be 1D array" # shape (N,)
    
    ConfFlav     = Moment_Sum(j - momshift, t, Para)
    # Transform the unevolved moments to evolution basis
    ConfEvoBasis = np.einsum('ij, ...j->...i', flav_trans, ConfFlav) # shape (N, 5)
    # Taking the non-singlet and singlet parts of the conformal moments
    ConfNS = ConfEvoBasis[..., :3] # (N, 3)
    ConfSG = ConfEvoBasis[..., -2:] # (N, 2)
   
    # Set up evolution operator for WCs
    Alphafact = np.array(AlphaS(nloop_alphaS, nf, mu)) / np.pi / 2
    R = AlphaS(nloop_alphaS, nf, mu)/AlphaS(nloop_alphaS, nf, Init_Scale_Q) # shape N
    R = np.array(R)
  
    b0 = beta0(nf)
    lam, pr = projectors(j+1, nf, p)
    pproj = amuindep(j, nf, p)
    
    rmu1 = rmudep(nf, lam, lam, mu)

    Rfact = R[...,np.newaxis]**(-lam/b0)  # LO evolution (alpha(mu)/alpha(mu0))^(-gamma/beta0)

    # S/G LO evolution operator
    evola0 = np.einsum('...aij,...a->...ij', pr, Rfact)
    
    # NS LO evolution operator
    gam0NS = non_singlet_LO(j+1, nf, p)
    evola0NS = R**(-gam0NS/b0)
    
    # LO evolved moments
    confNS_ev0 = np.einsum('...,...i->...i',evola0NS, ConfNS)
    confSG_ev0 = np.einsum('...ij,...j->...i',evola0,ConfSG) 
    
    # S/G diagonal NLO evolution operator    
    evola1_diag_ab = np.einsum('kab,kabij->kabij', rmu1, pproj)
    evola1_diag = np.einsum('...abij,...b,...->...ij', evola1_diag_ab, Rfact,Alphafact)
    
    # NLO NS evolutioon set to zero
    confNS_ev1 = np.zeros_like(confNS_ev0)
    # S/G diagonal NLO evolution operator    
    confSG_ev1_diag = np.einsum('...ij,...j->...i',evola1_diag,ConfSG)
    
    # Following are the second integral resumming the off diagonal pieces, note that (j,k) meshgrid is used for vectorized j and k input. Check the paper for expression
    reK = -0.6
    Max_imK = 150
        
    def non_diag_integrand_mesh(k):
        
        jmesh, kmesh = np.meshgrid(j,k)
        meshshape=jmesh.shape
        jmesh=jmesh.reshape(-1)
        kmesh=kmesh.reshape(-1)
        
        ConfFlavk     = Moment_Sum(kmesh+1, t, Para)
        ConfEvoBasisk = np.einsum('ij, ...j->...i', flav_trans, ConfFlavk)

        ConfSGk = ConfEvoBasisk[:,-2:]
        ConfFlav_shift = Moment_Sum(jmesh+kmesh-momshift, t, Para)
        
        ConfEvoBasis_shift = np.einsum('ij, ...j->...i', flav_trans, ConfFlav_shift) 
        ConfSG_shift = ConfEvoBasis_shift[:,-2:]
        
        Bjk=np.array(bmudep(mu, np.array(jmesh), np.array(kmesh+momshift+1), nf,p))*Alphafact
        Bjk_shift= np.array(bmudep(mu, np.array(jmesh), np.array(jmesh+kmesh), nf,p))*Alphafact
        
        CBjk = np.einsum('b,bij,bj->bi',xi**(-kmesh),Bjk_shift,ConfSG_shift)
        CBjk_shift= np.einsum('b,bij,bj->bi',xi**(jmesh-momshift-kmesh-1),Bjk,ConfSGk)
            
        if (p == 1):
            out = np.einsum('ij,i->ij',CBjk-CBjk_shift,  1/4/np.tan(np.pi * kmesh / 2))
        elif (p == -1):
            out = np.einsum('ij,i->ij',CBjk,  1/4/np.tan(np.pi * kmesh / 2))+np.einsum('ij,i->ij',CBjk_shift,  1/4*np.tan(np.pi * kmesh / 2))
        
        outorishape=out.shape
        out=out.reshape(meshshape[0],meshshape[1],*outorishape[1:])
        return out        
        
    intd_non_diag_ev1_vec=fixed_quadvec(lambda imK:non_diag_integrand_mesh(reK+1j*imK)+non_diag_integrand_mesh(reK-1j*imK),0,Max_imK,200)
    
    # Off-diagonal piece for the NS evolution
    confSG_ev1_non_diag = intd_non_diag_ev1_vec
    # Combine the diagonal and off-diagonal pieces
    confSG_ev1 = confSG_ev1_diag + confSG_ev1_non_diag     
    
    # LO plus NLO evolution    
    conf_ev_NS_tot = confNS_ev0 + confNS_ev1
    conf_ev_SG_tot = confSG_ev0 + confSG_ev1

    return conf_ev_NS_tot, conf_ev_SG_tot, confSG_ev0

def TFF_Evo_NLO_evMOM(j: np.array, nf: int, p: int, Q: float, t: float, xi: float, Para: np.array, momshift: int, meson: int, muf: float) -> np.array:
    """Next-to-leading order evolved conformal moments combined with the DVMP Wilson coefficients in the evolution space (Evolved moment method)
    
    Args:
        j: conformal spin j (conformal spin is actually j+2 but anyway): array (N,)
        nf: number of effective fermions; 
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et): scalar
        Q: final evolution scale: scalar
        t: momentum transfer squared
        xi: skewness
        ConfFlav: unevolved conformal moments
        Para: parameters of the conformal moments
        momshift: shift of moments
        muset: offset in the scale mu to study the scale dependence of the TFF
        More notes in Moment_Evo_NLO() function above

    Returns:
        Next-to-leading order evolved conformal moments combined with the DVMP Wilson coefficients in the evolution basis (qVal, q_du_plus, q_du_minus, qSigma, g);
        return shape (N, 5)
    
    | The j here accept array input and preferrably just 1D for the contour integral in j. 
    | Therefore, j has shape (N,) where N is the interpolating order of the fixed quad.
    | Other quantities must be broadcastable with j and thus they should be preferrably scalar
    """
    
    assert j.ndim == 1, "Check dimension of j, must be 1D array" # shape (N,)
    
    # Retrive the evolved moment in the evolution basis
    conf_ev_NS_tot, conf_ev_SG_tot, confSG_ev0 = Moment_Evo_NLO(j, nf, p, muf, t, xi, Para, momshift)
    
    # Leading order non-singlet DVMP Wilson coefficient
    NSCoef_0 = WilsonCoef_DVMP_LO(j,nf, meson)[:3]
    # Leading order singlet DVMP Wilson coefficient
    SCoef_0 = WilsonCoef_DVMP_LO(j,nf, meson)[-2:]
    # Next-to-leading order singlet DVMP Wilson coefficient
    SCoef_1 = WilsonCoef_DVMP_NLO(j,0,nf,Q,muf, meson)[-2:]
    
    # Combing the LO Wilson coefficients with corresponding evolved moment
    NS_full = np.einsum('i...,...i->...i',NSCoef_0,conf_ev_NS_tot)    
    Sev1_full = np.einsum('i...,...i->...i',SCoef_0,conf_ev_SG_tot)    
    # NLO Wilson coefficient with LO evolved moment
    Sev0_full = np.einsum('i...,...i->...i',SCoef_1,confSG_ev0)    
    # Combine two singlet terms
    S_full = Sev1_full + Sev0_full
    
    # Recombing the non-singlet and singlet parts
    EvoConf = np.concatenate((NS_full, S_full), axis=-1) # (N, 5)    

    return EvoConf

'''***********************************************'''

def GPD_Moment_Evo_NLO(j: np.array, nf: int, p: int, mu: float, t: float, xi: complex, Para: np.array, momshift: int) -> np.array:
    """Next-to-leading order evolved conformal moments in the evolution basis (Evolved moment method)
    
    Args:
        j: conformal spin j (conformal spin is actually j+2 but anyway): array (N,)
        nf: number of effective fermions; 
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et): scalar
        Q: final evolution scale: scalar
        t: momentum transfer squared
        xi: skewness
        ConfFlav: unevolved conformal moments
        Para: parameters of the conformal moments
        momshift: shift of moments
        muset: offset in the scale mu to study the scale dependence of the TFF
        More notes in Moment_Evo_NLO() function above

    Returns:
        Next-to-leading order evolved conformal moments in the evolution basis (to be combined with conformal wave function)
        return shape (N, 5)
    
    | The j here accept array input and preferrably just 1D for the contour integral in j. 
    | Therefore, j has shape (N,) where N is the interpolating order of the fixed quad.
    | Other quantities must be broadcastable with j and thus they should be preferrably scalar
    """
    
    assert j.ndim == 1, "Check dimension of j, must be 1D array" # shape (N,)
    
    # Retrive the evolved moment in evolution basis (the last term redundant here)
    conf_ev_NS_tot, conf_ev_SG_tot, confSG_ev0 = Moment_Evo_NLO(j, nf, p, mu, t, xi, Para, momshift)
    
    # Recombing the non-singlet and singlet parts
    EvoConf = np.concatenate((conf_ev_NS_tot, conf_ev_SG_tot), axis=-1) # (N, 5)

    return EvoConf

def tPDF_Moment_Evo_NLO(j: np.array, nf: int, p: int, mu: float, ConfFlav: np.array) -> np.array:
    """FORWARD Next-to-leading order evolved conformal moments in the evolution basis (Evolved moment method)    
    
    This function removes the off-diagonal pieces in Moment_Evo_NLO()
    
    Args:
        j: conformal spin j (conformal spin is actually j+2 but anyway): array (N,)
        nf: number of effective fermions; 
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et): scalar
        mu: final evolution scale: scalar
        ConfFlav: unevolved conformal moments

    Returns:
        Next-to-leading order evolved conformal moments in the evolution basis in the forward limit (to be combined with inverse Mellin transform wave function)
        return shape (N, 5)
        
    | The j here accept array input and preferrably just 1D for the contour integral in j. 
    | Therefore, j has shape (N,) where N is the interpolating order of the fixed quad.
    | Other quantities must be broadcastable with j and thus they should be preferrably scalar
    """
    
    assert j.ndim == 1, "Check dimension of j, must be 1D array" # shape (N,)
    
    # Transform the unevolved moments to evolution basis
    ConfEvoBasis = np.einsum('ij, ...j->...i', flav_trans, ConfFlav) # shape (N, 5)
    # Taking the non-singlet and singlet parts of the conformal moments
    ConfNS = ConfEvoBasis[..., :3] # (N, 3)
    ConfSG = ConfEvoBasis[..., -2:] # (N, 2)
   
    # Set up evolution operator for WCs
    Alphafact = np.array(AlphaS(nloop_alphaS, nf, mu)) / np.pi / 2
    R = AlphaS(nloop_alphaS, nf, mu)/AlphaS(nloop_alphaS, nf, Init_Scale_Q) # shape N
    R = np.array(R)
  
    b0 = beta0(nf)
    lam, pr = projectors(j+1, nf, p)
    pproj = amuindep(j, nf, p)
    
    rmu1 = rmudep(nf, lam, lam, mu)

    Rfact = R[...,np.newaxis]**(-lam/b0)  # LO evolution (alpha(mu)/alpha(mu0))^(-gamma/beta0)

    # S/G LO evolution operator
    evola0 = np.einsum('...aij,...a->...ij', pr, Rfact)
    
    # NS LO evolution operator
    gam0NS = non_singlet_LO(j+1, nf, p)
    evola0NS = R**(-gam0NS/b0)
    
    # LO evolved moments
    confNS_ev0 = np.einsum('...,...i->...i',evola0NS, ConfNS)
    confSG_ev0 = np.einsum('...ij,...j->...i',evola0,ConfSG) 
    
    # S/G diagonal NLO evolution operator    
    evola1_diag_ab = np.einsum('kab,kabij->kabij', rmu1, pproj)
    evola1_diag = np.einsum('...abij,...b,...->...ij', evola1_diag_ab, Rfact,Alphafact)
    
    # NLO NS evolutioon set to zero
    confNS_ev1 = np.zeros_like(confNS_ev0)
    # S/G diagonal NLO evolution operator    
    confSG_ev1_diag = np.einsum('...ij,...j->...i',evola1_diag,ConfSG)
    
    # Combine the diagonal and off-diagonal pieces
    confSG_ev1 = confSG_ev1_diag
    
    # LO plus NLO evolution    
    conf_ev_NS_tot = confNS_ev0 + confNS_ev1
    conf_ev_SG_tot = confSG_ev0 + confSG_ev1

    # Recombing the non-singlet and singlet parts
    EvoConf = np.concatenate((conf_ev_NS_tot, conf_ev_SG_tot), axis=-1) # (N, 5)    

    return EvoConf