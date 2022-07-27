"""
Credits to the Gepard program by K. Kumericki 

QCD anomalous dimensions.

Note:
    Functions in this module have as first argument Mellin moment
    n=j+1, where j is conformal moment used everywhere else.
    Thus, they should always be called as f(j+1, ...).

"""
# from cmath import exp
# from scipy.special import loggamma as clngamma

import numpy as np
from typing import Union

from constants import CA, CF, CG, TF

from scipy.special import psi

def S1(z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Harmonic sum S_1."""
    return np.euler_gamma + psi(z+1)

def non_singlet_LO(n: complex, nf: int, prty: int = 1) -> complex:
    """Non-singlet LO anomalous dimension.

    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        prty (int): 1 for NS^{+}, -1 for NS^{-}, irrelevant at LO

    Returns:
        Non-singlet LO anomalous dimension.

    """
    return CF*(-3.0-2.0/(n*(1.0+n))+4.0*S1(n))


def singlet_LO(n: complex, nf: int, prty: int = 1) -> np.ndarray:
    """Singlet LO anomalous dimensions.

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

