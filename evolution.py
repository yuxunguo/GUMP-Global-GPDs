"""
Credits to the Gepard program by K. Kumericki 

GPD evolution operator mathcal{E}.

Returns:
   Numpy array evol[p, k, j]  where flavor index j takes values
   in the evolution basis:
   (1) -- singlet quark
   (2) -- gluon
   (3) -- NS(+)
   (4) -- NS(-)  (not tested!)

Notes:
   GPD models may be defined in different basis and should
   provide appropriate transformation matrix

Todo:
    * Array indices ordering is a bit of a mess

"""

from typing import Tuple

import numpy as np

import adim, constants, evolution


def lambdaf(gam0) -> np.ndarray:
    """Eigenvalues of the LO singlet anomalous dimensions matrix.

    Args:
          gam0: matrix of LO anomalous dimensions

    Returns:
        lam[a, k]
        a in [+, -] and k is MB contour point index

    """
    # To avoid crossing of the square root cut on the
    # negative real axis we use trick by Dieter Mueller
    aux = ((gam0[..., 0, 0] - gam0[..., 1, 1]) *
           np.sqrt(1. + 4.0 * gam0[..., 0, 1] * gam0[..., 1, 0] /
                   (gam0[..., 0, 0] - gam0[..., 1, 1])**2))
    lam1 = 0.5 * (gam0[..., 0, 0] + gam0[..., 1, 1] - aux)
    lam2 = lam1 + aux
    return np.stack([lam1, lam2])


def projectors(gam0) -> Tuple[np.ndarray, np.ndarray]:
    """Projectors on evolution quark-gluon singlet eigenaxes.

    Args:
          gam0: LO anomalous dimension

    Returns:
         lam: eigenvalues of LO an. dimm matrix lam[a, k]  # Eq. (123)
          pr: Projector pr[k, a, i, j]  # Eq. (122)
               k is MB contour point index
               a in [+, -]
               i,j in {Q, G}

    """
    lam = lambdaf(gam0)
    den = 1. / (lam[0, ...] - lam[1, ...])

    # P+ and P-
    ssm = gam0 - np.einsum('...,ij->...ij', lam[1, ...], np.identity(2))
    ssp = gam0 - np.einsum('...,ij->...ij', lam[0, ...], np.identity(2))
    prp = np.einsum('...,...ij->...ij', den, ssm)
    prm = np.einsum('...,...ij->...ij', -den, ssp)
    # We insert a-axis before i,j-axes, i.e. on -3rd place
    pr = np.stack([prp, prm], axis=-3)
    return lam, pr
