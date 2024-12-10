### Module for calculating meson production cross-sections and TFFs

import numpy as np
import Evolution as ev
import Observables as obs

from numpy import cos as Cos
from numpy import sin as Sin
from numpy import real as Real
from numpy import imag as Imag
from numpy import conjugate as Conjugate
from scipy.integrate import quad

from numba import njit, vectorize

"""

********************************Masses, decay constants, etc.***********************

"""

M_p = 0.938
M_n = 0.940
M_rho = 0.775
M_phi = 1.019
M_jpsi = 3.097
gevtonb = 389.9 * 1000
alphaEM = 1 / 137.036




"""

******************************Cross-sections for proton target (currently for virtual photon scattering sub-process)*********************************

"""


f_rho= 0.209 
f_phi = 0.221 # Change to 0.233
f_jpsi = 0.406



def f_VL(meson:int):
    if (meson==1):
        return f_rho
    if (meson==2):
         return f_phi 
    if (meson==3):
         return f_jpsi 


def epsilon(y:float):
    """ Photon polarizability.

    Args:
       y (float): Beam energy lost parameter
     

    Returns:
        epsilon:  "Eq.(31) in https://arxiv.org/pdf/1112.2597" 
    """
    return (1 - y) / (1 - y + y**2 / 2)
   #Is epsilont different for DVCS and DVMP?


def R(Q:float, meson:int):
    """ The ratio of longitudinal to transverse cross-sections.

    Args:
        Q: The photon virtuality 
        meson:The meson being produced in DVMP process: 1 for rho, 2 for phi, 3 for j/psi
     

    Returns: The parametrization of R factor and the fitted (a,p) parameters in  L/T separation  as in  Eq.(32) in https://arxiv.org/pdf/1112.2597"
       
    """
       
   
    
    def M_meson(meson:int):
        """ Meson mass.

        Args:
            meson:The meson being produced in DVMP process: 1 for rho, 2 for phi, 3 for j/psi
         

        Returns:
           Meson mass: 1 for rho, 2 for phi, 3 for j/psi
        """
        if (meson==1):
            return M_rho
        if (meson==2):
             return M_phi
        if (meson==3):
             return M_jpsi
         
    def a_meson (meson:int):
        """ Parameter a in the parametrization of R

        Args:
            meson:The meson being produced in DVMP process: 1 for rho, 2 for phi, 3 for j/psi
         

        Returns:
          a as in  Eq.(32) in https://arxiv.org/pdf/1112.2597"
        """
        if (meson==1):
             return 2.2
        if (meson==2): 
             return 25.4
        if (meson==3):
             return 0
    
    def p_meson (meson:int):
        """ Parameter p in the parametrization of R

        Args:
            meson:The meson being produced in DVMP process: 1 for rho, 2 for phi, 3 for j/psi
         

        Returns:
          p as in  Eq.(32) in https://arxiv.org/pdf/1112.2597"
        """
        if (meson==1):
            return 0.451
        if (meson==2): 
            return 0.180
        if (meson==3):
            return 0
   
    return  (Q**2/M_meson(meson)**2)/(1+a_meson(meson)*Q**2/M_meson(meson)**2)**p_meson(meson)
    
def MassCorr(meson:int):
    """ Mass corrections 

     Args:
         meson:The meson being produced in DVMP process: 1 for rho, 2 for phi, 3 for j/psi
      

     Returns:
        mass correction only for j/psi
     """
  
    if (meson==3):
        return  M_jpsi
    else:
        return 0
  
         

    
    #return Q**2/M_meson(meson)**2 /(1 + a_meson(meson) * Q**2 / M_meson(meson)**2)**p_meson(meson)

@np.vectorize
def dsigmaL_dt(y: float, xB: float, t: float, Q: float, meson:int, HTFF: complex, ETFF: complex):
    """Longitudinal DVMP cross section differential only in t
          
      Args:
          y (float): Beam energy lost parameter
          xB (float): x_bjorken
          t (float): momentum transfer square
          Q (float): photon virtuality
          TFF (complex): Transition form factor H 
          ETFF (complex): Transition form factor E
          MassCorr(int): Mass corrections to the cross section.  Nonzero only for j/psi
   
      

      Returns:
          
          Eq.(2.8) as in https://arxiv.org/pdf/2409.17231"
    """

    return  ( 4* np.pi**2  *alphaEM * xB ** 2 / ((Q**2 + MassCorr(meson)**2) ** 2))* (Q/(Q**2+ MassCorr(meson)**2))**2 *(Real(HTFF* Conjugate(HTFF)) - t/4/ M_p**2 * Real(ETFF* Conjugate(ETFF)))

                                         
                                                
        
   # return  gevtonb *2*np.pi**2 * alphaEM * (xB ** 2 / (1 - xB) * Q**4 * np.sqrt(1 + eps(xB, Q) ** 2))* Cunp(xB, t, Q, HTFF_rho, ETFF_rho, meson)

@np.vectorize
def dsigma_dt(y: float, xB: float, t: float, Q: float, meson:int, HTFF: complex, ETFF: complex):
    """The total DVMP cross section differential only in t
          
      Args:
          y (float): Beam energy lost parameter
          xB (float): x_bjorken
          t (float): momentum transfer square
          Q (float): photon virtuality
          TFF (complex): Transition form factor H 
          ETFF (complex): Transition form factor E
          MassCorr(int): Mass corrections to the cross section.  Nonzero only for j/psi
   
      

      Returns:
          
          Eq.(2.16) as in https://arxiv.org/pdf/2409.17231"
    """

    return  gevtonb * dsigmaL_dt(y, xB, t, Q, meson, HTFF, ETFF)*(epsilon(y)+1/R(Q,meson))
 





















