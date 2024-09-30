"""
In this module we have the DVCS cross-section formulas with twist-two Compton form factors based on our work https://inspirehep.net/literature/1925449

The five fold cross-section dsigma over dxB dt dQ dphi dphi_S will be calculated.

"""
from scipy.integrate import quad_vec
#The proton mass M = 0.938 GeV
M = 0.938

#The fine structure constant 
alphaEM = 1 / 137.036

#Conversion factor from GeV to nb for the cross-section
conv = 389.9 * 1000

import numpy as np
from numpy import cos as Cos
from numpy import sin as Sin
from numpy import sqrt as Sqrt
from numpy import pi as Pi
from numpy import real as Real
from numpy import imag as Imag
from numpy import conjugate as Conjugate
import warnings

from numba import njit, vectorize

#warnings.filterwarnings(action="error", category=np.ComplexWarning)

#The prefactor for cross-section given by some kinematics
@njit 
def Gamma_prefac(y: float, xB: float, Q: float):
    """Kinematic prefactor in the cross-sections 
    
    Check e.g. https://inspirehep.net/literature/1925449

    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        Q (float): photon virtuality

    Returns:
        Gamma: the prefactor as defined in the ref.
    """
    return (alphaEM**3*xB*y**2)/(16.*Pi**2*Q**4*Sqrt(1 + (4*M**2*xB**2)/Q**2))

@njit
def xi(xB: float, t: float, Q: float):
    """Skewness parameter xi that can be expressed with the kinematics

    Args:
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality

    Returns:
        xi (float): skewness parameter
    """
    return (1/(2 - xB) - (2*t*(-1 + xB))/(Q**2*(-2 + xB)**2))*xB

@njit
def N(xB: float, t: float, Q: float):
    """Numeric factor, essentially the transverse momentum transfer squared

    Args:
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality

    Returns:
        N (float): numeric factor
    """
    return Sqrt(- 4 * M ** 2 * xi(xB,t,Q) ** 2 - t * (1 - xi(xB,t,Q) ** 2)) / M

@njit
def h_pureDVCS(y: float, xB: float, t: float, Q: float, phi: float):
    """Kinematical variable for pure DVCS cross-sections
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked
    
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        h (float): kinematical variables for pure DVCS
    """
    return -0.5*(Q**4*(Q**2 + t*(-1 + 2*xB))*(-2 + y))/((Q**2 + t)*(Q**2 + 4*M**2*xB**2)*y) - (2*Q**3*(Q**2 + t*xB)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))/(M*(Q**2 + t)*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*y) + (Q**4*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))**2)/(M**2*(Q**2 + t)**2*xB**2*(Q**2 + 4*M**2*xB**2)**3*y**2)

@njit
def hminus_pureDVCS(y: float, xB: float, t: float, Q: float, phi: float):
    """ Kinematical variable for pure DVCS cross-sections
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked
        
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        h (float): kinematical variables for pure DVCS 
    """ 
    return (Q**4*(Q**2 + t*(-1 + 2*xB))*(-2 + y))/(2.*Sqrt((Q**2 + t)**2)*(Q**2 + 4*M**2*xB**2)*y) + (2*Q**3*(Q**2 + t*xB)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))/(M*Sqrt((Q**2 + t)**2)*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*y)

@njit
def A_BH(y: float, xB: float, t: float, Q: float, phi: float):
    """ Kinematical variable for BH cross-section
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked
        
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        A_BH (float): kinematical variables for BH cross-section
    """ 
    return (-8*M*(Q**2 + 4*M**2*xB**2)*(M*(8*M**6*t**2*xB**6*y**2 + 4*M**4*Q**2*t*xB**4*(t + 2*t*xB - 4*M**2*xB**2)*y**2 - Q**8*(t*(-1 + xB) - M**2*xB**2)*(2 - 2*y + y**2) + M**2*Q**4*xB**2*(8*M**4*xB**4*y**2 - 8*M**2*t*xB**2*(-4 + 4*y + (-1 + xB)*y**2) + t**2*(2 - 2*xB**2*(-2 + y)**2 - 2*y + y**2 + 4*xB*y**2)) + Q**6*xB*(4*M**4*xB**3*y**2 + t**2*(2 - xB*(-2 + y)**2 - 2*y + y**2) - 2*M**2*t*xB*(-6 + 6*y + (-3 + 2*xB)*y**2))) + 2*Q**5*(Q**2 + t*xB)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(-2*M**2*t*xB - Q**2*(t - 2*M**2*xB))*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*(-2 + y)*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi) + 8*M**3*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2))*(Q**2*(-1 + y) + M**2*xB**2*y**2)*Cos(phi)**2))/(Q**2*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

@njit
def B_BH(y: float, xB: float, t: float, Q: float, phi: float):
    """ Kinematical variable for BH cross-section
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked
        
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        B_BH (float): kinematical variables for BH cross-section
    """ 
    return (-4*M*t*xB*(Q**2 + 4*M**2*xB**2)*(M*xB*(8*M**4*t**2*xB**4*y**2 + 4*M**2*Q**2*t*xB**2*(t - 4*M**2*xB**2)*y**2 + Q**8*(2 - 2*y + y**2) + 2*Q**6*(2*M**2*xB**2*y**2 + t*(-2 + xB*(-2 + y)**2 + 2*y - y**2)) + Q**4*(-8*M**2*t*xB**2*y**2 + 8*M**4*xB**4*y**2 + t**2*(2 - 2*xB*(-2 + y)**2 + 2*xB**2*(-2 + y)**2 - 2*y + y**2))) + 4*Q**5*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + t**2*xB*(-1 + 2*xB) + Q**2*t*(-1 + 3*xB))*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*(-2 + y)*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi) + 8*M*xB*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2))*(Q**2*(-1 + y) + M**2*xB**2*y**2)*Cos(phi)**2))/(Q**2*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

@njit
def At_BH_L(y: float, xB: float, t: float, Q: float, phi: float):
    """ Kinematical variable for polarized BH cross-section
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked
        
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        At_BH (float): kinematical variables for polarized BH cross-section
    """ 
    return (-4*M*(Q**2 + 4*M**2*xB**2)**2*y*(-(M*xB*(2*Q**2*t**3*(-1 + xB)*xB*(t*(-1 + xB)**2 + M**2*(4 - 3*xB)*xB) - (4*M**2 - t)*t**4*xB**2*(1 - 3*xB + 2*xB**2) - 2*Q**8*(-2 + xB)*(t - t*xB + M**2*xB**2) + 2*Q**6*t*xB*(-(M**2*xB*(-4 + xB + xB**2)) + t*(3 - 5*xB + 2*xB**2)) + Q**4*t**2*(2*M**2*(5 - 4*xB)*xB**3 + t*(-4 + 10*xB - 7*xB**2 - xB**3 + 2*xB**4)))*(-2 + y)) + 2*Q*(Q**2 + t*xB)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*(2*(4*M**2 - t)*t**3*(-1 + xB)*xB**2 - 2*Q**2*t**2*(-2 + xB)*xB*(t*(-1 + xB) - 2*M**2*xB) + Q**6*(-2 + xB)*(t*(2 - 3*xB) + 4*M**2*xB**2) + Q**4*t*(8*M**2*(-1 + xB)*xB**2 + t*(-4 + 4*xB + xB**2 - 2*xB**3)))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))/((Q**2 + t)*(Q**2*(-2 + xB) + t*(-1 + xB)*xB)*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

@njit
def Bt_BH_L(y: float, xB: float, t: float, Q: float, phi: float):
    """ Kinematical variable for polarized BH cross-section
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked
        
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        Bt_BH (float): kinematical variables for polarized BH cross-section
    """ 
    return (4*M*(Q**2 + 4*M**2*xB**2)**2*y*(-(M*xB*(Q**8*(-2 + xB)*(t*(-2 + xB) - 2*M**2*xB**2) - t**4*(-1 + xB)*xB**2*(4*M**2*(-1 + xB) + t*(-1 + 2*xB)) + Q**6*t*xB*(2*M**2*xB*(4 + xB - 2*xB**2) + t*(8 - 10*xB + 3*xB**2)) - Q**2*t**3*xB*(2*M**2*xB*(4 - 5*xB + 2*xB**2) + t*(4 - 10*xB + 5*xB**2)) + Q**4*t**2*(2*M**2*(7 - 6*xB)*xB**3 + t*(-4 + 8*xB - 5*xB**3 + 2*xB**4)))*(-2 + y)) + 4*Q*(Q**2 + t*xB)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*(t**3*(4*M**2 + t)*(-1 + xB)*xB**2 + Q**6*(-2 + xB)*(t - t*xB + 2*M**2*xB**2) + Q**2*t**2*xB*(2*M**2*(-2 + xB)*xB + t*(-3 + 2*xB)) - Q**4*t*(-4*M**2*(-1 + xB)*xB**2 + t*(2 - 2*xB**2 + xB**3)))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))/((Q**2 + t)*(Q**2*(-2 + xB) + t*(-1 + xB)*xB)*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

@njit
def At_BH_Tin(y: float, xB: float, t: float, Q: float, phi: float):
    """Kinematical variable for polarized BH cross-section
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked  
        
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        At_BH_Tin (float): kinematical variables for polarized BH cross-section
    """ 
    return (-2*(Q**2 + 4*M**2*xB**2)**2*((4*M**2 - t)*t*(-1 + xB)*xB + Q**2*(-2 + xB)*(-t + 2*M**2*xB))*y*(-(M*xB*(2*Q**2*t**3*(-1 + xB)*xB*(t*(-1 + xB)**2 + M**2*(4 - 3*xB)*xB) - (4*M**2 - t)*t**4*xB**2*(1 - 3*xB + 2*xB**2) - 2*Q**8*(-2 + xB)*(t - t*xB + M**2*xB**2) + 2*Q**6*t*xB*(-(M**2*xB*(-4 + xB + xB**2)) + t*(3 - 5*xB + 2*xB**2)) + Q**4*t**2*(2*M**2*(5 - 4*xB)*xB**3 + t*(-4 + 10*xB - 7*xB**2 - xB**3 + 2*xB**4)))*(-2 + y)) + 2*Q*(Q**2 + t*xB)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*(2*(4*M**2 - t)*t**3*(-1 + xB)*xB**2 - 2*Q**2*t**2*(-2 + xB)*xB*(t*(-1 + xB) - 2*M**2*xB) + Q**6*(-2 + xB)*(t*(2 - 3*xB) + 4*M**2*xB**2) + Q**4*t*(8*M**2*(-1 + xB)*xB**2 + t*(-4 + 4*xB + xB**2 - 2*xB**3)))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))/(Q**2*(Q**2 + t)*(-2 + xB)**2*(Q**2*(-2 + xB) + t*(-1 + xB)*xB)*Sqrt(-(((4*M**2 - t)*t**2*(-1 + xB)**2*xB**2 + Q**2*(4*M**2 - t)*t*xB**2*(2 - 3*xB + xB**2) + Q**4*(-2 + xB)**2*(t - t*xB + M**2*xB**2))/(Q**4*(-2 + xB)**4)))*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

@njit
def Bt_BH_Tin(y: float, xB: float, t: float, Q: float, phi: float):
    """ Kinematical variable for polarized BH cross-section
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked
        
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        Bt_BH_Tin (float): kinematical variables for polarized BH cross-section
    """ 
    return (8*M**2*(Q**2*(-2 + xB) + 2*t*(-1 + xB))*xB*(Q**2 + 4*M**2*xB**2)**2*y*(-(M*xB*(-2*M**2*t**4*(-1 + xB)**2*xB**2 - Q**8*(-2 + xB)*(t - t*xB + M**2*xB**2) - Q**2*t**3*xB*(t*(-1 + xB)**2 + M**2*xB*(4 - 5*xB + 2*xB**2)) + Q**6*t*xB*(M**2*xB*(4 + xB - 2*xB**2) + t*(5 - 8*xB + 3*xB**2)) + Q**4*t**2*(M**2*(7 - 6*xB)*xB**3 + t*(-2 + 3*xB + xB**2 - 4*xB**3 + 2*xB**4)))*(-2 + y)) + 2*Q*(Q**2 + t*xB)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*(4*M**2*t**3*(-1 + xB)*xB**2 + Q**2*t**2*xB*(t*(-1 + xB) + 2*M**2*(-2 + xB)*xB) + Q**6*(-2 + xB)*(t - 2*t*xB + 2*M**2*xB**2) + Q**4*t*(4*M**2*(-1 + xB)*xB**2 + t*(-2 + 3*xB**2 - 2*xB**3)))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))/(Q**2*(Q**2 + t)*(-2 + xB)**2*(Q**2*(-2 + xB) + t*(-1 + xB)*xB)*Sqrt(-(((4*M**2 - t)*t**2*(-1 + xB)**2*xB**2 + Q**2*(4*M**2 - t)*t*xB**2*(2 - 3*xB + xB**2) + Q**4*(-2 + xB)**2*(t - t*xB + M**2*xB**2))/(Q**4*(-2 + xB)**4)))*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

@njit
def At_BH_Tout(y: float, xB: float, t: float, Q: float, phi: float):
    """ Kinematical variable for polarized BH cross-section

    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked
                
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        At_BH_Tout (float): kinematical variables for polarized BH cross-section
    """ 
    return (2*(Q**2 - t)*t*(-4*M**2 + t)*(Q**2*(-2 + xB) + 2*t*(-1 + xB))*xB*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)**3*Sqrt(1 - (1 + (2*M**2*(Q**2 + t)*xB**2)/(Q**4 + Q**2*t*xB))**2/(1 + (4*M**2*xB**2)/Q**2))*y*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Sin(phi))/(Q*(Q**2 + t)*(-2 + xB)**2*Sqrt(-(((4*M**2 - t)*t**2*(-1 + xB)**2*xB**2 + Q**2*(4*M**2 - t)*t*xB**2*(2 - 3*xB + xB**2) + Q**4*(-2 + xB)**2*(t - t*xB + M**2*xB**2))/(Q**4*(-2 + xB)**4)))*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

@njit
def Bt_BH_Tout(y: float, xB: float, t: float, Q: float, phi: float):
    """ Kinematical variable for polarized BH cross-section
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked
                
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        Bt_BH_Tout (float): kinematical variables for polarized BH cross-section
    """
    return (8*M**2*(Q**2 - t)*t*(Q**2*(-2 + xB) + 2*t*(-1 + xB))*xB*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)**3*Sqrt(1 - (1 + (2*M**2*(Q**2 + t)*xB**2)/(Q**4 + Q**2*t*xB))**2/(1 + (4*M**2*xB**2)/Q**2))*y*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Sin(phi))/(Q*(Q**2 + t)*(-2 + xB)**2*Sqrt(-(((4*M**2 - t)*t**2*(-1 + xB)**2*xB**2 + Q**2*(4*M**2 - t)*t*xB**2*(2 - 3*xB + xB**2) + Q**4*(-2 + xB)**2*(t - t*xB + M**2*xB**2))/(Q**4*(-2 + xB)**4)))*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

@njit
def A_INT_unp(y: float, xB: float, t: float, Q: float, phi: float):
    """ Kinematical variable for interference cross-section
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked
        
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        A_INT_unp (float): kinematical variables for interference cross-section
    """
    return (-64*M**2*xB**2*(Q**2 + 4*M**2*xB**2)**3*y**2*((Q**4*(Q**2 + t)*(Q**2*(-1 + y) + t*(-1 + xB*y)))/(16.*xB*y) - (Q**2*(2*t**2 + Q**2*t*(1 + 2*xB)*y + Q**4*(-2 + 3*y))*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))/(16.*M*xB**2*(Q**2 + 4*M**2*xB**2)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*y**2) + (Q**4*(t*(1 + y) + Q**2*(-1 + 2*y))*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))**2)/(8.*M**2*xB**3*(Q**2 + 4*M**2*xB**2)**3*y**3) - (Q**4*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))**3)/(8.*M**3*xB**4*(Q**2 + 4*M**2*xB**2)**4*Sqrt(1 + (4*M**2*xB**2)/Q**2)*y**3)))/(Q**2*(Q**2 + t)*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

@njit
def B_INT_unp(y: float, xB: float, t: float, Q: float, phi: float):
    """ Kinematical variable for interference cross-section
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked
        
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        B_INT_unp (float): kinematical variables for interference cross-section
    """
    return (8*Q*t*(Q**2*(-2 + xB) + 2*t*(-1 + xB))*(M*Q*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**2 + t*(-1 + 2*xB))*(-2 + y) + 4*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))/(M*(Q**2 + t)**2*(-2 + xB)**2*(Q**2 + 4*M**2*xB**2)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*y)

@njit
def C_INT_unp(y: float, xB: float, t: float, Q: float, phi: float):
    """ Kinematical variable for interference cross-section
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked  
        
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        C_INT_unp (float): kinematical variables for interference cross-section
    """
    return (-8*(Q**2*(-2 + xB) + 2*t*(-1 + xB))*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(-(M**2*t**2*xB**2) + Q**2*t*xB*(t*(-1 + xB) - 2*M**2*xB) + Q**4*(t*(-1 + xB) - M**2*xB**2))*(-2 + y) + Q*(Q**2 + 4*M**2*xB**2)*(Q**4 + t**2*xB*(-1 + 2*xB) + Q**2*t*(-1 + 3*xB))*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(M*xB*(-4*M**2*Q**2*t**2*xB**2*y**2 - 8*M**4*t**2*xB**4*y**2 + Q**8*(2 - 2*y + y**2) + Q**6*(t*(-1 + 2*xB)*(-2 + y)**2 + 4*M**2*xB**2*y**2) + 2*Q**4*(t**2*(1 - xB*(-2 + y)**2 + xB**2*(-2 + y)**2 - y) + 4*M**4*xB**4*y**2)) + 4*Q**5*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + t**2*xB*(-1 + 2*xB) + Q**2*t*(-1 + 3*xB))*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*(-2 + y)*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi) + 8*M*xB*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2))*(Q**2*(-1 + y) + M**2*xB**2*y**2)*Cos(phi)**2))/(Q**2*(Q**2 + t)*Sqrt((Q**2 + t)**2)*(-2 + xB)**2*Sqrt(1 + (4*M**2*xB**2)/Q**2)*y*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

@njit
def At_INT_unp(y: float, xB: float, t: float, Q: float, phi: float):
    """ Kinematical variable for interference cross-section with polarized beam
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked
        
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        At_INT_unp (float): kinematical variables for interference cross-section with polarized beam
    """
    return (32*M*Q*Sqrt(t*(-1 + t/(4.*M**2)))*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)**3*Sqrt(1 - (1 + (2*M**2*(Q**2 + t)*xB**2)/(Q**4 + Q**2*t*xB))**2/(1 + (4*M**2*xB**2)/Q**2))*y*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*((Q**2*(Q**2 + t))/4. - ((Q**2 + t)*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))/(2.*M*xB*(Q**2 + 4*M**2*xB**2)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*y) + (Q**2*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))**2)/(2.*M**2*xB**2*(Q**2 + 4*M**2*xB**2)**3*y**2))*Sin(phi))/(Sqrt((Q**2 + t)**2)*Sqrt(t*(-4 + t/M**2))*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

@njit
def Bt_INT_unp(y: float, xB: float, t: float, Q: float, phi: float):
    """ Kinematical variable for interference cross-section with polarized beam
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked
        
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        Bt_INT_unp (float): kinematical variables for interference cross-section with polarized beam
    """
    return 0

@njit
def Ct_INT_unp(y: float, xB: float, t: float, Q: float, phi: float):
    """ Kinematical variable for interference cross-section with polarized beam
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked
        
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        Ct_INT_unp (float): kinematical variables for interference cross-section with polarized beam
    """
    return (-64*M*Sqrt(t*(-1 + t/(4.*M**2)))*(Q**2*(-2 + xB) + 2*t*(-1 + xB))*xB*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)**3*Sqrt(1 - (1 + (2*M**2*(Q**2 + t)*xB**2)/(Q**4 + Q**2*t*xB))**2/(1 + (4*M**2*xB**2)/Q**2))*y*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*(-0.125*(Q**2*(Q**2 + t)**2) + ((Q**2 - t)*(Q**2 + t)*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))/(4.*M*xB*(Q**2 + 4*M**2*xB**2)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*y) - (Q**2*(Q**2 - t)*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))**2)/(4.*M**2*xB**2*(Q**2 + 4*M**2*xB**2)**3*y**2))*Sin(phi))/(Q*(Q**2 + t)**2*Sqrt(t*(-4 + t/M**2))*(-2 + xB)**2*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

@njit
def A_INT_pol(y: float, xB: float, t: float, Q: float, phi: float):
    """ Kinematical variable for interference cross-section with polarized target
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked
        
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        A_INT_pol (float): kinematical variables for interference cross-section with polarized target
    """
    return (-4*Q**6*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*Sqrt(1 - (1 + (2*M**2*(Q**2 + t)*xB**2)/(Q**4 + Q**2*t*xB))**2/(1 + (4*M**2*xB**2)/Q**2))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*(M*Q*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**2 + t*(-1 + 2*xB))*(-2 + y) + 4*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*Sin(phi))/((Q**2 + t)*xB*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

@njit
def B_INT_pol(y: float, xB: float, t: float, Q: float, phi: float):
    """ Kinematical variable for interference cross-section with polarized target
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked
        
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        B_INT_pol (float): kinematical variables for interference cross-section with polarized target
    """
    return 0

@njit
def C_INT_pol(y: float, xB: float, t: float, Q: float, phi: float):
    """ Kinematical variable for interference cross-section with polarized target
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked
        
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        C_INT_pol (float): kinematical variables for interference cross-section with polarized target
    """
    return (4*Q**4*(Q**2*(-2 + xB) + 2*t*(-1 + xB))*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*Sqrt(1 - (1 + (2*M**2*(Q**2 + t)*xB**2)/(Q**4 + Q**2*t*xB))**2/(1 + (4*M**2*xB**2)/Q**2))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*(M*Q*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**2 + t*(-1 + 2*xB))*(-2 + y) + 4*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*Sin(phi))/(Sqrt((Q**2 + t)**2)*(-2 + xB)**2*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

@njit
def At_INT_pol(y: float, xB: float, t: float, Q: float, phi: float):
    """ Kinematical variable for interference cross-section with polarized target and target
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked
        
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        At_INT_pol (float): kinematical variables for interference cross-section with polarized target and target
    """
    return (64*M**2*xB**2*(Q**2 + 4*M**2*xB**2)**3*y**2*((Q**4*(Q**2 + t)*(Q**2*(-1 + y) + t*(-1 + xB*y)))/(16.*xB*y) - (Q**2*(Q**2 + t)*(2*t*xB*y + Q**2*(-2 + 3*y))*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))/(16.*M*xB**2*(Q**2 + 4*M**2*xB**2)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*y**2) + (Q**4*(Q**2 + t*xB)*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))**2)/(8.*M**2*xB**3*(Q**2 + 4*M**2*xB**2)**3*y**2)))/(Q**2*Sqrt((Q**2 + t)**2)*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

@njit
def Bt_INT_pol(y: float, xB: float, t: float, Q: float, phi: float):
    """ Kinematical variable for interference cross-section with polarized target and target
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked
        
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        Bt_INT_pol (float): kinematical variables for interference cross-section with polarized target and target
    """
    return (8*t*(Q**2*(-2 + xB) + 2*t*(-1 + xB))*xB)/(Sqrt((Q**2 + t)**2)*(-2 + xB)**2)

@njit
def Ct_INT_pol(y: float, xB: float, t: float, Q: float, phi: float):
    """ Kinematical variable for interference cross-section with polarized target and target
    
    | Defined in the appendix of https://inspirehep.net/literature/1925449 
    | Code converted from the Mathematica master code and numerically checked
        
    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): momentum transfer square
        Q (float): photon virtuality
        phi (float): azimuthal angel

    Returns:
        Ct_INT_pol (float): kinematical variables for interference cross-section with polarized target and target
    """
    return (8*Q**3*(Q**2*(-2 + xB) + 2*t*(-1 + xB))*(M*Q*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**2 + t*(-1 + 2*xB))*(-2 + y) + 4*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(-(M**2*t**2*xB**2) + Q**2*t*xB*(t*(-1 + xB) - 2*M**2*xB) + Q**4*(t*(-1 + xB) - M**2*xB**2))*(-2 + y) + Q*(Q**2 + 4*M**2*xB**2)*(Q**4 + t**2*xB*(-1 + 2*xB) + Q**2*t*(-1 + 3*xB))*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))/((Q**2 + t)**2*(-2 + xB)**2*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

@njit
def GE_FF(t):
    """ Electric form factors taken from global analysis
        
    Args:
        t (float): momentum transfer square

    Returns:
        GE (float): Electro-Magnetic form factors taken from global analysis https://inspirehep.net/literature/1613527
    """
    return (-9.14935193966684 + 54.294024034390276*Sqrt(0.0779191396 - t) + t*(136.5654233846795 - 467.9889798263139*Sqrt(0.0779191396 - t) + t*(-356.33976285348626 + 492.07500397303323*Sqrt(0.0779191396 - t) + t*(167.32689941569018 - 15.056820275484046*Sqrt(0.0779191396 - t) + t*(-0.8293426255316358 - 0.0034036166377127072*Sqrt(0.0779191396 - t) + (-0.0024470359051638347 + 0.0007055977957364234*Sqrt(0.0779191396 - t) - 0.00038999999999989043*t)*t)))))/(0.881997244666898 + Sqrt(0.0779191396 - t))**12

@njit
def GM_FF(t):
    """ Magentic form factors taken from global analysis
        
    Args:
        t (float): momentum transfer square

    Returns:
        GM (float): Magentic-Magnetic form factors taken from global analysis https://inspirehep.net/literature/1613527
    """
    return (62.80178446046987 - 164.88646954222185*Sqrt(0.0779191396 - t) + t*(-880.969678407686 + 1171.7507047011227*Sqrt(0.0779191396 - t) + t*(2260.679768631847 - 923.4532810241969*Sqrt(0.0779191396 - t) + t*(-495.99250268661956 + 68.849541922215*Sqrt(0.0779191396 - t) + t*(4.77087295630026 + 0.00785931565497526*Sqrt(0.0779191396 - t) + (0.0016567896964982563 - 0.0006355619225276535*Sqrt(0.0779191396 - t) - 0.000025276650001424538*t)*t)))))/(0.881997244666898 + Sqrt(0.0779191396 - t))**12

@njit
def F1_FF(t):
    """ Dirac form factors calculated with GE and GM
        
    Args:
        t (float): momentum transfer square

    Returns:
        F1 (float): Dirac form factors
    """
    return (GE_FF(t) - t/ (4 * M ** 2) * GM_FF(t))/(1 -  t/ (4 * M ** 2))

@njit
def F2_FF(t):
    """ Pauli form factors calculated with GE and GM
        
    Args:
        t (float): momentum transfer square

    Returns:
        F2 (float): Pauli form factors
    """
    return (-GE_FF(t) + GM_FF(t))/(1 -  t/ (4 * M ** 2))

####################################################################################################################

########################### The functions below are the definition of differential cross sections
# I had to change the order of the functions to make numba work

# The total cross-section is given by the sum of Bethe-Heitler (BH), pure DVCS and interference (INT) contributions
def dsigma_TOT(y: float, xB: float, t: float, Q: float, phi: float, pol, HCFF: complex, ECFF: complex, HtCFF: complex, EtCFF: complex):
    """ Combining the BH, Pure DVCS and interfernce cross-sections

    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): _description_
        Q (float): momentum transfer squared
        phi (float): azimuthal angel
        pol (_type_): polarization configuration
        HCFF (complex): Compton form factor H 
        ECFF (complex): Compton form factor E
        HtCFF (complex): Compton form factor Ht 
        EtCFF (complex): Compton form factor Et

    Returns:
        total differential cross-sections dsigma (float)
    """
    return dsigma_BH(y, xB, t, Q, phi, pol) + dsigma_DVCS(y, xB, t, Q, phi, pol, HCFF, ECFF, HtCFF, EtCFF) + dsigma_INT(y, xB, t, Q, phi, pol, HCFF, ECFF, HtCFF, EtCFF)

# The total cross-section integrated over phi
def dsigma_DVCS_HERA(y: float, xB: float, t: float, Q: float, pol, HCFF: complex, ECFF: complex, HtCFF: complex, EtCFF: complex):
    """cross-sections of virtual photon and proton scattering check for instance https://arxiv.org/abs/2310.13837
    
    | A flux factor from leptonic emission of virtual photon will be taken out compared to the leptonic-proton cross-sections
    | HEAR only measures the virtual-photon-proton cross-sections and integrate over phi

    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): _description_
        Q (float): momentum transfer squared
        pol (_type_): polarization configuration
        HCFF (complex): Compton form factor H 
        ECFF (complex): Compton form factor E
        HtCFF (complex): Compton form factor Ht 
        EtCFF (complex): Compton form factor Et

    Returns:
        total differential cross-sections dsigma (float) of virtual-photon and proton
    """
    Conv = alphaEM * (1- y + 1/2 * y ** 2)/(Pi * y * Q ** 2) * y / xB
    return 1/Conv * quad_vec(lambda phi: dsigma_DVCS(y, xB, t, Q, phi, pol,HCFF, ECFF, HtCFF, EtCFF), 0, 2 * Pi)[0]

@np.vectorize
def _pol_to_index(pol: str):
    """A helper function to convert pol string to integer. Because numba can deal with integer array but not string array
    
    pol (string): polarization configuration ={U,L}{U,L,Tin,Tout} first one is beam polarization, second one is target polarization
    """
    if(pol == "UU"):
        return 0

    if(pol == "LU"):
        return 1

    if(pol == "UL"):
        return 2

    if(pol == "LL"):
        return 3
    
    if(pol == "UTin"):
        return 4

    if(pol == "LTin"):
        return 5

    if(pol == "UTout"):
        return 6

    if(pol == "LTout"):
        return 7

    return -999

# The Bethe-Heitler cross-section contribute to four polarization configurations
def dsigma_BH(y: float, xB: float, t: float, Q: float, phi: float, pol):
    """A helper function converting the pol to number. check __dsigma_BH() for details
        
    pol (string): polarization configuration ={U,L}{U,L,Tin,Tout}, check _pol_to_index()
    """

    pol_index = _pol_to_index(pol)
    
    return __dsigma_BH(y, xB, t, Q, phi, pol_index)


# The Bethe-Heitler cross-section contribute to four polarization configurations
@vectorize(["float64(float64, float64, float64, float64, float64, int32)",
            "float64(float64, float64, float64, float64, float64, int64)"])
def __dsigma_BH(y: float, xB: float, t: float, Q: float, phi: float, pol_index: int):
    """The actual BH cross-section formulae, check https://inspirehep.net/literature/1925449 

    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): _description_
        Q (float): momentum transfer squared
        pol_index (int): polarization configuration, check _pol_to_index()

    Returns:
        differential cross-sections of BH 
    """
    prefac_BH = conv * Gamma_prefac(y, xB, Q) * t ** (-2)

    # For unpolarized/longitudinally polarized target, integrate over dphi_S gives an extra factor of 2 * pi.

    if(pol_index == 0):
        return 2 * Pi * prefac_BH * (A_BH(y, xB, t, Q, phi) * ( F1_FF(t) ** 2 - t / (4 * M ** 2) * F2_FF(t) ** 2 ) + B_BH(y, xB, t, Q, phi) * ( F1_FF(t)+ F2_FF(t) ) ** 2 )
    
    if(pol_index == 3):
        return 2 * Pi * prefac_BH * (At_BH_L(y, xB, t, Q, phi) * ( F1_FF(t) * F2_FF(t) + F2_FF(t) ** 2 ) + Bt_BH_L(y, xB, t, Q, phi) * ( F1_FF(t)+ F2_FF(t) ) ** 2 )
    
    if(pol_index == 5):
        return prefac_BH * (At_BH_Tin(y, xB, t, Q, phi) * ( F1_FF(t) * F2_FF(t) + F2_FF(t) ** 2 ) + Bt_BH_Tin(y, xB, t, Q, phi) * ( F1_FF(t)+ F2_FF(t) ) ** 2 )
    
    if(pol_index == 7):
        return prefac_BH * (At_BH_Tout(y, xB, t, Q, phi) * ( F1_FF(t) * F2_FF(t) + F2_FF(t) ** 2 ) + Bt_BH_Tout(y, xB, t, Q, phi) * ( F1_FF(t)+ F2_FF(t) ) ** 2 )
    
    return 0


def dsigma_DVCS(y: float, xB: float, t: float, Q: float, phi: float, pol, HCFF: complex, ECFF: complex, HtCFF: complex, EtCFF: complex):
    """A helper function converting the pol to number. check __dsigma_DVCS() for details
        
    pol (string): polarization configuration ={U,L}{U,L,Tin,Tout}, check _pol_to_index()
    """
    pol_index = _pol_to_index(pol)
  
    return __dsigma_DVCS(y, xB, t, Q, phi, pol_index, HCFF, ECFF, HtCFF, EtCFF)


@vectorize(["float64(float64, float64, float64, float64, float64, int32, complex128, complex128, complex128, complex128)",
            "float64(float64, float64, float64, float64, float64, int64, complex128, complex128, complex128, complex128)"])
def __dsigma_DVCS(y: float, xB: float, t: float, Q: float, phi: float, pol_index, HCFF: complex, ECFF: complex, HtCFF: complex, EtCFF: complex):
    """The actual pure DVCS cross-section formulae, check https://inspirehep.net/literature/1925449 

    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): _description_
        Q (float): momentum transfer squared
        pol_index (int): polarization configuration, check _pol_to_index()
        HCFF (complex): Compton form factor H 
        ECFF (complex): Compton form factor E
        HtCFF (complex): Compton form factor Ht 
        EtCFF (complex): Compton form factor Et

    Returns:
        differential cross-sections of pure DVCS 
    """
    prefac_DVCS = conv * Gamma_prefac(y, xB, Q) * Q ** (-4) 

    Xi = xi(xB, t, Q)

    # For unpolarized/longitudinally polarized target, integrate over dphi_S gives an extra factor of 2 * pi.

    if(pol_index == 0):
        return 2 * Pi * prefac_DVCS * 4 * h_pureDVCS(y, xB, t, Q, phi) * Real( (1 - Xi ** 2) * (Conjugate(HCFF) * HCFF + Conjugate(HtCFF) * HtCFF ) - t / (4 * M ** 2) * ( Conjugate(ECFF)* ECFF + Xi ** 2 * Conjugate(EtCFF)* EtCFF) - Xi ** 2 * (Conjugate(ECFF)* ECFF + Conjugate(ECFF) * HCFF + Conjugate(HCFF)* ECFF +   Conjugate(EtCFF) * HtCFF + Conjugate(HtCFF)* EtCFF) )
    
    if(pol_index == 6):
        return prefac_DVCS * 4 * N(xB, t, Q) * h_pureDVCS(y,xB,t,Q,phi) * Imag( Conjugate(HCFF) * ECFF - Xi * Conjugate(HtCFF) * EtCFF) 

    if(pol_index == 3):
        return 2 * Pi * prefac_DVCS * 8 * hminus_pureDVCS(y, xB, t, Q, phi) * Real( (1 - Xi ** 2) * Conjugate(HtCFF) * HCFF  - Xi ** 2 * ( Conjugate(HtCFF)* ECFF + Conjugate(EtCFF)* HCFF) - (Xi ** 2/ (1 + Xi) + t /(4 * M ** 2))* Xi* Conjugate(EtCFF) * ECFF)
    
    if(pol_index == 5):
        return prefac_DVCS * 4 * N(xB, t, Q) * hminus_pureDVCS(y,xB,t,Q,phi) * Real( Conjugate(HtCFF) * ECFF - Xi * Conjugate(EtCFF) * HCFF - Xi ** 2/ (1 + Xi) * Conjugate(EtCFF) * ECFF ) 

    return 0

def dsigma_INT(y: float, xB: float, t: float, Q: float, phi: float, pol, HCFF: complex, ECFF: complex, HtCFF: complex, EtCFF: complex):
    """A helper function converting the pol to number. check __dsigma_INT() for details
    
    pol (string): polarization configuration ={U,L}{U,L,Tin,Tout}, check _pol_to_index()
    """
    pol_index = _pol_to_index(pol)

    return __dsigma_INT(y, xB, t, Q, phi, pol_index, HCFF, ECFF, HtCFF, EtCFF)

    


@vectorize(["float64(float64, float64, float64, float64, float64, int32, complex128, complex128, complex128, complex128)",
            "float64(float64, float64, float64, float64, float64, int64, complex128, complex128, complex128, complex128)"])
def __dsigma_INT(y: float, xB: float, t: float, Q: float, phi: float, pol_index, HCFF: complex, ECFF: complex, HtCFF: complex, EtCFF: complex):
    """The actual interference cross-section formulae, check https://inspirehep.net/literature/1925449 

    Args:
        y (float): Beam energy lost parameter
        xB (float): x_bjorken
        t (float): _description_
        Q (float): momentum transfer squared
        pol_index (int): polarization configuration, check _pol_to_index()
        HCFF (complex): Compton form factor H 
        ECFF (complex): Compton form factor E
        HtCFF (complex): Compton form factor Ht 
        EtCFF (complex): Compton form factor Et

    Returns:
        differential cross-sections of interference 
    """
    prefac_INT = conv * Gamma_prefac(y, xB, Q) * Q ** (-2) * (-t) ** (-1)

    Xi = xi(xB, t, Q)
    
    # For unpolarized/longitudinally polarized target, integrate over dphi_S gives an extra factor of 2 * pi.

    if(pol_index == 0):
        return 2 * Pi * prefac_INT * Real(A_INT_unp(y, xB, t, Q, phi) * (HCFF * F1_FF(t) - t/ (4* M ** 2) * ECFF * F2_FF(t)) + B_INT_unp(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HCFF + ECFF) + C_INT_unp(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * HtCFF )

    if(pol_index == 1):
        return 2 * Pi * prefac_INT * Imag(A_INT_pol(y, xB, t, Q, phi) * (HCFF * F1_FF(t) - t/ (4* M ** 2) * ECFF * F2_FF(t)) + B_INT_pol(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HCFF + ECFF) + C_INT_pol(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * HtCFF )

    if(pol_index == 2):
        return 2 * Pi * prefac_INT * Imag(At_INT_unp(y, xB, t, Q, phi) * ( F1_FF(t) * (HtCFF - Xi ** 2/ (1 + Xi) * EtCFF) - F2_FF(t) * t/ (4 * M ** 2) * Xi* EtCFF ) + Bt_INT_unp(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HtCFF  + Xi/ (1 + Xi ) * EtCFF) - Ct_INT_unp(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HCFF  + Xi/ (1 + Xi ) * ECFF)  )

    if(pol_index == 3):
        return 2 * Pi * prefac_INT * (-1) * Real(At_INT_pol(y, xB, t, Q, phi) * ( F1_FF(t) * (HtCFF - Xi ** 2/ (1 + Xi) * EtCFF) - F2_FF(t) * t/ (4 * M ** 2) * Xi* EtCFF ) + Bt_INT_pol(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HtCFF  + Xi/ (1 + Xi ) * EtCFF) - Ct_INT_pol(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HCFF  + Xi/ (1 + Xi ) * ECFF)  )
    
    if(pol_index == 4):
        return prefac_INT * 2 / N(xB, t, Q) * Imag( At_INT_unp(y, xB, t, Q, phi) * ( Xi * F1_FF(t) * ( Xi * HtCFF + (Xi ** 2 /(1 + Xi) + t/ (4 * M ** 2)) * EtCFF) + F2_FF(t) * t /(4 * M ** 2) *( (Xi ** 2 -1) * HtCFF + Xi ** 2 * EtCFF ) ) + Bt_INT_unp(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HtCFF + (t /(4 * M ** 2)- Xi/(1 + Xi) )* Xi *EtCFF) + Ct_INT_unp(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (Xi * HCFF + (t /(4 * M ** 2) + Xi ** 2 /(1 + Xi) )* ECFF) )

    if(pol_index == 5):
        return prefac_INT * (-2) / N(xB, t, Q) * Real( At_INT_pol(y, xB, t, Q, phi) * ( Xi * F1_FF(t) * ( Xi * HtCFF + (Xi ** 2 /(1 + Xi) + t/ (4 * M ** 2)) * EtCFF) + F2_FF(t) * t /(4 * M ** 2) *( (Xi ** 2 -1) * HtCFF + Xi ** 2 * EtCFF ) ) + Bt_INT_pol(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HtCFF + (t /(4 * M ** 2)- Xi/(1 + Xi) )* Xi *EtCFF) + Ct_INT_pol(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (Xi * HCFF + (t /(4 * M ** 2) + Xi ** 2 /(1 + Xi) )* ECFF) )

    if(pol_index == 6):
        return prefac_INT * (-2) / N(xB, t, Q) * Imag( A_INT_unp(y, xB, t, Q, phi) * (F1_FF(t) * (Xi ** 2 * HCFF +(Xi ** 2 + t/(4 * M **2)) * ECFF) + F2_FF(t) *t /(4 * M ** 2) * ((Xi ** 2 -1)* HCFF + Xi ** 2 * ECFF) ) +  B_INT_unp(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HCFF + t/ (4 * M **2) *ECFF) - Xi * C_INT_unp(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HtCFF + t/ (4 * M **2) *EtCFF) )

    if(pol_index == 7):
        return prefac_INT * (2) / N(xB, t, Q) * Real( A_INT_pol(y, xB, t, Q, phi) * (F1_FF(t) * (Xi ** 2 * HCFF +(Xi ** 2 + t/(4 * M **2)) * ECFF) + F2_FF(t) *t /(4 * M ** 2) * ((Xi ** 2 -1)* HCFF + Xi ** 2 * ECFF) ) +  B_INT_pol(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HCFF + t/ (4 * M **2) *ECFF) - Xi * C_INT_pol(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HtCFF + t/ (4 * M **2) *EtCFF) )

    return -1

# The codes below are not meant to be readable as they are converted from the master Mathematica code presented in https://inspirehep.net/literature/1925449 
# Refer to our publication for the definition of scalar coefficients and how the cross-section can be expressed with them. 



# Some test print to numerical check with the master Mathematica code
"""
print(A_BH(0.2, 0.1, -0.1, 2, 3.14/2))
print(B_BH(0.2, 0.1, -0.1, 2, 3.14/2))
print(At_BH_L(0.2, 0.1, -0.1, 2, 3.14/2))
print(Bt_BH_L(0.2, 0.1, -0.1, 2, 3.14/2))
print(At_BH_Tin(0.2, 0.1, -0.1, 2, 3.14/2))
print(Bt_BH_Tin(0.2, 0.1, -0.1, 2, 3.14/2))
print(At_BH_Tout(0.2, 0.1, -0.1, 2, 3.14/2))
print(Bt_BH_Tout(0.2, 0.1, -0.1, 2, 3.14/2))
print(h_pureDVCS(0.2, 0.1, -0.1, 2, 3.14/2))
print(hminus_pureDVCS(0.2, 0.1, -0.1, 2, 3.14/2))
print(A_INT_unp(0.2, 0.1, -0.1, 2, 3.14/2))
print(B_INT_unp(0.2, 0.1, -0.1, 2, 3.14/2))
print(C_INT_unp(0.2, 0.1, -0.1, 2, 3.14/2))
print(At_INT_unp(0.2, 0.1, -0.1, 2, 3.14/2))
print(Bt_INT_unp(0.2, 0.1, -0.1, 2, 3.14/2))
print(Ct_INT_unp(0.2, 0.1, -0.1, 2, 3.14/2))
print(A_INT_pol(0.2, 0.1, -0.1, 2, 3.14/2))
print(B_INT_pol(0.2, 0.1, -0.1, 2, 3.14/2))
print(C_INT_pol(0.2, 0.1, -0.1, 2, 3.14/2))
print(At_INT_pol(0.2, 0.1, -0.1, 2, 3.14/2))
print(Bt_INT_pol(0.2, 0.1, -0.1, 2, 3.14/2))
print(Ct_INT_pol(0.2, 0.1, -0.1, 2, 3.14/2))

print(dsigma_DVCS(0.49624,0.34,-0.17,Sqrt(1.82),Pi/3, "UU", -4.19 + 2.67 * 1j, -3.49 + 0.785 * 1j,1.73 + 4.32 *1j,21 + 52*1j ))
print(dsigma_INT(0.49624,0.34,-0.17,Sqrt(1.82),Pi/3, "UU", -4.19 + 2.67 * 1j, -3.49 + 0.785 * 1j,1.73 + 4.32 *1j,21 + 52*1j ))
print(dsigma_INT(0.49624,0.34,-0.17,Sqrt(1.82),Pi/2, "LU", -4.19 + 2.67 * 1j, -3.49 + 0.785 * 1j,1.73 + 4.32 *1j,21 + 52*1j ))
"""