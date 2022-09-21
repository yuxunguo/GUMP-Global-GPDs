"""
In this module we have the DVCS cross-section formulas with twist-two Compton form factors based on our work https://inspirehep.net/literature/1925449

The five fold cross-section dsigma over dxB dt dQ dphi dphi_S will be calculated.
For unpolarized/longitudinally polarized target, integrate over dphi_S gives an extra factor of 2 * pi.

"""

#The proton mass M = 0.938 GeV
M = 0.938

#The fine structure constant 
alphaEM = 1 / 137.036

#Conversion factor from GeV to nb for the cross-section
conv = 389.9 * 1000

from numpy import cos as Cos
from numpy import sin as Sin
from numpy import sqrt as Sqrt
from numpy import pi as Pi
from numpy import real as Real
from numpy import imag as Imag
from numpy import conjugate as Conjugate

# The total cross-section is given by the sum of Bethe-Heitler (BH), pure DVCS and interference (INT) contributions
def dsigma_TOT(y: float, xB: float, t: float, Q: float, phi: float, pol, HCFF: complex, ECFF: complex, HtCFF: complex, EtCFF: complex):
    
    return dsigma_BH(y, xB, t, Q, phi, pol) + dsigma_DVCS(y, xB, t, Q, phi, pol, HCFF, ECFF, HtCFF, EtCFF) + dsigma_INT(y, xB, t, Q, phi, pol, HCFF, ECFF, HtCFF, EtCFF)

# The Bethe-Heitler cross-section contribute to four polarization configurations
def dsigma_BH(y: float, xB: float, t: float, Q: float, phi: float, pol):

    prefac_BH = conv * Gamma_prefac(y, xB, Q) * t ** (-2)

    if(pol == "UU"):
        return prefac_BH * (A_BH(y, xB, t, Q, phi) * ( F1_FF(t) ** 2 - t / (4 * M ** 2) * F2_FF(t) ** 2 ) + B_BH(y, xB, t, Q, phi) * ( F1_FF(t)+ F2_FF(t) ) ** 2 )
    
    if(pol == "LL"):
        return prefac_BH * (At_BH_L(y, xB, t, Q, phi) * ( F1_FF(t) * F2_FF(t) + F2_FF(t) ** 2 ) + Bt_BH_L(y, xB, t, Q, phi) * ( F1_FF(t)+ F2_FF(t) ) ** 2 )
    
    if(pol == "LTin"):
        return prefac_BH * (At_BH_Tin(y, xB, t, Q, phi) * ( F1_FF(t) * F2_FF(t) + F2_FF(t) ** 2 ) + Bt_BH_Tin(y, xB, t, Q, phi) * ( F1_FF(t)+ F2_FF(t) ) ** 2 )
    
    if(pol == "LTout"):
        return prefac_BH * (At_BH_Tout(y, xB, t, Q, phi) * ( F1_FF(t) * F2_FF(t) + F2_FF(t) ** 2 ) + Bt_BH_Tout(y, xB, t, Q, phi) * ( F1_FF(t)+ F2_FF(t) ) ** 2 )
    
    return 0

def dsigma_DVCS(y: float, xB: float, t: float, Q: float, phi: float, pol, HCFF: complex, ECFF: complex, HtCFF: complex, EtCFF: complex):

    prefac_DVCS = conv * Gamma_prefac(y, xB, Q) * Q ** (-4) 

    Xi = xi(xB, t, Q)

    if(pol == "UU"):
        return prefac_DVCS * 4 * h_pureDVCS(y, xB, t, Q, phi) * Real( (1 - Xi ** 2) * (Conjugate(HCFF) * HCFF + Conjugate(HtCFF) * HtCFF ) - t / (4 * M ** 2) * ( Conjugate(ECFF)* ECFF + Xi ** 2 * Conjugate(EtCFF)* EtCFF) - Xi ** 2 * (Conjugate(ECFF)* ECFF + Conjugate(ECFF) * HCFF + Conjugate(HCFF)* ECFF +   Conjugate(EtCFF) * HtCFF + Conjugate(HtCFF)* EtCFF) )
    
    if(pol == "UTout"):
        return prefac_DVCS * 4 * N(xB, t, Q) * h_pureDVCS(y,xB,t,Q,phi) * Imag( Conjugate(HCFF) * ECFF - Xi * Conjugate(HtCFF) * EtCFF) 

    if(pol == "LL"):
        return prefac_DVCS * 8 * hminus_pureDVCS(y, xB, t, Q, phi) * Real( (1 - Xi ** 2) * Conjugate(HtCFF) * HCFF  - Xi ** 2 * ( Conjugate(HtCFF)* ECFF + Conjugate(EtCFF)* HCFF) - (Xi ** 2/ (1 + Xi) + t /(4 * M ** 2))* Xi* Conjugate(EtCFF) * ECFF)
    
    if(pol == "LTin"):
        return prefac_DVCS * 4 * N(xB, t, Q) * hminus_pureDVCS(y,xB,t,Q,phi) * Real( Conjugate(HtCFF) * ECFF - Xi * Conjugate(EtCFF) * HCFF - Xi ** 2/ (1 + Xi) * Conjugate(EtCFF) * ECFF ) 

    return 0

def dsigma_INT(y: float, xB: float, t: float, Q: float, phi: float, pol, HCFF: complex, ECFF: complex, HtCFF: complex, EtCFF: complex):

    prefac_INT = conv * Gamma_prefac(y, xB, Q) * Q ** (-2) * (-t) ** (-1)

    Xi = xi(xB, t, Q)

    if(pol == "UU"):
        return prefac_INT * Real(A_INT_unp(y, xB, t, Q, phi) * (HCFF * F1_FF(t) - t/ (4* M ** 2) * ECFF * F2_FF(t)) + B_INT_unp(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HCFF + ECFF) + C_INT_unp(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * HtCFF )

    if(pol == "LU"):
        return prefac_INT * Imag(A_INT_pol(y, xB, t, Q, phi) * (HCFF * F1_FF(t) - t/ (4* M ** 2) * ECFF * F2_FF(t)) + B_INT_pol(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HCFF + ECFF) + C_INT_pol(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * HtCFF )

    if(pol == "UL"):
        return prefac_INT * Imag(At_INT_unp(y, xB, t, Q, phi) * ( F1_FF(t) * (HtCFF - Xi ** 2/ (1 + Xi) * EtCFF) - F2_FF(t) * t/ (4 * M ** 2) * Xi* EtCFF ) + Bt_INT_unp(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HtCFF  + Xi/ (1 + Xi ) * EtCFF) - Ct_INT_unp(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HCFF  + Xi/ (1 + Xi ) * ECFF)  )

    if(pol == "LL"):
        return prefac_INT * (-1) * Real(At_INT_pol(y, xB, t, Q, phi) * ( F1_FF(t) * (HtCFF - Xi ** 2/ (1 + Xi) * EtCFF) - F2_FF(t) * t/ (4 * M ** 2) * Xi* EtCFF ) + Bt_INT_pol(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HtCFF  + Xi/ (1 + Xi ) * EtCFF) - Ct_INT_pol(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HCFF  + Xi/ (1 + Xi ) * ECFF)  )
    
    if(pol == "UTin"):
        return prefac_INT * 2 / N(xB, t, Q) * Imag( At_INT_unp(y, xB, t, Q, phi) * ( Xi * F1_FF(t) * ( Xi * HtCFF + (Xi ** 2 /(1 + Xi) + t/ (4 * M ** 2)) * EtCFF) + F2_FF(t) * t /(4 * M ** 2) *( (Xi ** 2 -1) * HtCFF + Xi ** 2 * EtCFF ) ) + Bt_INT_unp(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HtCFF + (t /(4 * M ** 2)- Xi/(1 + Xi) )* Xi *EtCFF) + Ct_INT_unp(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (Xi * HCFF + (t /(4 * M ** 2) + Xi ** 2 /(1 + Xi) )* ECFF) )

    if(pol == "LTin"):
        return prefac_INT * (-2) / N(xB, t, Q) * Real( At_INT_pol(y, xB, t, Q, phi) * ( Xi * F1_FF(t) * ( Xi * HtCFF + (Xi ** 2 /(1 + Xi) + t/ (4 * M ** 2)) * EtCFF) + F2_FF(t) * t /(4 * M ** 2) *( (Xi ** 2 -1) * HtCFF + Xi ** 2 * EtCFF ) ) + Bt_INT_pol(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HtCFF + (t /(4 * M ** 2)- Xi/(1 + Xi) )* Xi *EtCFF) + Ct_INT_pol(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (Xi * HCFF + (t /(4 * M ** 2) + Xi ** 2 /(1 + Xi) )* ECFF) )

    if(pol == "UTout"):
        return prefac_INT * (-2) / N(xB, t, Q) * Imag( A_INT_unp(y, xB, t, Q, phi) * (F1_FF(t) * (Xi ** 2 * HCFF +(Xi ** 2 + t/(4 * M **2)) * ECFF) + F2_FF(t) *t /(4 * M ** 2) * ((Xi ** 2 -1)* HCFF + Xi ** 2 * ECFF) ) +  B_INT_unp(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HCFF + t/ (4 * M **2) *ECFF) - Xi * C_INT_unp(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HtCFF + t/ (4 * M **2) *EtCFF) )

    if(pol == "LTout"):
        return prefac_INT * (2) / N(xB, t, Q) * Real( A_INT_pol(y, xB, t, Q, phi) * (F1_FF(t) * (Xi ** 2 * HCFF +(Xi ** 2 + t/(4 * M **2)) * ECFF) + F2_FF(t) *t /(4 * M ** 2) * ((Xi ** 2 -1)* HCFF + Xi ** 2 * ECFF) ) +  B_INT_pol(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HCFF + t/ (4 * M **2) *ECFF) - Xi * C_INT_pol(y, xB, t, Q, phi) * (F1_FF(t) + F2_FF(t)) * (HtCFF + t/ (4 * M **2) *EtCFF) )

    return -1

# The codes below are not meant to be readable as they are converted from the master Mathematica code presented in https://inspirehep.net/literature/1925449 
# Refer to our publication for the definition of scalar coefficients and how the cross-section can be expressed with them. 

#The prefactor for cross-section given by some kinematics 
def Gamma_prefac(y: float, xB: float, Q: float):
    return (alphaEM**3*xB*y**2)/(16.*Pi**2*Q**4*Sqrt(1 + (4*M**2*xB**2)/Q**2))

def xi(xB: float, t: float, Q: float):
    return (1/(2 - xB) - (2*t*(-1 + xB))/(Q**2*(-2 + xB)**2))*xB

def N(xB: float, t: float, Q: float):
    return Sqrt(- 4 * M ** 2 * xi(xB,t,Q) ** 2 - t * (1 - xi(xB,t,Q) ** 2)) / M

def h_pureDVCS(y: float, xB: float, t: float, Q: float, phi: float):
    return -0.5*(Q**4*(Q**2 + t*(-1 + 2*xB))*(-2 + y))/((Q**2 + t)*(Q**2 + 4*M**2*xB**2)*y) - (2*Q**3*(Q**2 + t*xB)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))/(M*(Q**2 + t)*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*y) + (Q**4*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))**2)/(M**2*(Q**2 + t)**2*xB**2*(Q**2 + 4*M**2*xB**2)**3*y**2)

def hminus_pureDVCS(y: float, xB: float, t: float, Q: float, phi: float):    
    return (Q**4*(Q**2 + t*(-1 + 2*xB))*(-2 + y))/(2.*Sqrt((Q**2 + t)**2)*(Q**2 + 4*M**2*xB**2)*y) + (2*Q**3*(Q**2 + t*xB)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))/(M*Sqrt((Q**2 + t)**2)*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*y)

def A_BH(y: float, xB: float, t: float, Q: float, phi: float):
    return (-8*M*(Q**2 + 4*M**2*xB**2)*(M*(8*M**6*t**2*xB**6*y**2 + 4*M**4*Q**2*t*xB**4*(t + 2*t*xB - 4*M**2*xB**2)*y**2 - Q**8*(t*(-1 + xB) - M**2*xB**2)*(2 - 2*y + y**2) + M**2*Q**4*xB**2*(8*M**4*xB**4*y**2 - 8*M**2*t*xB**2*(-4 + 4*y + (-1 + xB)*y**2) + t**2*(2 - 2*xB**2*(-2 + y)**2 - 2*y + y**2 + 4*xB*y**2)) + Q**6*xB*(4*M**4*xB**3*y**2 + t**2*(2 - xB*(-2 + y)**2 - 2*y + y**2) - 2*M**2*t*xB*(-6 + 6*y + (-3 + 2*xB)*y**2))) + 2*Q**5*(Q**2 + t*xB)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(-2*M**2*t*xB - Q**2*(t - 2*M**2*xB))*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*(-2 + y)*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi) + 8*M**3*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2))*(Q**2*(-1 + y) + M**2*xB**2*y**2)*Cos(phi)**2))/(Q**2*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

def B_BH(y: float, xB: float, t: float, Q: float, phi: float):
    return (-4*M*t*xB*(Q**2 + 4*M**2*xB**2)*(M*xB*(8*M**4*t**2*xB**4*y**2 + 4*M**2*Q**2*t*xB**2*(t - 4*M**2*xB**2)*y**2 + Q**8*(2 - 2*y + y**2) + 2*Q**6*(2*M**2*xB**2*y**2 + t*(-2 + xB*(-2 + y)**2 + 2*y - y**2)) + Q**4*(-8*M**2*t*xB**2*y**2 + 8*M**4*xB**4*y**2 + t**2*(2 - 2*xB*(-2 + y)**2 + 2*xB**2*(-2 + y)**2 - 2*y + y**2))) + 4*Q**5*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + t**2*xB*(-1 + 2*xB) + Q**2*t*(-1 + 3*xB))*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*(-2 + y)*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi) + 8*M*xB*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2))*(Q**2*(-1 + y) + M**2*xB**2*y**2)*Cos(phi)**2))/(Q**2*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

def At_BH_L(y: float, xB: float, t: float, Q: float, phi: float):
    return (-4*M*(Q**2 + 4*M**2*xB**2)**2*y*(-(M*xB*(2*Q**2*t**3*(-1 + xB)*xB*(t*(-1 + xB)**2 + M**2*(4 - 3*xB)*xB) - (4*M**2 - t)*t**4*xB**2*(1 - 3*xB + 2*xB**2) - 2*Q**8*(-2 + xB)*(t - t*xB + M**2*xB**2) + 2*Q**6*t*xB*(-(M**2*xB*(-4 + xB + xB**2)) + t*(3 - 5*xB + 2*xB**2)) + Q**4*t**2*(2*M**2*(5 - 4*xB)*xB**3 + t*(-4 + 10*xB - 7*xB**2 - xB**3 + 2*xB**4)))*(-2 + y)) + 2*Q*(Q**2 + t*xB)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*(2*(4*M**2 - t)*t**3*(-1 + xB)*xB**2 - 2*Q**2*t**2*(-2 + xB)*xB*(t*(-1 + xB) - 2*M**2*xB) + Q**6*(-2 + xB)*(t*(2 - 3*xB) + 4*M**2*xB**2) + Q**4*t*(8*M**2*(-1 + xB)*xB**2 + t*(-4 + 4*xB + xB**2 - 2*xB**3)))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))/((Q**2 + t)*(Q**2*(-2 + xB) + t*(-1 + xB)*xB)*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

def Bt_BH_L(y: float, xB: float, t: float, Q: float, phi: float):
    return (4*M*(Q**2 + 4*M**2*xB**2)**2*y*(-(M*xB*(Q**8*(-2 + xB)*(t*(-2 + xB) - 2*M**2*xB**2) - t**4*(-1 + xB)*xB**2*(4*M**2*(-1 + xB) + t*(-1 + 2*xB)) + Q**6*t*xB*(2*M**2*xB*(4 + xB - 2*xB**2) + t*(8 - 10*xB + 3*xB**2)) - Q**2*t**3*xB*(2*M**2*xB*(4 - 5*xB + 2*xB**2) + t*(4 - 10*xB + 5*xB**2)) + Q**4*t**2*(2*M**2*(7 - 6*xB)*xB**3 + t*(-4 + 8*xB - 5*xB**3 + 2*xB**4)))*(-2 + y)) + 4*Q*(Q**2 + t*xB)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*(t**3*(4*M**2 + t)*(-1 + xB)*xB**2 + Q**6*(-2 + xB)*(t - t*xB + 2*M**2*xB**2) + Q**2*t**2*xB*(2*M**2*(-2 + xB)*xB + t*(-3 + 2*xB)) - Q**4*t*(-4*M**2*(-1 + xB)*xB**2 + t*(2 - 2*xB**2 + xB**3)))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))/((Q**2 + t)*(Q**2*(-2 + xB) + t*(-1 + xB)*xB)*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

def At_BH_Tin(y: float, xB: float, t: float, Q: float, phi: float):
    return (-2*(Q**2 + 4*M**2*xB**2)**2*((4*M**2 - t)*t*(-1 + xB)*xB + Q**2*(-2 + xB)*(-t + 2*M**2*xB))*y*(-(M*xB*(2*Q**2*t**3*(-1 + xB)*xB*(t*(-1 + xB)**2 + M**2*(4 - 3*xB)*xB) - (4*M**2 - t)*t**4*xB**2*(1 - 3*xB + 2*xB**2) - 2*Q**8*(-2 + xB)*(t - t*xB + M**2*xB**2) + 2*Q**6*t*xB*(-(M**2*xB*(-4 + xB + xB**2)) + t*(3 - 5*xB + 2*xB**2)) + Q**4*t**2*(2*M**2*(5 - 4*xB)*xB**3 + t*(-4 + 10*xB - 7*xB**2 - xB**3 + 2*xB**4)))*(-2 + y)) + 2*Q*(Q**2 + t*xB)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*(2*(4*M**2 - t)*t**3*(-1 + xB)*xB**2 - 2*Q**2*t**2*(-2 + xB)*xB*(t*(-1 + xB) - 2*M**2*xB) + Q**6*(-2 + xB)*(t*(2 - 3*xB) + 4*M**2*xB**2) + Q**4*t*(8*M**2*(-1 + xB)*xB**2 + t*(-4 + 4*xB + xB**2 - 2*xB**3)))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))/(Q**2*(Q**2 + t)*(-2 + xB)**2*(Q**2*(-2 + xB) + t*(-1 + xB)*xB)*Sqrt(-(((4*M**2 - t)*t**2*(-1 + xB)**2*xB**2 + Q**2*(4*M**2 - t)*t*xB**2*(2 - 3*xB + xB**2) + Q**4*(-2 + xB)**2*(t - t*xB + M**2*xB**2))/(Q**4*(-2 + xB)**4)))*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

def Bt_BH_Tin(y: float, xB: float, t: float, Q: float, phi: float):
    return (8*M**2*(Q**2*(-2 + xB) + 2*t*(-1 + xB))*xB*(Q**2 + 4*M**2*xB**2)**2*y*(-(M*xB*(-2*M**2*t**4*(-1 + xB)**2*xB**2 - Q**8*(-2 + xB)*(t - t*xB + M**2*xB**2) - Q**2*t**3*xB*(t*(-1 + xB)**2 + M**2*xB*(4 - 5*xB + 2*xB**2)) + Q**6*t*xB*(M**2*xB*(4 + xB - 2*xB**2) + t*(5 - 8*xB + 3*xB**2)) + Q**4*t**2*(M**2*(7 - 6*xB)*xB**3 + t*(-2 + 3*xB + xB**2 - 4*xB**3 + 2*xB**4)))*(-2 + y)) + 2*Q*(Q**2 + t*xB)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*(4*M**2*t**3*(-1 + xB)*xB**2 + Q**2*t**2*xB*(t*(-1 + xB) + 2*M**2*(-2 + xB)*xB) + Q**6*(-2 + xB)*(t - 2*t*xB + 2*M**2*xB**2) + Q**4*t*(4*M**2*(-1 + xB)*xB**2 + t*(-2 + 3*xB**2 - 2*xB**3)))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))/(Q**2*(Q**2 + t)*(-2 + xB)**2*(Q**2*(-2 + xB) + t*(-1 + xB)*xB)*Sqrt(-(((4*M**2 - t)*t**2*(-1 + xB)**2*xB**2 + Q**2*(4*M**2 - t)*t*xB**2*(2 - 3*xB + xB**2) + Q**4*(-2 + xB)**2*(t - t*xB + M**2*xB**2))/(Q**4*(-2 + xB)**4)))*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

def At_BH_Tout(y: float, xB: float, t: float, Q: float, phi: float):
    return (2*(Q**2 - t)*t*(-4*M**2 + t)*(Q**2*(-2 + xB) + 2*t*(-1 + xB))*xB*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)**3*Sqrt(1 - (1 + (2*M**2*(Q**2 + t)*xB**2)/(Q**4 + Q**2*t*xB))**2/(1 + (4*M**2*xB**2)/Q**2))*y*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Sin(phi))/(Q*(Q**2 + t)*(-2 + xB)**2*Sqrt(-(((4*M**2 - t)*t**2*(-1 + xB)**2*xB**2 + Q**2*(4*M**2 - t)*t*xB**2*(2 - 3*xB + xB**2) + Q**4*(-2 + xB)**2*(t - t*xB + M**2*xB**2))/(Q**4*(-2 + xB)**4)))*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

def Bt_BH_Tout(y: float, xB: float, t: float, Q: float, phi: float):
    return (8*M**2*(Q**2 - t)*t*(Q**2*(-2 + xB) + 2*t*(-1 + xB))*xB*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)**3*Sqrt(1 - (1 + (2*M**2*(Q**2 + t)*xB**2)/(Q**4 + Q**2*t*xB))**2/(1 + (4*M**2*xB**2)/Q**2))*y*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Sin(phi))/(Q*(Q**2 + t)*(-2 + xB)**2*Sqrt(-(((4*M**2 - t)*t**2*(-1 + xB)**2*xB**2 + Q**2*(4*M**2 - t)*t*xB**2*(2 - 3*xB + xB**2) + Q**4*(-2 + xB)**2*(t - t*xB + M**2*xB**2))/(Q**4*(-2 + xB)**4)))*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

def A_INT_unp(y: float, xB: float, t: float, Q: float, phi: float):
    return (-64*M**2*xB**2*(Q**2 + 4*M**2*xB**2)**3*y**2*((Q**4*(Q**2 + t)*(Q**2*(-1 + y) + t*(-1 + xB*y)))/(16.*xB*y) - (Q**2*(2*t**2 + Q**2*t*(1 + 2*xB)*y + Q**4*(-2 + 3*y))*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))/(16.*M*xB**2*(Q**2 + 4*M**2*xB**2)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*y**2) + (Q**4*(t*(1 + y) + Q**2*(-1 + 2*y))*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))**2)/(8.*M**2*xB**3*(Q**2 + 4*M**2*xB**2)**3*y**3) - (Q**4*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))**3)/(8.*M**3*xB**4*(Q**2 + 4*M**2*xB**2)**4*Sqrt(1 + (4*M**2*xB**2)/Q**2)*y**3)))/(Q**2*(Q**2 + t)*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

def B_INT_unp(y: float, xB: float, t: float, Q: float, phi: float):
    return (8*Q*t*(Q**2*(-2 + xB) + 2*t*(-1 + xB))*(M*Q*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**2 + t*(-1 + 2*xB))*(-2 + y) + 4*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))/(M*(Q**2 + t)**2*(-2 + xB)**2*(Q**2 + 4*M**2*xB**2)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*y)

def C_INT_unp(y: float, xB: float, t: float, Q: float, phi: float):
    return (-8*(Q**2*(-2 + xB) + 2*t*(-1 + xB))*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(-(M**2*t**2*xB**2) + Q**2*t*xB*(t*(-1 + xB) - 2*M**2*xB) + Q**4*(t*(-1 + xB) - M**2*xB**2))*(-2 + y) + Q*(Q**2 + 4*M**2*xB**2)*(Q**4 + t**2*xB*(-1 + 2*xB) + Q**2*t*(-1 + 3*xB))*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(M*xB*(-4*M**2*Q**2*t**2*xB**2*y**2 - 8*M**4*t**2*xB**4*y**2 + Q**8*(2 - 2*y + y**2) + Q**6*(t*(-1 + 2*xB)*(-2 + y)**2 + 4*M**2*xB**2*y**2) + 2*Q**4*(t**2*(1 - xB*(-2 + y)**2 + xB**2*(-2 + y)**2 - y) + 4*M**4*xB**4*y**2)) + 4*Q**5*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + t**2*xB*(-1 + 2*xB) + Q**2*t*(-1 + 3*xB))*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*(-2 + y)*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi) + 8*M*xB*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2))*(Q**2*(-1 + y) + M**2*xB**2*y**2)*Cos(phi)**2))/(Q**2*(Q**2 + t)*Sqrt((Q**2 + t)**2)*(-2 + xB)**2*Sqrt(1 + (4*M**2*xB**2)/Q**2)*y*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

def At_INT_unp(y: float, xB: float, t: float, Q: float, phi: float):
    return (32*M*Q*Sqrt(t*(-1 + t/(4.*M**2)))*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)**3*Sqrt(1 - (1 + (2*M**2*(Q**2 + t)*xB**2)/(Q**4 + Q**2*t*xB))**2/(1 + (4*M**2*xB**2)/Q**2))*y*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*((Q**2*(Q**2 + t))/4. - ((Q**2 + t)*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))/(2.*M*xB*(Q**2 + 4*M**2*xB**2)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*y) + (Q**2*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))**2)/(2.*M**2*xB**2*(Q**2 + 4*M**2*xB**2)**3*y**2))*Sin(phi))/(Sqrt((Q**2 + t)**2)*Sqrt(t*(-4 + t/M**2))*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

def Bt_INT_unp(y: float, xB: float, t: float, Q: float, phi: float):
    return 0

def Ct_INT_unp(y: float, xB: float, t: float, Q: float, phi: float):
    return (-64*M*Sqrt(t*(-1 + t/(4.*M**2)))*(Q**2*(-2 + xB) + 2*t*(-1 + xB))*xB*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)**3*Sqrt(1 - (1 + (2*M**2*(Q**2 + t)*xB**2)/(Q**4 + Q**2*t*xB))**2/(1 + (4*M**2*xB**2)/Q**2))*y*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*(-0.125*(Q**2*(Q**2 + t)**2) + ((Q**2 - t)*(Q**2 + t)*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))/(4.*M*xB*(Q**2 + 4*M**2*xB**2)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*y) - (Q**2*(Q**2 - t)*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))**2)/(4.*M**2*xB**2*(Q**2 + 4*M**2*xB**2)**3*y**2))*Sin(phi))/(Q*(Q**2 + t)**2*Sqrt(t*(-4 + t/M**2))*(-2 + xB)**2*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

def A_INT_pol(y: float, xB: float, t: float, Q: float, phi: float):
    return (-4*Q**6*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*Sqrt(1 - (1 + (2*M**2*(Q**2 + t)*xB**2)/(Q**4 + Q**2*t*xB))**2/(1 + (4*M**2*xB**2)/Q**2))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*(M*Q*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**2 + t*(-1 + 2*xB))*(-2 + y) + 4*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*Sin(phi))/((Q**2 + t)*xB*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

def B_INT_pol(y: float, xB: float, t: float, Q: float, phi: float):
    return 0

def C_INT_pol(y: float, xB: float, t: float, Q: float, phi: float):
    return (4*Q**4*(Q**2*(-2 + xB) + 2*t*(-1 + xB))*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*Sqrt(1 - (1 + (2*M**2*(Q**2 + t)*xB**2)/(Q**4 + Q**2*t*xB))**2/(1 + (4*M**2*xB**2)/Q**2))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*(M*Q*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**2 + t*(-1 + 2*xB))*(-2 + y) + 4*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*Sin(phi))/(Sqrt((Q**2 + t)**2)*(-2 + xB)**2*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

def At_INT_pol(y: float, xB: float, t: float, Q: float, phi: float):
    return (64*M**2*xB**2*(Q**2 + 4*M**2*xB**2)**3*y**2*((Q**4*(Q**2 + t)*(Q**2*(-1 + y) + t*(-1 + xB*y)))/(16.*xB*y) - (Q**2*(Q**2 + t)*(2*t*xB*y + Q**2*(-2 + 3*y))*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))/(16.*M*xB**2*(Q**2 + 4*M**2*xB**2)*Sqrt(1 + (4*M**2*xB**2)/Q**2)*y**2) + (Q**4*(Q**2 + t*xB)*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))**2)/(8.*M**2*xB**3*(Q**2 + 4*M**2*xB**2)**3*y**2)))/(Q**2*Sqrt((Q**2 + t)**2)*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

def Bt_INT_pol(y: float, xB: float, t: float, Q: float, phi: float):
    return (8*t*(Q**2*(-2 + xB) + 2*t*(-1 + xB))*xB)/(Sqrt((Q**2 + t)**2)*(-2 + xB)**2)

def Ct_INT_pol(y: float, xB: float, t: float, Q: float, phi: float):
    return (8*Q**3*(Q**2*(-2 + xB) + 2*t*(-1 + xB))*(M*Q*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**2 + t*(-1 + 2*xB))*(-2 + y) + 4*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(-(M**2*t**2*xB**2) + Q**2*t*xB*(t*(-1 + xB) - 2*M**2*xB) + Q**4*(t*(-1 + xB) - M**2*xB**2))*(-2 + y) + Q*(Q**2 + 4*M**2*xB**2)*(Q**4 + t**2*xB*(-1 + 2*xB) + Q**2*t*(-1 + 3*xB))*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))/((Q**2 + t)**2*(-2 + xB)**2*(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4*(-1 + y) + 2*M**2*t*xB**2*y + Q**2*(t + t*xB*(-2 + y) + 2*M**2*xB**2*y)) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi))*(-(M*xB*Sqrt(1 + (4*M**2*xB**2)/Q**2)*(Q**4 + 2*M**2*t*xB**2*y + Q**2*(2*M**2*xB**2*y + t*(-1 + 2*xB + y - xB*y)))) + 2*Q*(Q**2 + t*xB)*(Q**2 + 4*M**2*xB**2)*Sqrt(-((M**2*xB**2*(M**2*t**2*xB**2 + Q**2*t*xB*(t + 2*M**2*xB - t*xB) + Q**4*(t - t*xB + M**2*xB**2)))/((Q**3 + Q*t*xB)**2*(Q**2 + 4*M**2*xB**2))))*Sqrt(1 - y - (M**2*xB**2*y**2)/Q**2)*Cos(phi)))

def GE_FF(t):
    return (-9.023778209136253 + 53.766442021698246*Sqrt(0.07728400000000002 - t) + t*(135.53254432369985 - 465.508467241627*Sqrt(0.07728400000000002 - t) + t*(-354.4963834971874 + 489.94062678224225*Sqrt(0.07728400000000002 - t) + t*(166.1704692323253 - 14.197832146876648*Sqrt(0.07728400000000002 - t) + t*(-0.3903668466645851 - 0.30970063887842036*Sqrt(0.07728400000000002 - t) + (-0.07352839999999583 + 0.021159290725357494*Sqrt(0.07728400000000002 - t) + 0.00099999999999989*t)*t)))))/(0.8816371135563656 + Sqrt(0.07728400000000002 - t))**12

def GM_FF(t):
    return (68.10898125637905 - 182.65247233395095*Sqrt(0.07728400000000002 - t) + t*(-869.9440659116333 + 1081.4388068130365*Sqrt(0.07728400000000002 - t) + t*(2158.01873424079 - 1031.0128939386768*Sqrt(0.07728400000000002 - t) + t*(-754.9348356666937 + 123.05692251337982*Sqrt(0.07728400000000002 - t) + t*(-250.50819705017403 + 179.94930173846646*Sqrt(0.07728400000000002 - t) + t*(-52.425345832607995 + 92.77385177374654*Sqrt(0.07728400000000002 - t) + 26.276543999999998*t))))))/(0.8816371135563656 + Sqrt(0.07728400000000002 - t))**12

def F1_FF(t):
    return (GE_FF(t) - t/ (4 * M ** 2) * GM_FF(t))/(1 -  t/ (4 * M ** 2))

def F2_FF(t):
    return (-GE_FF(t) + GM_FF(t))/(1 -  t/ (4 * M ** 2))

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

print(2*Pi*dsigma_DVCS(0.49624,0.34,-0.17,Sqrt(1.82),Pi/3, "UU", -4.19 + 2.67 * 1j, -3.49 + 0.785 * 1j,1.73 + 4.32 *1j,21 + 52*1j ))
print(2*Pi*dsigma_INT(0.49624,0.34,-0.17,Sqrt(1.82),Pi/3, "UU", -4.19 + 2.67 * 1j, -3.49 + 0.785 * 1j,1.73 + 4.32 *1j,21 + 52*1j ))
print(2*Pi*dsigma_INT(0.49624,0.34,-0.17,Sqrt(1.82),Pi/2, "LU", -4.19 + 2.67 * 1j, -3.49 + 0.785 * 1j,1.73 + 4.32 *1j,21 + 52*1j ))
"""