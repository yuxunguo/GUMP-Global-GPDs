### Module for calculating meson production cross-sections and TFFs

import numpy as np
import pandas as pd
import os
from iminuit import Minuit
from functools import cache

from numpy import cos as Cos
from numpy import sin as Sin
from numpy import real as Real
from numpy import imag as Imag
from numpy import conjugate as Conjugate

dir_path = os.path.dirname(os.path.realpath(__file__))

"""
***************************** Masses, decay constants, etc. ***********************
"""
M_p = 0.938
M_n = 0.940
M_rho = 0.775
M_phi = 1.019
M_jpsi = 3.097
gevtonb = 389.9 * 1000
alphaEM = 1 / 137.036

"""
******* R Ratio (longitudinal/transverse separation) Parametrization and Fits ************
"""

def R(Q:float, a:float, p:float, meson:int):
    """ The R ratio: Longitudinal DVMP cross section/ Transverse DVMP cross section
    
    Args:
       Q: The photon virtuality 
       a: Parameter
       p:Parameter
       meson: 1 for rho, 3 for jpsi, 2 is saved for phi to use later
    
    Returns: The parametrization of R factor of the  L/T separation  as in  Eq.(32) in https://arxiv.org/pdf/1112.2597"
    
    """
    if (meson==1): 
        return (Q**2 / M_rho**2) * (1 + np.exp(a) * Q**2 / M_rho**2) ** (-p)
    if (meson==3): 
        return  (Q**2/M_jpsi**2)

"""
************************ Preprocess the R ratio: we fit the R ratio measure from experiments and extrapolate with uncertainties ****************************

R(Q,a,p,meson): the ratio σ_L / σ_T for meson production follows Eq.(32) in arXiv:1112.2597 for ρ, but here we handle both ρ (meson==1) and J/ψ (meson==3) cases in one function.

This will convert all the measured total cross-sections into longitudinal cross-sections that theory predicts with dσ_L/dt= (dσ_tot/dt) / (ε + 1/R) with error propagation
"""
#Below we convert the raw data into the one with total errors for future use.
# Loading the combined H1 and ZEUS R‐ratio data for ρ meson:
# We’ve taken both the ZEUS and H1 measurements, merged them into one table, and now we are fitting a single parametrization to the combined HERA data.
'''
def RrhoZEUSnH1_total_err():

    RrhoZEUSnH1= pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/R_rho_ZEUSnH1.csv'), header = None, 
                names = ['Q', 'R','stat_pos','stat_neg','syst_pos','syst_neg'] , dtype = {'Q': float, 'R': float, 'stat_pos': float,'stat_neg': float, 'syst_pos':float, 'syst_neg':float})

    # Generating the total errors of R --- Preprocessed already
    stat_errors_pos=RrhoZEUSnH1['stat_pos'].to_numpy()
    stat_errors_neg=RrhoZEUSnH1['stat_neg'].to_numpy()
    syst_errors_pos=RrhoZEUSnH1['syst_pos'].to_numpy()
    syst_errors_neg=RrhoZEUSnH1['syst_neg'].to_numpy()

    # Combining +/– errors in quadrature to get symmetric stat & syst errors,
    # then combining those to get a total uncertainty:
        
    stat_errors = np.sqrt(stat_errors_pos**2 + stat_errors_neg**2)/np.sqrt(2)
    syst_errors = np.sqrt(syst_errors_pos**2 + syst_errors_neg**2)/np.sqrt(2)
    tot_errors  = np.sqrt(stat_errors**2 + syst_errors**2)
    RrhoZEUSnH1['tot_err'] = tot_errors
    RrhoZEUSnH1.to_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/R_rho_ZEUSnH1_w_err.csv'),index=False,header = None)
    
RrhoZEUSnH1_total_err()
'''

RrhoZEUSnH1= pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/R_rho_ZEUSnH1_w_err.csv'), header = None, 
                names = ['Q', 'R','stat_pos','stat_neg','syst_pos','syst_neg','tot_err'] , dtype = {'Q': float, 'R': float, 'stat_pos': float,'stat_neg': float, 'syst_pos':float, 'syst_neg':float})
RrhoZEUSnH1['Q'] = np.sqrt(RrhoZEUSnH1['Q']) # Converting the Q² values in the file to Q by taking the square root.

# Defining the chi² cost function for fitting a, p to the ρ–data:
def R_rho_cost(a, p):
    """
    Cost function for R(Q) using H1 and Zeus data combined.
    
    Parameters:
      
      a, p  : Free parameters in the model
      
    Returns:
      Reduced chi2: (Sum of squared differences between model prediction and experimental data divided by the total errror)/Degrees of freedom
      
    """
    Q_vals = RrhoZEUSnH1['Q'].values 
    R_exp_rho  = RrhoZEUSnH1['R'].values
    tot_errors  = RrhoZEUSnH1['tot_err']
    
    R_pred = R(Q_vals, a, p,meson=1)   # Computing the model prediction for each Q value for ρ meson
    
    chi2 = np.sum(((R_exp_rho - R_pred)/tot_errors) ** 2)  # Standard χ² sum over data points
    ndof = len(Q_vals) - 2
    
    return chi2 / ndof

#Using iminuit to minimize the cost function and extract best–fit values:
@cache
def R_rho_fit():

    m = Minuit(R_rho_cost, a=2.5, p=0.7) # initial guesses.

    m.migrad()  # run the minimizer
    m.hesse()   # compute the uncertainties via the Hessian

    # Pulling out fitted parameters and their 1σ errors:
    val_a = m.values['a']
    val_p = m.values['p']
    var_a = m.errors['a']
    var_p = m.errors['p']

    # Correlation between a & p from the off–diagonal of the covariance matrix:
    corr_ap = m.covariance[0,1] 
    
    return val_a, var_a, val_p, var_p, corr_ap

def R_fitted(Q, meson: int =1):
    
    assert meson == 1, 'Not implemented yet, only rho meson (=1) now!'
        
    if(meson ==1):
        
        val_a, var_a, val_p, var_p, corr_ap = R_rho_fit()
        
    R_Mean = R(Q, val_a, val_p,meson)
    
    partial_derivative_a=-val_p * (Q**2 / M_rho**2)**2 * np.exp(val_a) * (1 + np.exp(val_a) * Q**2 / M_rho**2) ** (-val_p-1)
    partial_derivative_p=-(Q**2 / M_rho**2) * (1 + np.exp(val_a) * Q**2 / M_rho**2)**(-val_p) * np.log(1 + np.exp(val_a) * Q**2 / M_rho**2)
    
    part_a     = partial_derivative_a**2 * var_a**2
    part_p     = partial_derivative_p**2 * var_p**2
    part_ap    = partial_derivative_a*partial_derivative_p*corr_ap
    variance_R = part_a + part_p + 2 * part_ap

    return R_Mean, np.sqrt(variance_R)

# Plotting the results of the R fit (only for debugging)
'''
def R_fit_plt():

    # 2) Building an array of σ[R] over the fit grid Q_fit:
    Q_fit = np.linspace(min(RrhoZEUSnH1['Q']), max(RrhoZEUSnH1['Q']))
    R_rho_Mean, R_rho_Std = R_fitted(Q_fit) 
    
    # 3) Upper & lower error bands:
    R_upper_rho = R_rho_Mean + R_rho_Std
    R_lower_rho= R_rho_Mean - R_rho_Std    

    # 4) Plotting the central fit, the ±1σ band, and the data points:  

    Q_vals = RrhoZEUSnH1['Q'].to_numpy() 
    R_exp_rho  = RrhoZEUSnH1['R'].to_numpy()
    tot_errors  = RrhoZEUSnH1['tot_err'].to_numpy()
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(Q_fit, R_rho_Mean, label='R(Q^2)', color='blue')
    plt.fill_between(Q_fit, R_lower_rho, R_upper_rho, color='blue', alpha=0.2, label='Error Band')
    plt.errorbar(Q_vals, R_exp_rho, yerr=[tot_errors], fmt='bo', label="Experimental Data", capsize=5)
    plt.title('R(Q) with Variance Error Bands')
    plt.xlabel('$Q^2$ (GeV$^2$)')
    plt.ylabel('R($Q^2$)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
R_fit_plt()
'''
"""
******************************Cross-sections for proton target (currently for virtual photon scattering sub-process)*********************************
"""

def epsilon(y:float):
    """ Photon polarizability.

    Args:
       y (float): Beam energy lost parameter
     

    Returns:
        epsilon:  "Eq.(31) in https://arxiv.org/pdf/1112.2597" 
    """
    return (1 - y) / (1 - y + y**2 / 2)

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

# -----------------------------------------------------------------------------
# dsigmaL_dt : longitudinal differential cross section (in t)
# -----------------------------------------------------------------------------

@np.vectorize
def dsigmaL_DVMP_dt(y: float, xB: float, t: float, Q: float, meson:int, HTFF: complex, ETFF: complex):
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

    return gevtonb * ( 4* np.pi**2  *alphaEM * xB ** 2 / ((Q**2 + MassCorr(meson)**2) ** 2)) * (Q/ (Q**2 + MassCorr(meson)**2)) ** 2 * (Real(HTFF* Conjugate(HTFF)) - t/4/ M_p**2 * Real(ETFF* Conjugate(ETFF)))

# -----------------------------------------------------------------------------
# dsigma_dt : total differential cross section (only in t)
# -----------------------------------------------------------------------------

@np.vectorize
def dsigma_DVMP_dt(y: float, xB: float, t: float, Q: float, meson:int, HTFF: complex, ETFF: complex,a:float,p:float):
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

    return  dsigmaL_DVMP_dt(y, xB, t, Q, meson, HTFF, ETFF)*(epsilon(y)+1/R(Q,a,p,meson))