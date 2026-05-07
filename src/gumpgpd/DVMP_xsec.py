r"""
Deeply Virtual Meson Production (DVMP) cross-section calculations.

This module implements the differential cross-sections and the
:math:`R = \sigma_L / \sigma_T` ratio for exclusive meson production
(:math:`\rho^0`, :math:`\phi`, :math:`J/\psi`) off the proton.  The
cross-sections are expressed in terms of helicity Transition Form Factors
(TFFs) :math:`\mathcal{H}` and :math:`\mathcal{E}` computed from GPDs.

Key components:

* :func:`R` — L/T ratio parametrization (Eq. (32) of
    `arXiv:1112.2597 <https://arxiv.org/abs/1112.2597>`_).
* :func:`R_rho_fit` — iMinuit fit of ``R`` parameters to combined H1+ZEUS data.
* :func:`R_fitted` — best-fit :math:`R` with propagated uncertainty.
* :func:`dsigmaL_DVMP_dt` — longitudinal differential cross-section
  :math:`d\sigma_L/dt`.
* :func:`dsigma_DVMP_dt` — total :math:`d\sigma/dt` after L/T unseparation.
"""

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

# ---------------------------------------------------------------------------
# Physical constants: masses (GeV), decay constants (GeV), and unit conversion
# ---------------------------------------------------------------------------
M_p = 0.938
M_n = 0.940
M_rho = 0.775
M_phi = 1.019
M_jpsi = 3.097
gevtonb = 389.9 * 1000
alphaEM = 1 / 137.036

# ---------------------------------------------------------------------------
# R ratio (sigma_L / sigma_T) parametrization and fit to HERA data
# ---------------------------------------------------------------------------

def R(Q: float, a: float, p: float, meson: int) -> float:
    r"""L/T cross-section ratio :math:`R = \sigma_L / \sigma_T` parametrization.

    For :math:`\rho^0` (``meson=1``) the parametrization follows Eq. (32) of
    `arXiv:1112.2597 <https://arxiv.org/pdf/1112.2597>`_:

    .. math::

        R(Q) = \frac{Q^2}{M_{\rho}^2}\left(1 + e^a \frac{Q^2}{M_{\rho}^2}\right)^{-p}

    For :math:`J/\psi` (``meson=3``) the simple ratio :math:`Q^2 / M_{J/\psi}^2` is used.

    Args:
        Q (float): photon virtuality :math:`Q` in GeV.
        a (float): fit parameter controlling the transition scale.
        p (float): fit parameter controlling the power-law fall-off.
        meson (int): meson code — ``1`` for :math:`\rho^0`, ``3`` for
            :math:`J/\psi`; ``2`` is reserved for :math:`\phi`.

    Returns:
        float: :math:`R = \sigma_L / \sigma_T`.
    """
    if (meson==1): 
        return (Q**2 / M_rho**2) * (1 + np.exp(a) * Q**2 / M_rho**2) ** (-p)
    if (meson==3): 
        return  (Q**2/M_jpsi**2)

# Fit R to combined H1 + ZEUS rho data and propagate uncertainties.
# Total cross-sections are converted to longitudinal ones via
#   dsigma_L/dt = (dsigma_tot/dt) / (epsilon + 1/R)
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



    # Taking the maximum of the +/– errors 
    stat_errors = np.maximum(stat_errors_pos, stat_errors_neg)
    syst_errors = np.maximum(syst_errors_pos, syst_errors_neg)

    # Combining the symmetric stat and syst errors in quadrature for total uncertainty
    tot_errors = np.sqrt(stat_errors**2 + syst_errors**2)


    RrhoZEUSnH1['tot_err'] = tot_errors
    RrhoZEUSnH1.to_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/R_rho_ZEUSnH1_w_err.csv'),index=False,header = None)
    
RrhoZEUSnH1_total_err()
'''

RrhoZEUSnH1= pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/R_rho_ZEUSnH1_w_err.csv'), header = None, 
                names = ['Q', 'R','stat_pos','stat_neg','syst_pos','syst_neg','tot_err'] , dtype = {'Q': float, 'R': float, 'stat_pos': float,'stat_neg': float, 'syst_pos':float, 'syst_neg':float})
RrhoZEUSnH1['Q'] = np.sqrt(RrhoZEUSnH1['Q']) # Converting the Q² values in the file to Q by taking the square root.

# Defining the chi² cost function for fitting a, p to the ρ–data:
def R_rho_cost(a: float, p: float) -> float:
    r"""Reduced :math:`\chi^2` cost function for fitting :func:`R` to combined H1+ZEUS :math:`\rho` data.

    Args:
        a (float): fit parameter :math:`a` of :func:`R`.
        p (float): fit parameter :math:`p` of :func:`R`.

    Returns:
        float: :math:`\chi^2_{\rm red} = \sum_i [(R_i^{\rm exp} - R_i^{\rm pred}) /
        \sigma_i]^2 \,/\, N_{\rm dof}`.
    """
    Q_vals = RrhoZEUSnH1['Q'].values 
    R_exp_rho  = RrhoZEUSnH1['R'].values
    tot_errors  = RrhoZEUSnH1['tot_err']
    
    R_pred = R(Q_vals, a, p,meson=1)   # Computing the model prediction for each Q value for ρ meson
    
    chi2 = np.sum(((R_exp_rho - R_pred)/tot_errors) ** 2)  # Standard χ² sum over data points
    ndof = len(Q_vals) - 2
    
    return chi2 / ndof

@cache
def R_rho_fit() -> tuple:
    """Fit :func:`R` parameters ``(a, p)`` to the combined H1+ZEUS :math:`\\rho` data.

    Uses iMinuit ``migrad`` for minimization and ``hesse`` for uncertainty
    estimation.  The result is cached so the fit runs only once per session.

    Returns:
        tuple: ``(val_a, var_a, val_p, var_p, corr_ap)`` where

        * ``val_a`` — best-fit value of :math:`a`
        * ``var_a`` — :math:`1\\sigma` uncertainty on :math:`a`
        * ``val_p`` — best-fit value of :math:`p`
        * ``var_p`` — :math:`1\\sigma` uncertainty on :math:`p`
        * ``corr_ap`` — covariance between :math:`a` and :math:`p`
    """
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

def R_fitted(Q: float, meson: int = 1) -> tuple:
    r"""Evaluate the fitted :func:`R` ratio with propagated :math:`1\sigma` uncertainty.

    Calls :func:`R_rho_fit` to obtain best-fit parameters, then propagates
    their uncertainties analytically via first-order error propagation:

    .. math::

        \sigma_R^2 = \left(\frac{\partial R}{\partial a}\right)^2 \sigma_a^2
                   + \left(\frac{\partial R}{\partial p}\right)^2 \sigma_p^2
                   + 2\,\frac{\partial R}{\partial a}\frac{\partial R}{\partial p}\,
                     \mathrm{cov}(a,\,p)

    Args:
        Q (float): photon virtuality in GeV (scalar or array).
        meson (int): meson code; currently only ``1`` (:math:`\rho^0`) is
            implemented.

    Returns:
        tuple: ``(R_mean, R_std)`` — central value and :math:`1\sigma`
        uncertainty of :math:`R(Q)`.
    """
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
# ---------------------------------------------------------------------------
# Cross-sections for proton target (virtual photon sub-process)
# ---------------------------------------------------------------------------

def epsilon(y: float) -> float:
    r"""Virtual photon polarization parameter :math:`\varepsilon`.

    .. math::

        \varepsilon = \frac{1 - y}{1 - y + y^2/2}

    See Eq. (31) of `arXiv:1112.2597 <https://arxiv.org/pdf/1112.2597>`_.

    Args:
        y (float): inelasticity (beam energy loss fraction).

    Returns:
        float: :math:`\varepsilon \in (0, 1)`.
    """
    return (1 - y) / (1 - y + y**2 / 2)

def MassCorr(meson: int) -> float:
    r"""Meson mass correction entering the propagator denominator.

    Returns :math:`M_{J/\psi}` for :math:`J/\psi` production and ``0`` for
    lighter mesons, where the mass correction to :math:`Q^2` is negligible.

    Args:
        meson (int): meson code — ``1`` for :math:`\rho^0`, ``2`` for
            :math:`\phi`, ``3`` for :math:`J/\psi`.

    Returns:
        float: mass correction in GeV (:math:`M_{J/\psi}` for ``meson=3``,
        ``0`` otherwise).
    """
  
    if (meson==3):
        return  M_jpsi
    else:
        return 0

# -----------------------------------------------------------------------------
# dsigmaL_dt : longitudinal differential cross section (in t)
# -----------------------------------------------------------------------------

@np.vectorize
def dsigmaL_DVMP_dt(y: float, xB: float, t: float, Q: float, meson: int,
                    HTFF: complex, ETFF: complex) -> float:
    r"""Longitudinal DVMP differential cross-section :math:`d\sigma_L/dt`.

    Implements Eq. (2.8) of `arXiv:2409.17231 <https://arxiv.org/pdf/2409.17231>`_:

    .. math::

        \frac{d\sigma_L}{dt} = \frac{4\pi^2\alpha_{\rm EM}\,x_B^2}
            {(Q^2+M_c^2)^2}\cdot\frac{Q^2}{(Q^2+M_c^2)^2}
            \left[|\mathcal{H}|^2 - \frac{t}{4M_p^2}|\mathcal{E}|^2\right]

    where :math:`M_c = M_{J/\psi}` for :math:`J/\psi` and ``0`` otherwise
    (see :func:`MassCorr`).  The result is converted to nb/GeV\ :sup:`2`.

    Args:
        y (float): inelasticity.
        xB (float): Bjorken-:math:`x`.
        t (float): squared momentum transfer :math:`t` in GeV\ :sup:`2`.
        Q (float): photon virtuality :math:`Q` in GeV.
        meson (int): meson code — ``1`` for :math:`\rho^0`, ``2`` for
            :math:`\phi`, ``3`` for :math:`J/\psi`.
        HTFF (complex): helicity-conserving TFF :math:`\mathcal{H}`.
        ETFF (complex): helicity-flip TFF :math:`\mathcal{E}`.

    Returns:
        float: :math:`d\sigma_L/dt` in nb/GeV\ :sup:`2`.
    """

    return gevtonb * ( 4* np.pi**2  *alphaEM * xB ** 2 / ((Q**2 + MassCorr(meson)**2) ** 2)) * (Q/ (Q**2 + MassCorr(meson)**2)) ** 2 * (Real(HTFF* Conjugate(HTFF)) - t/4/ M_p**2 * Real(ETFF* Conjugate(ETFF)))

# -----------------------------------------------------------------------------
# dsigma_dt : total differential cross section (only in t)
# -----------------------------------------------------------------------------

@np.vectorize
def dsigma_DVMP_dt(y: float, xB: float, t: float, Q: float, meson: int,
                   HTFF: complex, ETFF: complex, a: float, p: float) -> float:
    r"""Total (L+T) DVMP differential cross-section :math:`d\sigma/dt`.

    Obtains the longitudinal cross-section from :func:`dsigmaL_DVMP_dt` and
    undoes the L/T separation using Eq. (2.16) of
    `arXiv:2409.17231 <https://arxiv.org/pdf/2409.17231>`_:

    .. math::

        \frac{d\sigma}{dt} = \frac{d\sigma_L}{dt}
            \left(\varepsilon + \frac{1}{R}\right)

    where :math:`\varepsilon` is the virtual-photon polarization
    (see :func:`epsilon`) and :math:`R = \sigma_L/\sigma_T` (see :func:`R`).

    Args:
        y (float): inelasticity.
        xB (float): Bjorken-:math:`x`.
        t (float): squared momentum transfer :math:`t` in GeV\ :sup:`2`.
        Q (float): photon virtuality :math:`Q` in GeV.
        meson (int): meson code — ``1`` for :math:`\rho^0`, ``2`` for
            :math:`\phi`, ``3`` for :math:`J/\psi`.
        HTFF (complex): helicity-conserving TFF :math:`\mathcal{H}`.
        ETFF (complex): helicity-flip TFF :math:`\mathcal{E}`.
        a (float): :func:`R` fit parameter :math:`a`.
        p (float): :func:`R` fit parameter :math:`p`.

    Returns:
        float: :math:`d\sigma/dt` in nb/GeV\ :sup:`2`.
    """

    return  dsigmaL_DVMP_dt(y, xB, t, Q, meson, HTFF, ETFF)*(epsilon(y)+1/R(Q,a,p,meson))