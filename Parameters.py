"""
The minimizer using iMinuit, which takes 1-D array for the input parameters only.

Extra efforts needed to convert the form of the parameters.

"""
# Number of GPD species, 4 leading-twist GPDs including H, E Ht, Et are needed.
NumofGPDSpecies = 4
# Number of flavor factor, Flavor_Factor = 2 * nf + 1 needed including 2 * nf quark (antiquark) and one gluon
Flavor_Factor = 2 * 2 + 1
# Number of ansatz, 1 set of (N, alpha, beta, alphap) will be used to start with
init_NumofAnsatz = 1
# Size of one parameter set, a set of parameters (N, alpha, beta, alphap) contain 4 parameters
Single_Param_Size = 4
# A factor of 2 including the xi^0 and xi^2 terms
xi2_Factor = 2
# Total number of parameters 
Tot_param_Size = NumofGPDSpecies * xi2_Factor * Flavor_Factor *  init_NumofAnsatz * Single_Param_Size

"""
The parameters will form a 5-dimensional matrix such that each para[#1,#2,#3,#4,#5] is a real number.
#1 = [0,1,2,3] corresponds to [H, E, Ht, Et]
#2 = [0,1] corresponds to [xi^0 terms, xi^2 terms]
#3 = [0,1,2,3,4] corresponds to [u - ubar, ubar, d - dbar, dbar, g]
#4 = [0,1,...,init_NumofAnsatz] corresponds to different set of parameters
#5 = [0,1,2,3] correspond to [norm, alpha, beta, alphap] as a set of parameters
"""

import numpy as np
import PDF_Fitting as fit

def ParaManager(Paralst: np.array):

    """
     Here is the parameters manager, as there are over 100 free parameters. Therefore not all of them can be set free.
     Each element F_{q} is a two-dimensional matrix with init_NumofAnsatz = 1 row and Single_Param_Size = 4 columns
    """
    [alphapV_H, alphapS_H, R_H_xi2, R_E, R_E_xi2, alphapV_Ht, alphapS_Ht, R_Ht_xi2, R_Et, R_Et_xi2] = Paralst
    # Initial forward parameters for the H of (uV, ubar, dV, dbar,g) distributions
    """
        Two free parameters alphapV_H, alphapS_H for the valence Regge slope and sea Pomeron slope
    """
    
    H_uV =   np.array([np.insert(fit.uV_Unp, 3, alphapV_H) ])
    H_ubar = np.array([np.insert(fit.ubar_Unp, 3, alphapS_H) ])
    H_dV =   np.array([np.insert(fit.dV_Unp, 3, alphapV_H) ])
    H_dbar = np.array([np.insert(fit.dbar_Unp, 3, alphapS_H) ])
    H_g =    np.array([np.insert(fit.g_Unp, 3, alphapS_H) ])

    # Initial xi^2 parameters for the H of (uV, ubar, dV, dbar,g) distributions
    """
        One free parameter R_H_xi2 for the ratio to the forward PDF H for u, d ang g
    """

    H_uV_xi2 =   np.einsum('...i,i->...i', H_uV,   [R_H_xi2,1,1,1])
    H_ubar_xi2 = np.einsum('...i,i->...i', H_ubar, [R_H_xi2,1,1,1])
    H_dV_xi2 =   np.einsum('...i,i->...i', H_dV,   [R_H_xi2,1,1,1])
    H_dbar_xi2 = np.einsum('...i,i->...i', H_dV,   [R_H_xi2,1,1,1])
    H_g_xi2 =    np.einsum('...i,i->...i', H_g,    [R_H_xi2,1,1,1])

    # Initial forward parameters for the E of (uV, ubar, dV, dbar,g) distributions
    """
        One free parameter R_E for the E/H ratio 
    """
    E_uV =   np.einsum('...i,i->...i', H_uV,   [R_E,1,1,1])
    E_ubar = np.einsum('...i,i->...i', H_ubar, [R_E,1,1,1])
    E_dV =   np.einsum('...i,i->...i', H_dV,   [R_E,1,1,1])
    E_dbar = np.einsum('...i,i->...i', H_dV,   [R_E,1,1,1])
    E_g =    np.einsum('...i,i->...i', H_g,    [R_E,1,1,1])

    # Initial xi^2 parameters for the E of (uV, ubar, dV, dbar,g) distributions
    """
        One free parameter R_E_xi2 for the ratio to the forward PDF E for u, d ang g
    """
    E_uV_xi2 =   np.einsum('...i,i->...i', E_uV,   [R_E_xi2,1,1,1])
    E_ubar_xi2 = np.einsum('...i,i->...i', E_ubar,   [R_E_xi2,1,1,1])
    E_dV_xi2 =   np.einsum('...i,i->...i', E_dV,   [R_E_xi2,1,1,1])
    E_dbar_xi2 = np.einsum('...i,i->...i', E_dbar,   [R_E_xi2,1,1,1])
    E_g_xi2 =    np.einsum('...i,i->...i', E_g,   [R_E_xi2,1,1,1])

    # Initial forward parameters for the Ht of (uV, ubar, dV, dbar,g) distributions

    Ht_uV =   np.array([np.insert(fit.uV_Pol, 3, alphapV_Ht) ])
    Ht_ubar = np.array([np.insert(fit.qbar_Pol, 3, alphapS_Ht) ])
    Ht_dV =   np.array([np.insert(fit.dV_Pol, 3, alphapV_Ht) ])
    Ht_dbar = np.array([np.insert(fit.qbar_Pol, 3, alphapS_Ht) ])
    Ht_g =    np.array([np.insert(fit.g_Pol, 3, alphapS_Ht) ])

    # Initial xi^2 parameters for the Ht of (uV, ubar, dV, dbar,g) distributions
    """
        One free parameter R_Ht_xi2 for the ratio to the forward PDF H for u, d ang g
    """
    Ht_uV_xi2 =   np.einsum('...i,i->...i', Ht_uV,   [R_Ht_xi2,1,1,1])
    Ht_ubar_xi2 = np.einsum('...i,i->...i', Ht_ubar, [R_Ht_xi2,1,1,1])
    Ht_dV_xi2 =   np.einsum('...i,i->...i', Ht_dV,   [R_Ht_xi2,1,1,1])
    Ht_dbar_xi2 = np.einsum('...i,i->...i', Ht_dbar, [R_Ht_xi2,1,1,1])
    Ht_g_xi2 =    np.einsum('...i,i->...i', Ht_g,    [R_Ht_xi2,1,1,1])

    # Initial forward parameters for the Et of (uV, ubar, dV, dbar,g) distributions
    """
        One free parameter R_Et for the Et/Ht ratio 
    """
    Et_uV =   np.einsum('...i,i->...i', Ht_uV,   [R_Et,1,1,1])
    Et_ubar = np.einsum('...i,i->...i', Ht_ubar, [R_Et,1,1,1])
    Et_dV =   np.einsum('...i,i->...i', Ht_dV,   [R_Et,1,1,1])
    Et_dbar = np.einsum('...i,i->...i', Ht_dbar, [R_Et,1,1,1])
    Et_g =    np.einsum('...i,i->...i', Ht_g,    [R_Et,1,1,1])

    # Initial xi^2 parameters for the Et of (uV, ubar, dV, dbar,g) distributions
    """
        One free parameter R_Et_xi2 for the ratio to the forward PDF Et for u, d ang g
    """
    Et_uV_xi2 =   np.einsum('...i,i->...i', Et_uV,   [R_Et_xi2,1,1,1])
    Et_ubar_xi2 = np.einsum('...i,i->...i', Et_ubar, [R_Et_xi2,1,1,1])
    Et_dV_xi2 =   np.einsum('...i,i->...i', Et_dV,   [R_Et_xi2,1,1,1])
    Et_dbar_xi2 = np.einsum('...i,i->...i', Et_dbar, [R_Et_xi2,1,1,1])
    Et_g_xi2 =    np.einsum('...i,i->...i', Et_g,    [R_Et_xi2,1,1,1])

    Hlst = np.array([[H_uV,     H_ubar,     H_dV,     H_dbar,     H_g],
                     [H_uV_xi2, H_ubar_xi2, H_dV_xi2, H_dbar_xi2, H_g_xi2]])
    
    Elst = np.array([[E_uV,     E_ubar,     E_dV,     E_dbar,     E_g],
                     [E_uV_xi2, E_ubar_xi2, E_dV_xi2, E_dbar_xi2, E_g_xi2]])

    Htlst = np.array([[Ht_uV,     Ht_ubar,     Ht_dV,     Ht_dbar,     Ht_g],
                      [Ht_uV_xi2, Ht_ubar_xi2, Ht_dV_xi2, Ht_dbar_xi2, Ht_g_xi2]])
    
    Etlst = np.array([[Et_uV,     Et_ubar,     Et_dV,     Et_dbar,     Et_g],
                      [Et_uV_xi2, Et_ubar_xi2, Et_dV_xi2, Et_dbar_xi2, Et_g_xi2]])

    return np.array([Hlst, Elst, Htlst, Etlst])