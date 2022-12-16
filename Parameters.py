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
# A factor of 3 including the xi^0, xi^2, xi^4 terms
xi2_Factor = 3
# Total number of parameters 
Tot_param_Size = NumofGPDSpecies * xi2_Factor * Flavor_Factor *  init_NumofAnsatz * Single_Param_Size

"""
The parameters will form a 5-dimensional matrix such that each para[#1,#2,#3,#4,#5] is a real number.
#1 = [0,1,2,3] corresponds to [H, E, Ht, Et]
#2 = [0,1] corresponds to [xi^0 terms, xi^2 terms]
#3 = [0,1,2,3,4] corresponds to [u - ubar, ubar, d - dbar, dbar, g]
#4 = [0,1,...,init_NumofAnsatz-1] corresponds to different set of parameters
#5 = [0,1,2,3] correspond to [norm, alpha, beta, alphap] as a set of parameters
"""

import numpy as np

def ParaManager_Unp(Paralst: np.array):

    """
     Here is the parameters manager, as there are over 100 free parameters. Therefore not all of them can be set free.
     Each element F_{q} is a two-dimensional matrix with init_NumofAnsatz = 1 row and Single_Param_Size = 4 columns
    """
    [Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
     Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
     Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
     Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar, 
     Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg,
     Norm_EuV,    alpha_EuV,    beta_EuV,    alphap_EuV,
     Norm_EdV,    R_E_Sea,      R_Hu_xi2,    R_Hd_xi2,    R_Hg_xi2,
     R_Eu_xi2,    R_Ed_xi2,     R_Eg_xi2,
     R_Hu_xi4,    R_Hd_xi4,     R_Hg_xi4,
     R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea] = Paralst

    R_Hu_xi4 = 0
    R_Hd_xi4 = 0
    R_Eu_xi4 = 0
    R_Ed_xi4 = 0

    R_Hg_xi2 = 0
    R_Eg_xi2 = 0
    R_Hg_xi4 = 0
    R_Eg_xi4 = 0

    # Initial forward parameters for the H of (uV, ubar, dV, dbar,g) distributions
    H_uV =   np.array([[Norm_HuV,   alpha_HuV,   beta_HuV,   alphap_HuV,   0]])
    H_ubar = np.array([[Norm_Hubar, alpha_Hubar, beta_Hubar, alphap_Hqbar, bexp_HSea]])
    H_dV =   np.array([[Norm_HdV,   alpha_HdV,   beta_HdV,   alphap_HdV,   0]])
    H_dbar = np.array([[Norm_Hdbar, alpha_Hdbar, beta_Hdbar, alphap_Hqbar, bexp_HSea]])
    H_g =    np.array([[Norm_Hg,    alpha_Hg,    beta_Hg,    alphap_Hg,    bexp_HSea ]])

    # Initial xi^2 parameters for the H of (uV, ubar, dV, dbar,g) distributions
    """
        One free parameter R_H_xi2 for the ratio to the forward PDF H for u, d ang g
    """

    H_uV_xi2 =   np.einsum('...i,i->...i', H_uV,   [R_Hu_xi2,1,1,1,1])
    H_ubar_xi2 = np.einsum('...i,i->...i', H_ubar, [R_Hu_xi2,1,1,1,1])
    H_dV_xi2 =   np.einsum('...i,i->...i', H_dV,   [R_Hd_xi2,1,1,1,1])
    H_dbar_xi2 = np.einsum('...i,i->...i', H_dbar, [R_Hd_xi2,1,1,1,1])
    H_g_xi2 =    np.einsum('...i,i->...i', H_g,    [R_Hg_xi2,1,1,1,1])

    H_uV_xi4 =   np.einsum('...i,i->...i', H_uV,   [R_Hu_xi4,1,1,1,1])
    H_ubar_xi4 = np.einsum('...i,i->...i', H_ubar, [R_Hu_xi4,1,1,1,1])
    H_dV_xi4 =   np.einsum('...i,i->...i', H_dV,   [R_Hd_xi4,1,1,1,1])
    H_dbar_xi4 = np.einsum('...i,i->...i', H_dbar, [R_Hd_xi4,1,1,1,1])
    H_g_xi4 =    np.einsum('...i,i->...i', H_g,    [R_Hg_xi4,1,1,1,1])

    # Initial forward parameters for the E of (uV, ubar, dV, dbar,g) distributions
    """
        Three free parameter R_E_u, R_E_d, R_E_g for the E/H ratio 
    """
    E_uV =   np.array([[Norm_EuV,   alpha_EuV,   beta_EuV,   alphap_EuV, 0]])
    E_ubar = np.einsum('...i,i->...i', H_ubar,   [R_E_Sea,1,1,1,1])
    E_dV =   np.array([[Norm_EdV,   alpha_EuV,   beta_EuV,   alphap_EuV, 0]])
    E_dbar = np.einsum('...i,i->...i', H_dbar,   [R_E_Sea,1,1,1,1])
    E_g =    np.einsum('...i,i->...i', H_g,      [R_E_Sea,1,1,1,1])

    # Initial xi^2 parameters for the E of (uV, ubar, dV, dbar,g) distributions
    """
        One free parameter R_E_xi2 for the ratio to the forward PDF E
    """
    E_uV_xi2 =   np.einsum('...i,i->...i', E_uV,     [R_Eu_xi2,1,1,1,1])
    E_ubar_xi2 = np.einsum('...i,i->...i', E_ubar,   [R_Eu_xi2,1,1,1,1])
    E_dV_xi2 =   np.einsum('...i,i->...i', E_dV,     [R_Ed_xi2,1,1,1,1])
    E_dbar_xi2 = np.einsum('...i,i->...i', E_dbar,   [R_Ed_xi2,1,1,1,1])
    E_g_xi2 =    np.einsum('...i,i->...i', E_g,      [R_Eg_xi2,1,1,1,1])

    E_uV_xi4 =   np.einsum('...i,i->...i', E_uV,     [R_Eu_xi4,1,1,1,1])
    E_ubar_xi4 = np.einsum('...i,i->...i', E_ubar,   [R_Eu_xi4,1,1,1,1])
    E_dV_xi4 =   np.einsum('...i,i->...i', E_dV,     [R_Ed_xi4,1,1,1,1])
    E_dbar_xi4 = np.einsum('...i,i->...i', E_dbar,   [R_Ed_xi4,1,1,1,1])
    E_g_xi4 =    np.einsum('...i,i->...i', E_g,      [R_Eg_xi4,1,1,1,1])

    Hlst = np.array([[H_uV,     H_ubar,     H_dV,     H_dbar,     H_g],
                     [H_uV_xi2, H_ubar_xi2, H_dV_xi2, H_dbar_xi2, H_g_xi2],
                     [H_uV_xi4, H_ubar_xi4, H_dV_xi4, H_dbar_xi4, H_g_xi4]])
    
    Elst = np.array([[E_uV,     E_ubar,     E_dV,     E_dbar,     E_g],
                     [E_uV_xi2, E_ubar_xi2, E_dV_xi2, E_dbar_xi2, E_g_xi2],
                     [E_uV_xi4, E_ubar_xi4, E_dV_xi4, E_dbar_xi4, E_g_xi4]])

    return np.array([Hlst, Elst])

def ParaManager_Pol(Paralst: np.array):

    [Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV, 
     Norm_Htubar, alpha_Htubar, beta_Htubar, alphap_Htqbar,
     Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,
     Norm_Htdbar, alpha_Htdbar, beta_Htdbar, 
     Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_Htg,
     Norm_EtuV,   alpha_EtuV,   beta_EtuV,   alphap_EtuV,
     Norm_EtdV,   R_Et_Sea,     R_Htu_xi2,   R_Htd_xi2,    R_Htg_xi2,
     R_Etu_xi2,   R_Etd_xi2,    R_Etg_xi2,
     R_Htu_xi4,   R_Htd_xi4,    R_Htg_xi4,
     R_Etu_xi4,   R_Etd_xi4,    R_Etg_xi4,   bexp_HtSea] = Paralst

    R_Htu_xi4 = 0
    R_Htd_xi4 = 0
    R_Etu_xi4 = 0
    R_Etd_xi4 = 0
    
    R_Htg_xi2 = 0
    R_Etg_xi2 = 0
    R_Htg_xi4 = 0
    R_Etg_xi4 = 0

    # Initial forward parameters for the Ht of (uV, ubar, dV, dbar,g) distributions

    Ht_uV =   np.array([[Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV,   0]])
    Ht_ubar = np.array([[Norm_Htubar, alpha_Htubar, beta_Htubar, alphap_Htqbar, bexp_HtSea]])
    Ht_dV =   np.array([[Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,   0]])
    Ht_dbar = np.array([[Norm_Htdbar, alpha_Htdbar, beta_Htdbar, alphap_Htqbar, bexp_HtSea]])
    Ht_g =    np.array([[Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_Htg,    bexp_HtSea]])

    # Initial xi^2 parameters for the Ht of (uV, ubar, dV, dbar,g) distributions
    """
        One free parameter R_Ht_xi2 for the ratio to the forward PDF H for u, d ang g
    """
    Ht_uV_xi2 =   np.einsum('...i,i->...i', Ht_uV,   [R_Htu_xi2,1,1,1,1])
    Ht_ubar_xi2 = np.einsum('...i,i->...i', Ht_ubar, [R_Htu_xi2,1,1,1,1])
    Ht_dV_xi2 =   np.einsum('...i,i->...i', Ht_dV,   [R_Htd_xi2,1,1,1,1])
    Ht_dbar_xi2 = np.einsum('...i,i->...i', Ht_dbar, [R_Htd_xi2,1,1,1,1])
    Ht_g_xi2 =    np.einsum('...i,i->...i', Ht_g,    [R_Htg_xi2,1,1,1,1])

    Ht_uV_xi4 =   np.einsum('...i,i->...i', Ht_uV,   [R_Htu_xi4,1,1,1,1])
    Ht_ubar_xi4 = np.einsum('...i,i->...i', Ht_ubar, [R_Htu_xi4,1,1,1,1])
    Ht_dV_xi4 =   np.einsum('...i,i->...i', Ht_dV,   [R_Htd_xi4,1,1,1,1])
    Ht_dbar_xi4 = np.einsum('...i,i->...i', Ht_dbar, [R_Htd_xi4,1,1,1,1])
    Ht_g_xi4 =    np.einsum('...i,i->...i', Ht_g,    [R_Htg_xi4,1,1,1,1])

    # Initial forward parameters for the Et of (uV, ubar, dV, dbar,g) distributions
    """
        Three free parameter R_Et_u, R_Et_d, R_E_g for the Et/Ht ratio 
    """
    Et_uV =   np.array([[Norm_EtuV,   alpha_EtuV,   beta_EtuV,   alphap_EtuV, 0]])
    Et_ubar = np.einsum('...i,i->...i', Ht_ubar, [R_Et_Sea,1,1,1,1])
    Et_dV =   np.array([[Norm_EtdV,   alpha_EtuV,   beta_EtuV,   alphap_EtuV, 0]])
    Et_dbar = np.einsum('...i,i->...i', Ht_dbar, [R_Et_Sea,1,1,1,1])
    Et_g =    np.einsum('...i,i->...i', Ht_g,    [R_Et_Sea,1,1,1,1])

    # Initial xi^2 parameters for the Et of (uV, ubar, dV, dbar,g) distributions
    """
        One free parameter R_Et_xi2 for the ratio to the forward PDF Et for u, d ang g
    """
    Et_uV_xi2 =   np.einsum('...i,i->...i', Et_uV,   [R_Etu_xi2,1,1,1,1])
    Et_ubar_xi2 = np.einsum('...i,i->...i', Et_ubar, [R_Etu_xi2,1,1,1,1])
    Et_dV_xi2 =   np.einsum('...i,i->...i', Et_dV,   [R_Etd_xi2,1,1,1,1])
    Et_dbar_xi2 = np.einsum('...i,i->...i', Et_dbar, [R_Etd_xi2,1,1,1,1])
    Et_g_xi2 =    np.einsum('...i,i->...i', Et_g,    [R_Etg_xi2,1,1,1,1])

    Et_uV_xi4 =   np.einsum('...i,i->...i', Et_uV,   [R_Etu_xi4,1,1,1,1])
    Et_ubar_xi4 = np.einsum('...i,i->...i', Et_ubar, [R_Etu_xi4,1,1,1,1])
    Et_dV_xi4 =   np.einsum('...i,i->...i', Et_dV,   [R_Etd_xi4,1,1,1,1])
    Et_dbar_xi4 = np.einsum('...i,i->...i', Et_dbar, [R_Etd_xi4,1,1,1,1])
    Et_g_xi4 =    np.einsum('...i,i->...i', Et_g,    [R_Etg_xi4,1,1,1,1])

    Htlst = np.array([[Ht_uV,     Ht_ubar,     Ht_dV,     Ht_dbar,     Ht_g],
                      [Ht_uV_xi2, Ht_ubar_xi2, Ht_dV_xi2, Ht_dbar_xi2, Ht_g_xi2],
                      [Ht_uV_xi4, Ht_ubar_xi4, Ht_dV_xi4, Ht_dbar_xi4, Ht_g_xi4]])
    
    Etlst = np.array([[Et_uV,     Et_ubar,     Et_dV,     Et_dbar,     Et_g],
                      [Et_uV_xi2, Et_ubar_xi2, Et_dV_xi2, Et_dbar_xi2, Et_g_xi2],
                      [Et_uV_xi4, Et_ubar_xi4, Et_dV_xi4, Et_dbar_xi4, Et_g_xi4]])

    return np.array([Htlst, Etlst])