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

import numpy as np
import scipy as sp

def ParaManager_Unp(Paralst: np.array):
    """Unpolairzed parameters manager that turns all parameters as a list to the wanted form.
    
    | Most minimizer take regular form of inputs (list or tuple).
    | However, it's better if we convert these input parameters to given shape.
    | Here we convert the function to a 5-dimensional array with shape (2,3,5,n1,n2)
    | Each rows means:
    |        #1 = [0,1] corresponds to [H, E]
    |        #2 = [0,1,2,...] corresponds to [xi^0 terms, xi^2 terms, xi^4 terms, ...]
    |        #3 = [0,1,2,3,4] corresponds to [u - ubar, ubar, d - dbar, dbar, g]
    |        #4 = [0,1,...,init_NumofAnsatz-1] corresponds to different set of parameters
    |        #5 = [0,1,2,3,...] correspond to [norm, alpha, beta, alphap,...] as a set of parameters
        
    Args:
        Paralst (np.array): list of all parameters

    Returns:
        Paralunp (np.ndrray): shape (2,3,5,n1,n2) 
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
     R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea, bexp_Hg, Invm2_Hg] = Paralst
    
    #R_E_Sea = 0
    #R_Hu_xi4 = 0
    #R_Hd_xi4 = 0
    #R_Eu_xi4 = 0
    #R_Ed_xi4 = 0

    # Initial forward parameters for the H of (uV, ubar, dV, dbar,g) distributions
    H_uV =   np.array([[Norm_HuV,   alpha_HuV,   beta_HuV,   alphap_HuV,   0,         0]])
    H_ubar = np.array([[Norm_Hubar, alpha_Hubar, beta_Hubar, alphap_Hqbar, bexp_HSea, 0]])
    H_dV =   np.array([[Norm_HdV,   alpha_HdV,   beta_HdV,   alphap_HdV,   0,         0]])
    H_dbar = np.array([[Norm_Hdbar, alpha_Hdbar, beta_Hdbar, alphap_Hqbar, bexp_HSea, 0]])
    H_g =    np.array([[Norm_Hg,    alpha_Hg,    beta_Hg,    alphap_Hg,    bexp_Hg,   Invm2_Hg]])

    # Initial xi^2 parameters for the H of (uV, ubar, dV, dbar,g) distributions
    """
        One free parameter R_H_xi2 for the ratio to the forward PDF H for u, d ang g
    """

    H_uV_xi2 =   np.einsum('...i,i->...i', H_uV,   [R_Hu_xi2,1,1,1,1,1])
    H_ubar_xi2 = np.einsum('...i,i->...i', H_ubar, [R_Hu_xi2,1,1,1,1,1])
    H_dV_xi2 =   np.einsum('...i,i->...i', H_dV,   [R_Hd_xi2,1,1,1,1,1])
    H_dbar_xi2 = np.einsum('...i,i->...i', H_dbar, [R_Hd_xi2,1,1,1,1,1])
    H_g_xi2 =    np.einsum('...i,i->...i', H_g,    [R_Hg_xi2,1,1,1,1,1])

    H_uV_xi4 =   np.einsum('...i,i->...i', H_uV,   [R_Hu_xi4,1,1,1,1,1])
    H_ubar_xi4 = np.einsum('...i,i->...i', H_ubar, [R_Hu_xi4,1,1,1,1,1])
    H_dV_xi4 =   np.einsum('...i,i->...i', H_dV,   [R_Hd_xi4,1,1,1,1,1])
    H_dbar_xi4 = np.einsum('...i,i->...i', H_dbar, [R_Hd_xi4,1,1,1,1,1])
    H_g_xi4 =    np.einsum('...i,i->...i', H_g,    [R_Hg_xi4,1,1,1,1,1])

    # Initial forward parameters for the E of (uV, ubar, dV, dbar,g) distributions
    """
        Three free parameter R_E_u, R_E_d, R_E_g for the E/H ratio 
    """
    E_uV =   np.array([[Norm_EuV,   alpha_EuV,   beta_EuV,   alphap_EuV, 0, 0]])
    E_ubar = np.einsum('...i,i->...i', H_ubar,   [R_E_Sea,1,1,1,1,1])
    E_dV =   np.array([[Norm_EdV,   alpha_EuV,   beta_EuV,   alphap_EuV, 0, 0]])
    E_dbar = np.einsum('...i,i->...i', H_dbar,   [R_E_Sea,1,1,1,1,1])
    E_g =    np.einsum('...i,i->...i', H_g,      [R_E_Sea,1,1,1,1,1])

    # Initial xi^2 parameters for the E of (uV, ubar, dV, dbar,g) distributions
    """
        One free parameter R_E_xi2 for the ratio to the forward PDF E
    """
    E_uV_xi2 =   np.einsum('...i,i->...i', E_uV,     [R_Eu_xi2,1,1,1,1,1])
    E_ubar_xi2 = np.einsum('...i,i->...i', E_ubar,   [R_Eu_xi2,1,1,1,1,1])
    E_dV_xi2 =   np.einsum('...i,i->...i', E_dV,     [R_Ed_xi2,1,1,1,1,1])
    E_dbar_xi2 = np.einsum('...i,i->...i', E_dbar,   [R_Ed_xi2,1,1,1,1,1])
    E_g_xi2 =    np.einsum('...i,i->...i', E_g,      [R_Eg_xi2,1,1,1,1,1])

    E_uV_xi4 =   np.einsum('...i,i->...i', E_uV,     [R_Eu_xi4,1,1,1,1,1])
    E_ubar_xi4 = np.einsum('...i,i->...i', E_ubar,   [R_Eu_xi4,1,1,1,1,1])
    E_dV_xi4 =   np.einsum('...i,i->...i', E_dV,     [R_Ed_xi4,1,1,1,1,1])
    E_dbar_xi4 = np.einsum('...i,i->...i', E_dbar,   [R_Ed_xi4,1,1,1,1,1])
    E_g_xi4 =    np.einsum('...i,i->...i', E_g,      [R_Eg_xi4,1,1,1,1,1])

    Hlst = np.array([[H_uV,     H_ubar,     H_dV,     H_dbar,     H_g],
                     [H_uV_xi2, H_ubar_xi2, H_dV_xi2, H_dbar_xi2, H_g_xi2],
                     [H_uV_xi4, H_ubar_xi4, H_dV_xi4, H_dbar_xi4, H_g_xi4]])
    
    Elst = np.array([[E_uV,     E_ubar,     E_dV,     E_dbar,     E_g],
                     [E_uV_xi2, E_ubar_xi2, E_dV_xi2, E_dbar_xi2, E_g_xi2],
                     [E_uV_xi4, E_ubar_xi4, E_dV_xi4, E_dbar_xi4, E_g_xi4]])

    return np.array([Hlst, Elst])

def ParaManager_Pol(Paralst: np.array):
    """Polarized parameters manager that turns all parameters as a list to the wanted form.
    
    | Most minimizer take regular form of inputs (list or tuple).
    | However, it's better if we convert these input parameters to given shape.
    | Here we convert the function to a 5-dimensional array with shape (2,3,5,n1,n2)
    | Each rows means:
    |        #1 = [0,1] corresponds to [Ht, Et]
    |        #2 = [0,1,2,...] corresponds to [xi^0 terms, xi^2 terms, xi^4 terms, ...]
    |        #3 = [0,1,2,3,4] corresponds to [u - ubar, ubar, d - dbar, dbar, g]
    |        #4 = [0,1,...,init_NumofAnsatz-1] corresponds to different set of parameters
    |        #5 = [0,1,2,3,...] correspond to [norm, alpha, beta, alphap,...] as a set of parameters
        
    Args:
        Paralst (np.array): list of all parameters

    Returns:
        Paralunp (np.ndrray): shape (2,3,5,n1,n2) 
    """
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

    Ht_uV =   np.array([[Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV,   0,          0]])
    Ht_ubar = np.array([[Norm_Htubar, alpha_Htubar, beta_Htubar, alphap_Htqbar, bexp_HtSea, 0]])
    Ht_dV =   np.array([[Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,   0,          0]])
    Ht_dbar = np.array([[Norm_Htdbar, alpha_Htdbar, beta_Htdbar, alphap_Htqbar, bexp_HtSea, 0]])
    Ht_g =    np.array([[Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_Htg,    bexp_HtSea, 0]])

    # Initial xi^2 parameters for the Ht of (uV, ubar, dV, dbar,g) distributions
    """
        One free parameter R_Ht_xi2 for the ratio to the forward PDF H for u, d ang g
    """
    Ht_uV_xi2 =   np.einsum('...i,i->...i', Ht_uV,   [R_Htu_xi2,1,1,1,1,1])
    Ht_ubar_xi2 = np.einsum('...i,i->...i', Ht_ubar, [R_Htu_xi2,1,1,1,1,1])
    Ht_dV_xi2 =   np.einsum('...i,i->...i', Ht_dV,   [R_Htd_xi2,1,1,1,1,1])
    Ht_dbar_xi2 = np.einsum('...i,i->...i', Ht_dbar, [R_Htd_xi2,1,1,1,1,1])
    Ht_g_xi2 =    np.einsum('...i,i->...i', Ht_g,    [R_Htg_xi2,1,1,1,1,1])

    Ht_uV_xi4 =   np.einsum('...i,i->...i', Ht_uV,   [R_Htu_xi4,1,1,1,1,1])
    Ht_ubar_xi4 = np.einsum('...i,i->...i', Ht_ubar, [R_Htu_xi4,1,1,1,1,1])
    Ht_dV_xi4 =   np.einsum('...i,i->...i', Ht_dV,   [R_Htd_xi4,1,1,1,1,1])
    Ht_dbar_xi4 = np.einsum('...i,i->...i', Ht_dbar, [R_Htd_xi4,1,1,1,1,1])
    Ht_g_xi4 =    np.einsum('...i,i->...i', Ht_g,    [R_Htg_xi4,1,1,1,1,1])

    # Initial forward parameters for the Et of (uV, ubar, dV, dbar,g) distributions
    """
        Three free parameter R_Et_u, R_Et_d, R_E_g for the Et/Ht ratio 
    """
    Et_uV =   np.array([[Norm_EtuV,   alpha_EtuV,   beta_EtuV,   alphap_EtuV, 0, 0]])
    Et_ubar = np.einsum('...i,i->...i', Ht_ubar, [R_Et_Sea,1,1,1,1,1])
    Et_dV =   np.array([[Norm_EtdV,   alpha_EtuV,   beta_EtuV,   alphap_EtuV, 0, 0]])
    Et_dbar = np.einsum('...i,i->...i', Ht_dbar, [R_Et_Sea,1,1,1,1,1])
    Et_g =    np.einsum('...i,i->...i', Ht_g,    [R_Et_Sea,1,1,1,1,1])

    # Initial xi^2 parameters for the Et of (uV, ubar, dV, dbar,g) distributions
    """
        One free parameter R_Et_xi2 for the ratio to the forward PDF Et for u, d ang g
    """
    Et_uV_xi2 =   np.einsum('...i,i->...i', Et_uV,   [R_Etu_xi2,1,1,1,1,1])
    Et_ubar_xi2 = np.einsum('...i,i->...i', Et_ubar, [R_Etu_xi2,1,1,1,1,1])
    Et_dV_xi2 =   np.einsum('...i,i->...i', Et_dV,   [R_Etd_xi2,1,1,1,1,1])
    Et_dbar_xi2 = np.einsum('...i,i->...i', Et_dbar, [R_Etd_xi2,1,1,1,1,1])
    Et_g_xi2 =    np.einsum('...i,i->...i', Et_g,    [R_Etg_xi2,1,1,1,1,1])

    Et_uV_xi4 =   np.einsum('...i,i->...i', Et_uV,   [R_Etu_xi4,1,1,1,1,1])
    Et_ubar_xi4 = np.einsum('...i,i->...i', Et_ubar, [R_Etu_xi4,1,1,1,1,1])
    Et_dV_xi4 =   np.einsum('...i,i->...i', Et_dV,   [R_Etd_xi4,1,1,1,1,1])
    Et_dbar_xi4 = np.einsum('...i,i->...i', Et_dbar, [R_Etd_xi4,1,1,1,1,1])
    Et_g_xi4 =    np.einsum('...i,i->...i', Et_g,    [R_Etg_xi4,1,1,1,1,1])

    Htlst = np.array([[Ht_uV,     Ht_ubar,     Ht_dV,     Ht_dbar,     Ht_g],
                      [Ht_uV_xi2, Ht_ubar_xi2, Ht_dV_xi2, Ht_dbar_xi2, Ht_g_xi2],
                      [Ht_uV_xi4, Ht_ubar_xi4, Ht_dV_xi4, Ht_dbar_xi4, Ht_g_xi4]])
    
    Etlst = np.array([[Et_uV,     Et_ubar,     Et_dV,     Et_dbar,     Et_g],
                      [Et_uV_xi2, Et_ubar_xi2, Et_dV_xi2, Et_dbar_xi2, Et_g_xi2],
                      [Et_uV_xi4, Et_ubar_xi4, Et_dV_xi4, Et_dbar_xi4, Et_g_xi4]])

    return np.array([Htlst, Etlst])

# Euler Beta function B(a,b) with complex arguments
def beta_loggamma(a: complex, b: complex) -> complex:
    return np.exp(sp.special.loggamma(a) + sp.special.loggamma(b)-sp.special.loggamma(a + b))

# Conformal moment in j space F(j)
def ConfMoment(j: complex, t: float, ParaSets: np.ndarray):
    """Conformal moment in j space F(j)

    Args:
        j: conformal spin j (conformal spin is actually j+2 but anyway)
        t: momentum transfer squared t
        ParaSets: the array of parameters, explained below
        
    ParaSets: in the form of (norm, alpha, beta, alphap, bexp, invm, ....), 
        * norm: ParaSets[0], overall normalization constant
        * alpha & beta: ParaSets[1] & ParaSet[2], the two parameters corresponding to x ^ (-alpha) * (1 - x) ^ beta
        * alphap: ParaSets[3], regge trajectory alpha(t) = alpha + alphap * t
        * bexp: ParaSets[4], the residual term modeled with exp(bexp*t)
        * invm: ParaSets[5], the residual term modeled with (1 - t * invm2 ) ** (-3), alternative option of bexp
    
    Returns:
        Conformal moment in j space F(j,t) (...,5,init_NumofAnsatz)
    
    Note:
     |    ParaSets is a set of parameters having (...,5, init_NumofAnsatz, 6).
     |    So each parameter has shape (...,5,init_NumofAnsatz)
     |    We assume that the first few dimension of j and t and each parameter are broadcastable.
     |    So the return shape should be (...,5,init_NumofAnsatz)
        
    Recommended usage:
     |    j has shape (N), t has shape (), ParaSet has shape ( 5, init_NumofAnsatz, 6)
     |    or j has shape (N), t has shape (N), ParaSet has shape (N, 5, init_NumofAnsatz, 6)
     |    The first line is more common when integrate over j
    """
    
    # [norm, alpha, beta, alphap, bexp, invm2] = ParaSet
    norm = ParaSets[..., 0]  # in recommended usage, has shape (N, 5, init_NumofAnsatz)
    alpha = ParaSets[..., 1] # in general, can have shape (N), (N, m1), (N, m1, m2), ......
    beta  = ParaSets[..., 2]
    alphap = ParaSets[..., 3]
    bexp = ParaSets[..., 4]
    invm2 = ParaSets[..., 5]

    if np.ndim(norm) < np.ndim(t):
        raise ValueError("Input format is wrong.")
    
    t_new_shape = list(np.shape(t)) + [1]*(np.ndim(norm) - np.ndim(t))
    j_new_shape = list(np.shape(j)) + [1]*(np.ndim(norm) - np.ndim(t))  # not a typo, it is np.ndim(norm) - np.ndim(t)
    t = np.reshape(t, t_new_shape) # to make sure t can be broadcasted with norm, alpha, etc.
    # t will have shape (N) or (N, m1) or (N, m1, m2)... depends
    j = np.reshape(j, j_new_shape)

    # Currently with KM ansatz and dipole residual

    return norm * beta_loggamma (j + 1 - alpha, 1 + beta) * (j + 1  - alpha)/ (j + 1 - alpha - alphap * t) * np.exp(t*bexp) * (1 - t * invm2 ) ** (-3)
    # (N) or (N, m1) or (N, m1, m2) .... depends on usage
    # For the recommended usage, the output is (N, 5, init_NumofAnsatz)

def Moment_Sum(j: complex, t: float, ParaSets: np.ndarray) -> complex:
    """Sum of the conformal moments when the ParaSets contain more than just one set of parameters 

    Args:
        ParaSets : contains [ParaSet1, ParaSet0, ParaSet2,...] with each ParaSet = [norm, alpha, beta ,alphap,...] for valence and sea distributions repsectively.        
        j: conformal spin j (or j+2 but anyway)
        t: momentum transfer squared t

    Returns:
        sum of conformal moments over all the ParaSet
        
    Read more about this in the comments of ConfMoment():
    |    It will return shape (...,5,init_NumofAnsatz)
    |    This function simply sums over the last dimension init_NumofAnsatz.
    """
    
    # ConfMoment_vec(j, t, ParaSets) should have shape (N, 5, init_NumofAnsatz)
    return np.sum(ConfMoment(j, t, ParaSets) ,  axis=-1) # (N, 5)
