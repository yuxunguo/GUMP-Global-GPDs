r"""
Parameter management for the GUMP GPD fitting framework.

This module converts the flat 1-D parameter arrays expected by minimizers
(e.g. iMinuit) into the structured multi-dimensional arrays consumed by the
conformal-moment calculation routines.

Parameter array layout
----------------------
Each GPD species is parameterized per flavor as a set of ansatz terms.  A
single ansatz term contains six numbers::

    [norm, alpha, beta, alphap, bexp, invm2]

The structured output of :func:`ParaManager_Unp` and :func:`ParaManager_Pol`
has shape ``(2, 3, 5, n_ansatz, 6)`` where the axes correspond to:

* axis 0 – GPD pair: ``[H, E]`` or ``[\~H, \~E]``
* axis 1 – skewness power: ``[xi^0, xi^2, xi^4]`` terms
* axis 2 – flavor: ``[u_V, u-bar, d_V, d-bar, g]``
* axis 3 – ansatz index
* axis 4 – parameter within one ansatz: ``[norm, alpha, beta, alphap, bexp, invm2]``
"""
# Number of GPD species, 4 leading-twist GPDs including H, E Ht, Et are needed.
#NumofGPDSpecies = 4
# Number of flavor factor, Flavor_Factor = 2 * nf + 1 needed including 2 * nf quark (antiquark) and one gluon
#Flavor_Factor = 2 * 2 + 1
# Number of ansatz, 1 set of (N, alpha, beta, alphap) will be used to start with
#init_NumofAnsatz = 1
# Size of one parameter set, a set of parameters (N, alpha, beta, alphap) contain 4 parameters
#Single_Param_Size = 4
# A factor of 3 including the xi^0, xi^2, xi^4 terms
#xi2_Factor = 3
# Total number of parameters 
#Tot_param_Size = NumofGPDSpecies * xi2_Factor * Flavor_Factor *  init_NumofAnsatz * Single_Param_Size

import numpy as np
import scipy as sp

def ParaManager_Unp(Paralst: np.ndarray) -> np.ndarray:
    """Convert a flat unpolarized parameter list into a structured array.

    Most minimizers (e.g. iMinuit) pass parameters as a flat list or tuple.
    This function reshapes that flat list into a 5-D array of shape
    ``(2, 3, 5, n_ansatz, 6)`` whose axes are:

    * axis 0 – GPD: ``[H, E]``
    * axis 1 – skewness power: ``[xi^0, xi^2, xi^4]``
    * axis 2 – flavor: ``[u_V, u-bar, d_V, d-bar, g]``
    * axis 3 – ansatz index
    * axis 4 – parameters per ansatz: ``[norm, alpha, beta, alphap, bexp, invm2]``

    Args:
        Paralst (np.ndarray): flat 1-D array of all unpolarized fit parameters
            in the order defined by the destructuring assignment inside this
            function.

    Returns:
        np.ndarray: structured parameter array of shape ``(2, 3, 5, n_ansatz, 6)``
        containing ``[Hlst, Elst]``.
    """""

    [Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, Invm2_HuV,
     Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
     Norm_Hubar_2,  alpha_Hubar_2,  beta_Hubar_2,
     Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV, Invm2_HdV,
     Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar, 
     Norm_Hdbar_2,  alpha_Hdbar_2,  beta_Hdbar_2,
     Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg, Invm2_Hg,
     Norm_Hg_2,     alpha_Hg_2,     beta_Hg_2,
     Norm_EuV,    alpha_EuV,    beta_EuV,    alphap_EuV,
     Norm_EdV,    alpha_EdV,    beta_EdV,    alphap_EdV,
     R_E_ubar,    R_E_dbar,    R_E_g,
     R_Hu_xi2,    R_Hd_xi2,    R_Hg_xi2,
     R_Eu_xi2,    R_Ed_xi2,     R_Eg_xi2,
     R_Hu_xi4,    R_Hd_xi4,     R_Hg_xi4,
     R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea, bexp_Hg] = Paralst
    
    #R_E_Sea = 0
    #R_Hu_xi4 = 0
    #R_Hd_xi4 = 0
    #R_Eu_xi4 = 0
    #R_Ed_xi4 = 0
    # Ansatz_Place_Holder here is used such that it doesn't contribute to the moments, but serve a place holder 
    # To activate the second (or more) ansatz but only for gluons, we use place holder to keep the shape regular.
    Ansatz_Place_Holder = [0,0,1,0,0,0]
    # Initial forward parameters for the H of (uV, ubar, dV, dbar,g) distributions
    H_uV =   np.array([[Norm_HuV,   alpha_HuV,   beta_HuV,   alphap_HuV,   0,         Invm2_HuV], Ansatz_Place_Holder])
    H_ubar = np.array([[Norm_Hubar, alpha_Hubar, beta_Hubar, alphap_Hqbar, bexp_HSea, 0        ], [Norm_Hubar_2, alpha_Hubar_2, beta_Hubar_2, alphap_Hqbar, bexp_HSea, 0]])
    H_dV =   np.array([[Norm_HdV,   alpha_HdV,   beta_HdV,   alphap_HdV,   0,         Invm2_HdV], Ansatz_Place_Holder])
    H_dbar = np.array([[Norm_Hdbar, alpha_Hdbar, beta_Hdbar, alphap_Hqbar, bexp_HSea, 0        ], [Norm_Hdbar_2, alpha_Hdbar_2, beta_Hdbar_2, alphap_Hqbar, bexp_HSea, 0]])
    H_g =    np.array([[Norm_Hg,    alpha_Hg,    beta_Hg,    alphap_Hg,    bexp_Hg,   0 ], [Norm_Hg_2,    alpha_Hg_2,    beta_Hg_2,    alphap_Hg,    0,   Invm2_Hg]])

    # xi^2 prefactor: only the normalization column is rescaled; all other
    # columns (alpha, beta, alphap, bexp, invm2) inherit from the xi^0 ansatz.
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

    # E is parameterized with independent valence parameters and a sea/gluon
    # ratio R_E_{flavor} relative to the corresponding H ansatz.
    E_uV =   np.array([[Norm_EuV,   alpha_EuV,   beta_EuV,   alphap_EuV, 0, 0], Ansatz_Place_Holder])
    E_ubar = np.einsum('...i,i->...i', H_ubar,   [R_E_ubar,1,1,1,1,1])
    E_dV =   np.array([[Norm_EdV,   alpha_EdV,   beta_EdV,   alphap_EdV, 0, 0], Ansatz_Place_Holder])
    E_dbar = np.einsum('...i,i->...i', H_dbar,   [R_E_dbar,1,1,1,1,1])
    E_g =    np.einsum('...i,i->...i', H_g,      [R_E_g,1,1,1,1,1])

    # xi^2 prefactor for E: only normalization is rescaled.
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

def ParaManager_Pol(Paralst: np.ndarray) -> np.ndarray:
    """Convert a flat polarized parameter list into a structured array.

    Analogous to :func:`ParaManager_Unp` but for the helicity GPDs
    ``~H`` (``Ht``) and ``~E`` (``Et``).

    The output has shape ``(2, 3, 5, n_ansatz, 6)`` whose axes are:

    * axis 0 – GPD: ``[~H, ~E]``
    * axis 1 – skewness power: ``[xi^0, xi^2, xi^4]``
    * axis 2 – flavor: ``[u_V, u-bar, d_V, d-bar, g]``
    * axis 3 – ansatz index
    * axis 4 – parameters per ansatz: ``[norm, alpha, beta, alphap, bexp, invm2]``

    Args:
        Paralst (np.ndarray): flat 1-D array of all polarized fit parameters
            in the order defined by the destructuring assignment inside this
            function.

    Returns:
        np.ndarray: structured parameter array of shape ``(2, 3, 5, n_ansatz, 6)``
        containing ``[Htlst, Etlst]``.
    """""
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
    
    # Ansatz_Place_Holder here is used such that it doesn't contribute to the moments, but serve a place holder 
    # To activate the second (or more) ansatz but only for gluons, we use place holder to keep the shape regular.
    Ansatz_Place_Holder = [0,0,1,0,0,0]
    # Initial forward parameters for the Ht of (uV, ubar, dV, dbar,g) distributions

    Ht_uV =   np.array([[Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV,   0,          0], Ansatz_Place_Holder])
    Ht_ubar = np.array([[Norm_Htubar, alpha_Htubar, beta_Htubar, alphap_Htqbar, bexp_HtSea, 0], Ansatz_Place_Holder])
    Ht_dV =   np.array([[Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,   0,          0], Ansatz_Place_Holder])
    Ht_dbar = np.array([[Norm_Htdbar, alpha_Htdbar, beta_Htdbar, alphap_Htqbar, bexp_HtSea, 0], Ansatz_Place_Holder])
    Ht_g =    np.array([[Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_Htg,    bexp_HtSea, 0], Ansatz_Place_Holder])

    # xi^2 prefactor for ~H: only normalization is rescaled.
    # Note: gluon and higher-xi contributions for ~H and ~E are currently
    # fixed to zero (R_Htg_xi2 = R_Etg_xi2 = R_Htg_xi4 = R_Etg_xi4 = 0).
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

    # ~E is parameterized with independent valence parameters and a common
    # sea/gluon ratio R_Et_Sea relative to the corresponding ~H ansatz.
    Et_uV =   np.array([[Norm_EtuV,   alpha_EtuV,   beta_EtuV,   alphap_EtuV, 0, 0], Ansatz_Place_Holder])
    Et_ubar = np.einsum('...i,i->...i', Ht_ubar, [R_Et_Sea,1,1,1,1,1])
    Et_dV =   np.array([[Norm_EtdV,   alpha_EtuV,   beta_EtuV,   alphap_EtuV, 0, 0], Ansatz_Place_Holder])
    Et_dbar = np.einsum('...i,i->...i', Ht_dbar, [R_Et_Sea,1,1,1,1,1])
    Et_g =    np.einsum('...i,i->...i', Ht_g,    [R_Et_Sea,1,1,1,1,1])

    # xi^2 prefactor for ~E: only normalization is rescaled.
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

def beta_loggamma(a: complex, b: complex) -> complex:
    """Euler Beta function evaluated via log-gamma for numerical stability.

    Computes :math:`B(a, b) = \\Gamma(a)\\Gamma(b) / \\Gamma(a+b)` using
    ``scipy.special.loggamma`` to handle complex arguments safely.

    Args:
        a (complex): first argument
        b (complex): second argument

    Returns:
        complex: :math:`B(a, b)`
    """
    return np.exp(sp.special.loggamma(a) + sp.special.loggamma(b)-sp.special.loggamma(a + b))

def ConfMoment(j: complex, t: float, ParaSets: np.ndarray) -> complex:
    """Evaluate the conformal (Mellin) moment :math:`F(j, t)` for a single ansatz.

    The ansatz follows the Kumeri\u010dki\u2013M\u00fcller (KM) form:

    .. math::

        F(j, t) = \\frac{N}{B(2-\\alpha, 1+\\beta)}\\,
                  B(j+1-\\alpha-\\alpha^\\prime t,\\,1+\\beta)\\,
                  e^{b_{\\text{exp}}\\,t}\\,
                  (1 - t / m^2)^{-2}

    where :math:`B` is the Euler Beta function (see :func:`beta_loggamma`).

    Args:
        j (complex): conformal spin variable (the physical conformal spin is
            :math:`j+2`, but the offset is absorbed into the normalization).
        t (float): momentum transfer squared.
        ParaSets (np.ndarray): parameter array with last dimension of size 6,
            laid out as ``[norm, alpha, beta, alphap, bexp, invm2]``:

            * ``norm``   – overall normalization :math:`N`
            * ``alpha``  – small-:math:`x` Regge intercept
            * ``beta``   – large-:math:`x` power
            * ``alphap`` – Regge slope :math:`\\alpha^\\prime`
            * ``bexp``   – exponential :math:`t`-slope
            * ``invm2``  – inverse mass squared :math:`1/m^2` for the dipole
              residual; set to 0 to disable

    Returns:
        complex or np.ndarray: conformal moment :math:`F(j, t)` with the same
        leading shape as ``ParaSets[..., 0]``.

    Note:
        ``ParaSets`` may carry arbitrary leading dimensions.  The typical
        usage is:

        * ``j`` shape ``(N,)``, ``t`` shape ``()``, ``ParaSets`` shape
          ``(5, n_ansatz, 6)`` – used when integrating over :math:`j`.
        * ``j`` shape ``(N,)``, ``t`` shape ``(N,)``, ``ParaSets`` shape
          ``(N, 5, n_ansatz, 6)`` – vectorized over kinematics.

        The function reshapes ``j`` and ``t`` to broadcast correctly with
        the leading dimensions of ``ParaSets``.
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
    return norm/ beta_loggamma(2 - alpha,1 + beta) * beta_loggamma (j + 1 - alpha - alphap * t, 1 + beta) * np.exp(t*bexp) * (1 - t * invm2 ) ** (-2)
    #return norm * beta_loggamma (j + 1 - alpha, 1 + beta) * (j + 1  - alpha)/ (j + 1 - alpha - alphap * t) * np.exp(t*bexp) * (1 - t * invm2 ) ** (-2)
    # (N) or (N, m1) or (N, m1, m2) .... depends on usage
    # For the recommended usage, the output is (N, 5, init_NumofAnsatz)

def Moment_Sum(j: complex, t: float, ParaSets: np.ndarray) -> complex:
    """Sum conformal moments over all ansatz terms.

    When ``ParaSets`` contains multiple ansatz terms (e.g. a valence term plus
    a sea term), :func:`ConfMoment` returns one contribution per term along the
    last axis.  This function reduces that axis by summing, yielding the total
    conformal moment for each flavor.

    Args:
        j (complex): conformal spin variable; see :func:`ConfMoment`.
        t (float): momentum transfer squared.
        ParaSets (np.ndarray): parameter array whose last axis has length 6
            and whose second-to-last axis indexes the ansatz terms;
            shape ``(..., 5, n_ansatz, 6)``.

    Returns:
        complex or np.ndarray: sum of conformal moments over all ansatz terms;
        shape ``(..., 5)``.
    """
    
    # ConfMoment_vec(j, t, ParaSets) should have shape (N, 5, init_NumofAnsatz)
    return np.sum(ConfMoment(j, t, ParaSets) ,  axis=-1) # (N, 5)
