r"""
GPD ansatz and observable calculations with QCD evolution.

The GPD ansatz follows the moment-space form
:math:`N\,x^{-\alpha}(1-x)^{\beta}` (see :mod:`Parameters`).  Starting
from those ansatz parameters, this module computes:

* **t-dependent PDFs** (:meth:`GPDobserv.tPDF`) via inverse Mellin transform.
* **GPDs** in :math:`x`-space (:meth:`GPDobserv.GPD`, :meth:`GPDobserv.GPDNLO_evMom`)
  via Mellin-Barnes contour integration.
* **Compton Form Factors** (CFFs, :meth:`GPDobserv.CFF`, :meth:`GPDobserv.CFFNLO`,
  :meth:`GPDobserv.CFFNLO_evMom`) for DVCS.
* **Transition Form Factors** (TFFs, :meth:`GPDobserv.TFF`, :meth:`GPDobserv.TFFNLO`,
  :meth:`GPDobserv.TFFNLO_evMom`) for DVMP.
* **Generalized Form Factors** (:meth:`GPDobserv.GFFj0`).
* The discrete :math:`j=0` pion-pole contributions to
  :math:`\widetilde E` GFFs and CFFs.
* The fixed :math:`j=1` C form factor and its discrete CFF and TFF
  contributions.

All integrals use conformal (Mellin-Barnes) contours and support both
leading-order (LO, ``p_order=1``) and next-to-leading-order (NLO,
``p_order=2``) QCD evolution.
"""
from .Evolution import Moment_Evo_LO,Moment_Evo_LO_NSp1, TFF_Evo_LO, CFF_Evo_LO, TFF_Evo_NLO_evWC, TFF_Evo_NLO_evMOM, CFF_Evo_NLO_evWC,CFF_Evo_NLO_evMOM, GPD_Moment_Evo_NLO,tPDF_Moment_Evo_NLO, tPDF_Moment_Evo_NLO_NSp1, fixed_quadvec, inv_flav_trans
from .Parameters import Moment_Sum
from .Evolution import InvMellinWaveFuncQ, InvMellinWaveFuncG, ConfWaveFuncQ, ConfWaveFuncG, ConfWaveFuncQ_over_sinpij, ConfWaveFuncG_over_sinpij

import scipy as sp
import numpy as np
from scipy.integrate import quad_vec, fixed_quad
from scipy.special import gamma

# ---------------------------------------------------------------------------
# Module-level contour and integration parameters
# ---------------------------------------------------------------------------
#intercept for inverse Mellin transformation
inv_Mellin_intercept = 0.25

#Cutoff for inverse Mellin transformation
inv_Mellin_cutoff = 20

#Cutoff for Mellin Barnes integral
Mellin_Barnes_intercept = 0.3

#Cutoff for Mellin Barnes integral
Mellin_Barnes_cutoff = 20

#Number of effective fermions
NFEFF = 2

#Relative precision Goal of quad set to be 1e-3
Prec_Goal = 1e-3

# Fixed inputs for the dipole-regulated pion-pole residue (all masses in GeV).
PION_POLE_G_A = 1.2756
PION_POLE_NUCLEON_MASS = 0.938
PION_MASS = 0.14
   

def flv_to_indx(flv: str) -> int:
    """Map a flavor string to its integer index.

    Args:
        flv (str): flavor label — ``'u'``, ``'d'``, ``'g'``,
            ``'NS'`` (non-singlet), or ``'S'`` (singlet).

    Returns:
        int: index ``0`` (u), ``1`` (d), ``2`` (g), ``3`` (NS), or ``4`` (S).
    """
    if(flv=="u"):
        return 0
    if(flv=="d"):
        return 1
    if(flv=="g"):
        return 2
    if(flv=="NS"):
        return 3
    if(flv=="S"):
        return 4

def flvs_to_indx(flvs: list) -> np.ndarray:
    """Apply :func:`flv_to_indx` to every element of a flavor list.

    Args:
        flvs (list of str): list of flavor labels accepted by
            :func:`flv_to_indx`.

    Returns:
        np.ndarray: integer index array of dtype ``int32``.
    """

    output = [flv_to_indx(flv) for flv in flvs]
    return np.array(output, dtype=np.int32)

def Flv_Intp(Flv_array: np.ndarray, flv: str) -> np.ndarray:
    """Extract or combine flavor components from an array in the ``[u, d, g]`` basis.

    Args:
        Flv_array (np.ndarray): shape ``(..., 5)`` complex array in the flavor
            basis ``[u_V + u-bar, u-bar, d_V + d-bar, d-bar, g]``.
        flv (str): flavor selector — ``'u'``, ``'d'``, ``'g'``,
            ``'NS'`` (:math:`u - d`), or ``'S'`` (:math:`u + d`).

    Returns:
        np.ndarray: shape ``(...)`` complex array for the requested flavor
        combination.
    """
    _flv_index = flv_to_indx(flv)
    return np.choose(_flv_index, [Flv_array[...,0], Flv_array[..., 1], Flv_array[..., 2],\
                        Flv_array[..., 0]-Flv_array[..., 1], Flv_array[..., 0]+Flv_array[..., 1]])
    # return np.einsum('...j,...j', Flv_array, _helper) # (N)
    
def flvmask(flv: str) -> np.ndarray:
    """Return a binary flavor mask in the evolution basis.

    Used to project evolved conformal moments onto a quark, gluon, or all-flavor
    combination when computing CFFs and TFFs.

    Args:
        flv (str): ``'All'`` (all flavors), ``'q'`` (quarks only), or
            ``'g'`` (gluon only).

    Returns:
        np.ndarray: length-5 integer mask, with ``1`` for active and ``0`` for
        inactive flavors in the evolution basis
        ``[uV, u-bar, dV, d-bar, g]``.
    """
    # Note: this mask operates in the evolution basis.
    if (flv == 'All'):
        return np.array([1,1,1,1,1])
    elif (flv == 'g'):
        return np.array([0,0,0,0,1])
    elif (flv == 'q'):
        return np.array([1,1,1,1,0])

'''
InvMellinWaveC = np.array([[InvMellinWaveFuncQ(s, self.x), InvMellinWaveFuncQ(s, self.x) - self.p * InvMellinWaveFuncQ(s, -self.x),0,0,0],
                            [0,0,InvMellinWaveFuncQ(s, self.x), InvMellinWaveFuncQ(s, self.x) - self.p * InvMellinWaveFuncQ(s, -self.x),0],
                            [0,0,0,0,(InvMellinWaveFuncG(s, self.x)+ self.p * InvMellinWaveFuncG(s, -self.x))]]) # (3, 5) matrix
                            # in my case, I want it to be (N, 3, 5) ndarray
'''
            
WF_helper1 = np.array([[1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0]])
WF_helper2 = np.array([[0, -1, 0, 0, 0],
                    [0, 0, 0, -1, 0],
                    [0, 0, 0, 0, 0]])
WF_helper3 = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1]])


# ---------------------------------------------------------------------------
# GPDobserv class
# ---------------------------------------------------------------------------

# Class for observables
class GPDobserv (object) :
    #Initialization of observables. Each is a function of (x, xi ,t, Q), p for parity: p = 1 for vector GPDs (H, E) and p = -1 for axial-vector GPDs (Ht, Et)
    def __init__(self, init_x: float, init_xi: float, init_t: float, init_Q: float, p: int) -> None:
        r"""Initialize a GPD kinematics point.

        Args:
            init_x (float): momentum fraction :math:`x`.
            init_xi (float): skewness parameter :math:`\xi`.
            init_t (float): squared momentum transfer :math:`t` in GeV\ :sup:`2`.
            init_Q (float): hard scale :math:`Q` (photon virtuality or
                factorization scale) in GeV.
            p (int): parity — ``+1`` for vector-like GPDs (:math:`H`, :math:`E`),
                ``-1`` for axial-vector-like GPDs (:math:`\tilde{H}`, :math:`\tilde{E}`).
        """
        self.x = init_x
        self.xi = init_xi
        self.t = init_t
        self.Q = init_Q
        self.p = p

    def tPDF(self, flv: str, ParaAll: np.ndarray, p_order: int = 1) -> np.ndarray:
        r"""t-dependent PDF :math:`f(x, t)` for a given flavor.

        Computes the impact-parameter-space PDF via the inverse Mellin
        transform along a straight contour with real part
        ``reS = 1.25``.

        See also: :func:`Evolution.Moment_Evo_LO`,
        :func:`Evolution.tPDF_Moment_Evo_NLO`.

        Args:
            flv (str): flavor — ``'u'``, ``'d'``, ``'g'``, ``'S'``, or ``'NS'``.
            ParaAll (np.ndarray): shape ``(..., 3, 5, n_ansatz, 6)`` parameter
                array. ``ParaAll[..., 0, ...]`` contains the forward
                (:math:`\xi^0`) parameters for the five flavors
                ``[uV, ubar, dV, dbar, g]``; the :math:`\xi^2` slice is
                accepted for interface consistency but is not used here.
            p_order (int): perturbative order — ``1`` for LO (default),
                ``2`` for NLO.

        Returns:
            np.ndarray: :math:`f(x, t)` for the requested flavor.
        """
        # originally, all parameters should be (4, 3, 5, 1, 5)
        # ParaAll would be a ( 3, 5, 1, 5) matrix
        # this means Para_Forward would be a matrix of (5, 1, 5)

        # For now, I will pass ParaAll as (N, 3, 5, 1, 5) array
        # This is not optimal for performance, but it somewhat retains backwards compatibility
        # For better speed, more changes are needed. 

        Para_Forward = ParaAll[..., 0, :, :, :] # (N, 3, 5, 1, 5) 

        def InvMellinWaveConf(s: complex):
            # s is scalar (but it actually can be an ndarray as long as broadcasting rule allows it)

            InvMellinWaveC =np.einsum('..., ij->...ij', InvMellinWaveFuncQ(s, self.x), WF_helper1) \
                            + np.einsum('... ,ij->...ij', self.p * InvMellinWaveFuncQ(s, -self.x), WF_helper2) \
                            + np.einsum('... ,ij->...ij', (InvMellinWaveFuncG(s, self.x)+ self.p * InvMellinWaveFuncG(s, -self.x)), WF_helper3)

            return InvMellinWaveC #(N, 3, 5)

        def Integrand_inv_Mellin(s: complex):
            # Calculate the unevolved moments in the orginal flavor basis
            # originally, Para_Forward will have shape (5, 1, 5) now (N, 5, 1, 5)  # in previous version is is (5, 1, 4) and (N, 5, 1, 4)           

            ConfFlav = Moment_Sum(s-1, self.t, Para_Forward) # shape (N, 5)

            # Evolved moments in evolution basis
            if (p_order == 1):
                ConfEv = Moment_Evo_LO(s - 1, NFEFF, self.p, self.Q, ConfFlav)
            elif (p_order == 2):
                ConfEv = tPDF_Moment_Evo_NLO(s - 1, NFEFF, self.p, self.Q, ConfFlav)
            
            # Inverse transform the evolved moments back to the flavor basis
            EvoConfFlav = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv) #(N, 5)
    
            # Return the evolved moments with x^(-s) for quark or x^(-s+1) for gluon for the given flavor flv = "u", "d", "S", "NS" or "g"
            #  InvMellinWaveConf(s): (N, 3, 5)            
            # the result of np.einsum will be (N, 3)
            # Flv_Intp  result (N)
            return Flv_Intp(np.einsum('...ij,...j->...i', InvMellinWaveConf(s), EvoConfFlav), flv)
        
        # The contour for inverse Meliin transform. Note that S here is the analytically continued n which is j + 1 not j !
        reS = 0.25 + 1
        Max_imS = 100 
        
        return 1/(2 * np.pi) * np.real(fixed_quadvec(lambda imS : Integrand_inv_Mellin(reS + 1j * imS) + Integrand_inv_Mellin(reS - 1j * imS) ,0, + Max_imS, n=300))

    def GPD(self, flv: str, ParaAll: np.ndarray, p_order: int = 1) -> float:
        r"""GPD :math:`F(x, \xi, t)` in flavor space.

        Evaluates the GPD at the kinematics ``(self.x, self.xi, self.t)``
        using a Mellin-Barnes contour integral.  Dispatches to
        :meth:`GPDNLO_evMom` automatically when ``p_order=2``.

        Args:
            flv (str): flavor — ``'u'``, ``'d'``, ``'g'``, ``'S'``, or ``'NS'``.
            ParaAll (np.ndarray): shape ``(..., 3, 5, n_ansatz, 6)`` parameter
                array with slices:

                * ``[..., 0, ...]`` — :math:`\xi^0` (forward) parameters;
                  flavors ``[uV, ubar, dV, dbar, g]``
                * ``[..., 1, ...]`` — :math:`\xi^2` correction parameters
                * ``[..., 2, ...]`` — :math:`\xi^4` correction parameters

            p_order (int): perturbative order — ``1`` for LO (default),
                ``2`` for NLO.

        Returns:
            float: :math:`F(x, \xi, t)` for the requested flavor.
        """
        #[Para_Forward, Para_xi2, Para_xi4] = ParaAll
        if (p_order == 2):
            return self.GPDNLO_evMom(flv, ParaAll)
        
        Para_Forward = ParaAll[..., 0, :, :, :]  # each (N, 5, 1, 5)
        Para_xi2     = ParaAll[..., 1, :, :, :]
        Para_xi4     = ParaAll[..., 2, :, :, :]

        # The contour for Mellin-Barnes integral in terms of j not n.         
        def ConfWaveConv(j: complex):

            ConfWaveC =np.einsum('..., ij->...ij', ConfWaveFuncQ(j, self.x, self.xi), WF_helper1) \
                     + np.einsum('... ,ij->...ij', self.p * ConfWaveFuncQ(j, -self.x, self.xi), WF_helper2) \
                     + np.einsum('... ,ij->...ij', ConfWaveFuncG(j, self.x, self.xi)+ self.p * ConfWaveFuncG(j, -self.x, self.xi), WF_helper3)

            return ConfWaveC
        
        def ConfWaveConv_over_sinpiji(j: complex):
            """
            This function exists for technical reasons:
            
            The 1/sin(pi*(j+1)) factor is exponentially suppressed on the imaginary axes on both side,
            whereas the conformal partial wave function is exponentiall divergent.
            So we absorb 1/sin(pi*(j+1)) factor into the conformal partial wave function in advance to avoid the overflow in the conformal wave function.
            This is only need for Mellin-Barnes integral!
            """
            
            ConfWaveC =np.einsum('..., ij->...ij', ConfWaveFuncQ_over_sinpij(j, self.x, self.xi), WF_helper1) \
                     + np.einsum('... ,ij->...ij', self.p * ConfWaveFuncQ_over_sinpij(j, -self.x, self.xi), WF_helper2) \
                     + np.einsum('... ,ij->...ij', ConfWaveFuncG_over_sinpij(j, self.x, self.xi)+ self.p * ConfWaveFuncG_over_sinpij(j, -self.x, self.xi), WF_helper3)

            return ConfWaveC
        
        # Put in the extra 1/sin(pi*(j+1)) factor if over_pij = 1. Included (=1) by default
        def Integrand_Mellin_Barnes(j: complex, over_pij: int = 1):

            ConfFlav     = Moment_Sum(j, self.t, Para_Forward) #(N, 5)
            ConfFlav_xi2 = Moment_Sum(j, self.t, Para_xi2)
            ConfFlav_xi4 = Moment_Sum(j, self.t, Para_xi4)
            
            ConfEv     = Moment_Evo_LO(j, NFEFF, self.p, self.Q, ConfFlav)
            ConfEv_xi2 = Moment_Evo_LO(j+2, NFEFF, self.p, self.Q, ConfFlav_xi2)
            ConfEv_xi4 = Moment_Evo_LO(j+4, NFEFF, self.p, self.Q, ConfFlav_xi4)
                
            ConfFlavEv     = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv) #(N, 5)
            ConfFlavEv_xi2 = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv_xi2)
            ConfFlavEv_xi4 = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv_xi4)
            
            if(over_pij == 1):
                return Flv_Intp(np.einsum('...ij,...j->...i', ConfWaveConv_over_sinpiji(j), ConfFlavEv) \
                        + self.xi ** 2 * np.einsum('...ij,...j->...i', ConfWaveConv_over_sinpiji(j+2), ConfFlavEv_xi2) \
                        + self.xi ** 4 * np.einsum('...ij,...j->...i', ConfWaveConv_over_sinpiji(j+4), ConfFlavEv_xi4),flv)
            else:
                return Flv_Intp(np.einsum('...ij,...j->...i', ConfWaveConv(j), ConfFlavEv) \
                        + self.xi ** 2 * np.einsum('...ij,...j->...i', ConfWaveConv(j+2), ConfFlavEv_xi2) \
                        + self.xi ** 4 * np.einsum('...ij,...j->...i', ConfWaveConv(j+4), ConfFlavEv_xi4),flv)
                
        # Adding a j = 0 term because the contour do not enclose the j = 0 pole which should be the 0th conformal moment.
        def GPD0():

            #Note: Naively, this function simply returns Integrand_Mellin_Barnes([0.]) like the GPD1() above.
            #      However, the zeroth moment is only defined for valence quark not sea quark or gluon
            #      Thus there will be divergences in moment when j = 0.       
            #      This will be taken care of by the Moment_Evo_LO_NSp1() function that evolve the NS part of the moments only (charge even contributions are zero)
            #      
            #      The better choice is to model the leading moment terms separately, and fit them to other quantities since those terms are not well constrained by the CFF/TFF anyway.
            eps = 10. ** (-6)
            j0 = np.array([0.]) + eps

            ConfFlav     = Moment_Sum(j0, self.t, Para_Forward)
            ConfFlav_xi2 = Moment_Sum(j0, self.t, Para_xi2)
            ConfFlav_xi4 = Moment_Sum(j0, self.t, Para_xi4)
            
            if(self.p == 1):
                ConfEv     = Moment_Evo_LO_NSp1(j0, NFEFF, self.p, self.Q, ConfFlav)
                ConfEv_xi2 = Moment_Evo_LO_NSp1(j0+2, NFEFF, self.p, self.Q, ConfFlav_xi2)
                ConfEv_xi4 = Moment_Evo_LO_NSp1(j0+4, NFEFF, self.p, self.Q, ConfFlav_xi4)
            else:
                ConfEv     = Moment_Evo_LO(j0, NFEFF, self.p, self.Q, ConfFlav)
                ConfEv_xi2 = Moment_Evo_LO(j0+2, NFEFF, self.p, self.Q, ConfFlav_xi2)
                ConfEv_xi4 = Moment_Evo_LO(j0+4, NFEFF, self.p, self.Q, ConfFlav_xi4)
                
            ConfFlavEv     = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv) #(N, 5)
            ConfFlavEv_xi2 = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv_xi2)
            ConfFlavEv_xi4 = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv_xi4)

            return Flv_Intp(np.einsum('...ij,...j->...i', ConfWaveConv(j0), ConfFlavEv) \
                    + self.xi ** 2 * np.einsum('...ij,...j->...i', ConfWaveConv(j0+2), ConfFlavEv_xi2) \
                    + self.xi ** 4 * np.einsum('...ij,...j->...i', ConfWaveConv(j0+4), ConfFlavEv_xi4),flv)
            
        reJ = 1 - 0.2
        Max_imJ = 180
        return 1/2*np.real(fixed_quadvec(lambda imJ : Integrand_Mellin_Barnes(reJ + 1j* imJ) + Integrand_Mellin_Barnes(reJ - 1j* imJ),0, Max_imJ, n=300)) + np.real(GPD0())[0]
    
    def GPDNLO_evMom(self, flv: str, ParaAll: np.ndarray) -> float:
        r"""NLO GPD :math:`F(x, \xi, t)` using the evolved-moment method.

        Equivalent to :meth:`GPD` with ``p_order=2`` but uses moment evolution
        (:func:`Evolution.GPD_Moment_Evo_NLO`) instead of evolved Wilson
        coefficients.  Both approaches yield numerically consistent results.

        Args:
            flv (str): flavor — ``'u'``, ``'d'``, ``'g'``, ``'S'``, or ``'NS'``.
            ParaAll (np.ndarray): shape ``(..., 3, 5, n_ansatz, 6)`` parameter
                array with slices:

                * ``[..., 0, ...]`` — :math:`\xi^0` (forward) parameters;
                  flavors ``[uV, ubar, dV, dbar, g]``
                * ``[..., 1, ...]`` — :math:`\xi^2` correction parameters
                * ``[..., 2, ...]`` — :math:`\xi^4` correction parameters

        Returns:
            float: :math:`F(x, \xi, t)` for the requested flavor at NLO.
        """
        
        Para_Forward = ParaAll[..., 0, :, :, :]  # each (N, 5, 1, 5)
        Para_xi2     = ParaAll[..., 1, :, :, :]
        Para_xi4     = ParaAll[..., 2, :, :, :]

        # The contour for Mellin-Barnes integral in terms of j not n.         
        def ConfWaveConv(j: complex):

            ConfWaveC =np.einsum('..., ij->...ij', ConfWaveFuncQ(j, self.x, self.xi), WF_helper1) \
                     + np.einsum('... ,ij->...ij', self.p * ConfWaveFuncQ(j, -self.x, self.xi), WF_helper2) \
                     + np.einsum('... ,ij->...ij', ConfWaveFuncG(j, self.x, self.xi)+ self.p * ConfWaveFuncG(j, -self.x, self.xi), WF_helper3)

            return ConfWaveC
        
        def ConfWaveConv_over_sinpiji(j: complex):
            """
            This function exists for technical reasons:
            
            The 1/sin(pi*(j+1)) factor is exponentially suppressed on the imaginary axes on both side,
            whereas the conformal partial wave function is exponentiall divergent.
            So we absorb 1/sin(pi*(j+1)) factor into the conformal partial wave function in advance to avoid the overflow in the conformal wave function.
            This is only need for Mellin-Barnes integral!
            """

            ConfWaveC =np.einsum('..., ij->...ij', ConfWaveFuncQ_over_sinpij(j, self.x, self.xi), WF_helper1) \
                     + np.einsum('... ,ij->...ij', self.p * ConfWaveFuncQ_over_sinpij(j, -self.x, self.xi), WF_helper2) \
                     + np.einsum('... ,ij->...ij', ConfWaveFuncG_over_sinpij(j, self.x, self.xi)+ self.p * ConfWaveFuncG_over_sinpij(j, -self.x, self.xi), WF_helper3)

            return ConfWaveC
        
        # Put in the extra 1/sin(pi*(j+1)) factor if over_pij = 1. Included (=1) by default
        def Integrand_Mellin_Barnes(j: complex, over_pij: int = 1):

            ConfEv     = GPD_Moment_Evo_NLO(j, NFEFF, self.p, self.Q, self.t, self.xi, Para_Forward,0)
            ConfEv_xi2 = GPD_Moment_Evo_NLO(j+2, NFEFF, self.p, self.Q, self.t, self.xi, Para_xi2,2)
            ConfEv_xi4 = GPD_Moment_Evo_NLO(j+4, NFEFF, self.p, self.Q, self.t, self.xi, Para_xi4,4)
        
            # Inverse transform the evolved moments back to the flavor basis
            ConfFlavEv     = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv) #(N, 5)
            ConfFlavEv_xi2 = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv_xi2)
            ConfFlavEv_xi4 = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv_xi4)
            
            if(over_pij == 1):
                return Flv_Intp(np.einsum('...ij,...j->...i', ConfWaveConv_over_sinpiji(j), ConfFlavEv) \
                        + self.xi ** 2 * np.einsum('...ij,...j->...i', ConfWaveConv_over_sinpiji(j+2), ConfFlavEv_xi2) \
                        + self.xi ** 4 * np.einsum('...ij,...j->...i', ConfWaveConv_over_sinpiji(j+4), ConfFlavEv_xi4),flv)
            else:
                return Flv_Intp(np.einsum('...ij,...j->...i', ConfWaveConv(j), ConfFlavEv) \
                        + self.xi ** 2 * np.einsum('...ij,...j->...i', ConfWaveConv(j+2), ConfFlavEv_xi2) \
                        + self.xi ** 4 * np.einsum('...ij,...j->...i', ConfWaveConv(j+4), ConfFlavEv_xi4),flv)
        # Adding a j = 0 term because the contour do not enclose the j = 0 pole which should be the 0th conformal moment.
        
        def GPD0():

            #Note: Naively, this function simply returns Integrand_Mellin_Barnes([0.]) like the GPD1() above.
            #      However, the zeroth moment is only defined for valence quark not sea quark or gluon
            #      Thus there will be divergences in moment when j = 0.       
            #      This will be taken care of by the tPDF_Moment_Evo_NLO_NSp1() function that evolve the NS part of the moments only (charge even contributions are zero)
            #      
            #      The better choice is to model the leading moment terms separately, and fit them to other quantities since those terms are not well constrained by the CFF/TFF anyway.
            eps = 10. ** (-6)
            j0 = np.array([0.]) + eps

            ConfFlav     = Moment_Sum(j0, self.t, Para_Forward)
            ConfFlav_xi2 = Moment_Sum(j0, self.t, Para_xi2)
            ConfFlav_xi4 = Moment_Sum(j0, self.t, Para_xi4)
            
            if(self.p == 1):
                ConfEv     = tPDF_Moment_Evo_NLO_NSp1(j0, NFEFF, self.p, self.Q, ConfFlav)
                ConfEv_xi2 = tPDF_Moment_Evo_NLO_NSp1(j0+2, NFEFF, self.p, self.Q, ConfFlav_xi2)
                ConfEv_xi4 = tPDF_Moment_Evo_NLO_NSp1(j0+4, NFEFF, self.p, self.Q, ConfFlav_xi4)
            else:
                ConfEv     = tPDF_Moment_Evo_NLO(j0, NFEFF, self.p, self.Q, ConfFlav)
                ConfEv_xi2 = tPDF_Moment_Evo_NLO(j0+2, NFEFF, self.p, self.Q, ConfFlav_xi2)
                ConfEv_xi4 = tPDF_Moment_Evo_NLO(j0+4, NFEFF, self.p, self.Q, ConfFlav_xi4)
                
            ConfFlavEv     = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv) #(N, 5)
            ConfFlavEv_xi2 = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv_xi2)
            ConfFlavEv_xi4 = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv_xi4)
            
            return Flv_Intp(np.einsum('...ij,...j->...i', ConfWaveConv(j0), ConfFlavEv) \
                    + self.xi ** 2 * np.einsum('...ij,...j->...i', ConfWaveConv(j0+2), ConfFlavEv_xi2) \
                    + self.xi ** 4 * np.einsum('...ij,...j->...i', ConfWaveConv(j0+4), ConfFlavEv_xi4),flv)
        
        reJ = 1 - 0.2
        Max_imJ = 180
        return 1/2*np.real(fixed_quadvec(lambda imJ : Integrand_Mellin_Barnes(reJ + 1j* imJ) + Integrand_Mellin_Barnes(reJ - 1j* imJ),0, Max_imJ, n=300)) + np.real(GPD0())[0]
    
    def GFFj0(self, j: int, flv: str, ParaAll: np.ndarray, p_order: int) -> float:
        r"""Generalized Form Factor :math:`A_{j+1,0}(t)` (the :math:`\xi^0` Mellin moment).

        Computes :math:`\int dx\, x^j F(x, \xi, t)` for quarks and
        :math:`\int dx\, x^{j-1} F(x, \xi, t)` for gluons at
        :math:`\xi = 0`.

        Note:
            Only LO and NLO evolution are implemented.  Gluon GPDs reduce to
            :math:`x g(x)`, so there is a moment index shift relative to
            quarks.  This method is **not well-maintained** and is marked for
            future revision.

        Args:
            j (int): Mellin moment index (:math:`n = j + 1`).
            flv (str): flavor — ``'u'``, ``'d'``, ``'g'``, ``'S'``, or ``'NS'``.
            ParaAll (np.ndarray): shape ``(..., 3, 5, n_ansatz, 6)`` parameter
                array; only the :math:`\xi^0` slice is used.
            p_order (int): perturbative order — ``1`` for LO, ``2`` for NLO.

        Returns:
            float: :math:`A_{j+1,0}(t)` for the requested flavor.
        """

        # j, flv both have shape (N)
        # ParaAll: (N, 3, 5, 1, 5)
        eps = 10. ** (-6)
        j_arr_float = np.array([float(j)])
        
        if(j==0):
            j_arr_float = j_arr_float + eps
        
        Para_Forward = ParaAll[..., 0, :, :, :]  # (N, 5, 1, 5)
        _helper1 = np.array([[1, 1, 0, 0, 0],
                             [0, 0, 1, 1, 0],
                             [0, 0, 0, 0, 1/2]])
        _helper2 = np.array([[0, -1, 0, 0, 0],
                             [0, 0, 0, -1, 0],
                             [0, 0, 0, 0, -1/2]])
        GFF_trans = np.einsum('... , ij->...ij', self.p * (-1)**j, _helper2) + _helper1  # (N, 3, 5)
        
        ConfFlav = Moment_Sum(j_arr_float, self.t, Para_Forward)

        if (j==0) and (self.p == 1):
            if (p_order == 1):
                ConfEv = Moment_Evo_LO_NSp1(j_arr_float, NFEFF, self.p, self.Q, ConfFlav)[0]
            elif (p_order == 2):
                ConfEv = tPDF_Moment_Evo_NLO_NSp1(j_arr_float, NFEFF, self.p, self.Q, ConfFlav)[0]
        else:
            if (p_order == 1):
                ConfEv = Moment_Evo_LO(j_arr_float, NFEFF, self.p, self.Q, ConfFlav)[0]
            elif (p_order == 2):
                ConfEv = tPDF_Moment_Evo_NLO(j_arr_float, NFEFF, self.p, self.Q, ConfFlav)[0]

        # Inverse transform the evolved moments back to the flavor basis
        EvoConfFlav = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv) #(N, 5)
        result = Flv_Intp(np.einsum('...ij, ...j->...i', GFF_trans, EvoConfFlav), flv) # (N_~mask)
        
        return np.real(result)
    
    def CFF(self, ParaAll: np.ndarray, muf: float, p_order: int = 1, flv: str = 'All') -> np.ndarray:
        r"""Charge-weighted Compton Form Factor :math:`\mathcal{F}(\xi, t)`.

        Computes :math:`\mathcal{F} = Q_u^2 F_u + Q_d^2 F_d` via a
        Mellin-Barnes contour integral.  Dispatches to :meth:`CFFNLO`
        automatically when ``p_order=2``.

        Args:
            ParaAll (np.ndarray): shape ``(..., 3, 5, n_ansatz, 6)`` parameter
                array with slices:

                * ``[..., 0, ...]`` — :math:`\xi^0` (forward) parameters
                * ``[..., 1, ...]`` — :math:`\xi^2` correction parameters
                * ``[..., 2, ...]`` — :math:`\xi^4` correction parameters

                Flavor axis order: ``[uV, ubar, dV, dbar, g]``.
            muf (float): factorization scale in GeV.
            p_order (int): perturbative order — ``1`` for LO (default),
                ``2`` for NLO.
            flv (str): flavor filter — ``'All'``, ``'q'``, or ``'g'``.

        Returns:
            np.ndarray: :math:`\mathcal{F}(\xi, t) = Q_u^2 F_u + Q_d^2 F_d`.
        """
        if (p_order == 2):
            return self.CFFNLO(ParaAll, muf, flv)
        
        Para_Forward = ParaAll[..., 0, :, :, :]  # each (N, 5, 1, 5)
        Para_xi2     = ParaAll[..., 1, :, :, :]
        Para_xi4     = ParaAll[..., 2, :, :, :]

        def Integrand_Mellin_Barnes_CFF(j: complex):

            ConfFlav     = Moment_Sum(j, self.t, Para_Forward) #(N, 5)
            ConfFlav_xi2 = Moment_Sum(j, self.t, Para_xi2)
            ConfFlav_xi4 = Moment_Sum(j, self.t, Para_xi4)

            # shape (N, 5)

            EvoConf_Wilson = CFF_Evo_LO(j, NFEFF, self.p, self.Q, ConfFlav) \
                                + CFF_Evo_LO(j+2, NFEFF, self.p, self.Q, ConfFlav_xi2) \
                                    + CFF_Evo_LO(j+4, NFEFF, self.p, self.Q, ConfFlav_xi4)
            fmask = flvmask(flv)
            return np.einsum('j, ...j', fmask, EvoConf_Wilson)

        def Integrand_CFF(imJ: complex):
            # mask = (self.p==1) # assume p can only be either 1 or -1

            result = np.ones_like(self.p) * self.xi ** (-reJ - 1j * imJ - 1) * Integrand_Mellin_Barnes_CFF(reJ + 1j * imJ) / 2

            if self.p==1:
                result *= (1j + np.tan((reJ + 1j * imJ) * np.pi / 2))
            else:
                result *= (1j - 1/np.tan((reJ + 1j * imJ) * np.pi / 2))

            return result

        # Adding extra j = 0 term for the axial vector CFFs
        def CFFj0():

            if self.p==1:
                result = np.ones_like(self.p) * 0
            else:
                result = np.ones_like(self.p) * self.xi ** (- 1) * Integrand_Mellin_Barnes_CFF(np.array([0.]))[0] *(2)

            return result
        
        # The contour for Mellin-Barnes integral in terms of j not n.
        reJ = 0.5 
        Max_imJ = 180 
        return fixed_quadvec(lambda imJ: Integrand_CFF(imJ)+Integrand_CFF(-imJ), 0,  Max_imJ, n=500) + CFFj0()

    def TFF(self, ParaAll: np.ndarray, muf: float, meson: int, p_order: int = 1, flv: str = 'All') -> np.ndarray:
        r"""Transition Form Factor :math:`\mathcal{F}(\xi, t)` for meson production.

        Computes the TFF via a Mellin-Barnes contour integral.  Dispatches to
        :meth:`TFFNLO` automatically when ``p_order=2``.

        Args:
            ParaAll (np.ndarray): shape ``(..., 3, 5, n_ansatz, 6)`` parameter
                array with slices:

                * ``[..., 0, ...]`` — :math:`\xi^0` (forward) parameters
                * ``[..., 1, ...]`` — :math:`\xi^2` correction parameters
                * ``[..., 2, ...]`` — :math:`\xi^4` correction parameters

                Flavor axis order: ``[uV, ubar, dV, dbar, g]``.
            muf (float): factorization scale in GeV.
            meson (int): meson code — ``1`` for :math:`\rho^0`, ``2`` for
                :math:`\phi`, ``3`` for :math:`J/\psi`.
            p_order (int): perturbative order — ``1`` for LO (default),
                ``2`` for NLO.
            flv (str): flavor filter — ``'All'``, ``'q'``, or ``'g'``.

        Returns:
            np.ndarray: :math:`\mathcal{F}(\xi, t)` for the requested meson.
        """
        if (p_order == 2):
            return self.TFFNLO(ParaAll, muf, meson, flv)
        
        #[Para_Forward, Para_xi2, Para_xi4] = ParaAll  # each (N, 5, 1, 5)
        Para_Forward = ParaAll[..., 0, :, :, :]  # each (N, 5, 1, 5)
        Para_xi2     = ParaAll[..., 1, :, :, :]
        Para_xi4     = ParaAll[..., 2, :, :, :]

        def Integrand_Mellin_Barnes_TFF(j: complex):

            ConfFlav     = Moment_Sum(j, self.t, Para_Forward) #(N, 5)
            ConfFlav_xi2 = Moment_Sum(j, self.t, Para_xi2)
            ConfFlav_xi4 = Moment_Sum(j, self.t, Para_xi4)
            
            EvoConf_Wilson = (TFF_Evo_LO(j, NFEFF, self.p, self.Q, ConfFlav, meson) \
                                + TFF_Evo_LO(j+2, NFEFF, self.p, self.Q, ConfFlav_xi2, meson) \
                                    + TFF_Evo_LO(j+4, NFEFF, self.p, self.Q, ConfFlav_xi4, meson))
            
            fmask = flvmask(flv)
            return np.einsum('j, ...j', fmask, EvoConf_Wilson)  
        
        def Integrand_TFF(imJ: complex):
            # mask = (self.p==1) # assume p can only be either 1 or -1

            result = np.ones_like(self.p) * self.xi ** (-reJ - 1j * imJ - 1) * Integrand_Mellin_Barnes_TFF(reJ + 1j * imJ) / 2

            if self.p==1:
                result *= (1j + np.tan((reJ + 1j * imJ) * np.pi / 2))
            else:
                result *= (1j - 1/np.tan((reJ + 1j * imJ) * np.pi / 2))

            return result

        # Adding extra j = 0 term for the axial vector CFFs
        def TFFj0():

            if self.p==1:
                result = np.ones_like(self.p) * 0
            else:
                result = np.ones_like(self.p) * self.xi ** (- 1) * Integrand_Mellin_Barnes_TFF(0) *(2)

            return result
        
        # The contour for Mellin-Barnes integral in terms of j not n.
        reJ = 0.5 
        Max_imJ = 120 
        
        return fixed_quadvec(lambda imJ: Integrand_TFF(imJ)+Integrand_TFF(-imJ), 0,  Max_imJ, n=500) + TFFj0()
    
    def CFFNLO(self, ParaAll: np.ndarray, muf: float, flv: str = 'All') -> np.ndarray:
        r"""NLO Compton Form Factor :math:`\mathcal{F}(\xi, t)` via evolved Wilson coefficients.

        Implements the NLO CFF using the evolved-Wilson-coefficient (evWC)
        method.  Can be called directly or via :meth:`CFF` with
        ``p_order=2``.

        Args:
            ParaAll (np.ndarray): shape ``(..., 3, 5, n_ansatz, 6)`` parameter
                array with slices:

                * ``[..., 0, ...]`` — :math:`\xi^0` (forward) parameters
                * ``[..., 1, ...]`` — :math:`\xi^2` correction parameters
                * ``[..., 2, ...]`` — :math:`\xi^4` correction parameters

                Flavor axis order: ``[uV, ubar, dV, dbar, g]``.
            muf (float): factorization scale in GeV.
            flv (str): flavor filter — ``'All'``, ``'q'``, or ``'g'``.

        Returns:
            np.ndarray: NLO :math:`\mathcal{F}(\xi, t)`.
        """
        #[Para_Forward, Para_xi2, Para_xi4] = ParaAll  # each (N, 5, 1, 5)
        Para_Forward = ParaAll[..., 0, :, :, :]  # each (N, 5, 1, 5)
        Para_xi2     = ParaAll[..., 1, :, :, :]
        Para_xi4     = ParaAll[..., 2, :, :, :]

        # The contour for Mellin-Barnes integral in terms of j not n.
        reJ = Mellin_Barnes_intercept 
        Max_imJ = Mellin_Barnes_cutoff
        
        def Integrand_Mellin_Barnes_CFF(j: complex):

            ConfFlav     = Moment_Sum(j, self.t, Para_Forward) #(N, 5)
            ConfFlav_xi2 = Moment_Sum(j, self.t, Para_xi2)
            ConfFlav_xi4 = Moment_Sum(j, self.t, Para_xi4)

            EvoConf_Wilson = (CFF_Evo_NLO_evWC(j, NFEFF, self.p, self.Q, ConfFlav, muf) \
                                +  CFF_Evo_NLO_evWC(j+2, NFEFF, self.p, self.Q, ConfFlav_xi2, muf) \
                                +  CFF_Evo_NLO_evWC(j+4, NFEFF, self.p, self.Q, ConfFlav_xi4, muf))
                        
            fmask = flvmask(flv)
            return np.einsum('j, ...j', fmask, EvoConf_Wilson)
        
        def tan_factor(j):
            if (self.p==1):
                return  1/(2j)*self.xi ** (-j-1)*(1j+np.tan(j * np.pi / 2))
            else:
                return  1/(2j)*self.xi ** (-j-1)*(1j-1/np.tan(j * np.pi / 2))
        
        eps= 10. **(-6)            
        # adding back the j=0 contribution            
        def CFFj0():
            if self.p==1:
                return 0
            else:
                return self.xi ** (- 1.) * Integrand_Mellin_Barnes_CFF(np.array([0.+eps]))[0] *(2) # the last factor of 2 is the residual of -1/(2j)*np.cot(j * np.pi / 2) at j=0
        
        reJ = 1-0.8
        
        Max_imJ = 150
        
        return 1j*fixed_quadvec(lambda imJ: tan_factor(reJ+1j*imJ)*Integrand_Mellin_Barnes_CFF(reJ+1j*imJ)+tan_factor(reJ-1j*imJ)*Integrand_Mellin_Barnes_CFF(reJ-1j*imJ), 0, Max_imJ,n = 300) + CFFj0()

    def CFFNLO_evMom(self, ParaAll: np.ndarray, muf: float, flv: str = 'All') -> np.ndarray:
        r"""NLO Compton Form Factor :math:`\mathcal{F}(\xi, t)` via evolved moments.

        Alternative NLO implementation using moment evolution
        (:func:`Evolution.CFF_Evo_NLO_evMOM`) instead of evolved Wilson
        coefficients.  Both methods produce numerically consistent results and
        can be used to cross-check each other.

        The :math:`j = 0` pole term is added back using the evWC method
        because the double-sum formula excludes it by construction.  For the
        same reason a :math:`j = 1` pole term is also restored.

        Args:
            ParaAll (np.ndarray): shape ``(..., 3, 5, n_ansatz, 6)`` parameter
                array with slices:

                * ``[..., 0, ...]`` — :math:`\xi^0` (forward) parameters
                * ``[..., 1, ...]`` — :math:`\xi^2` correction parameters
                * ``[..., 2, ...]`` — :math:`\xi^4` correction parameters

                Flavor axis order: ``[uV, ubar, dV, dbar, g]``.
            muf (float): factorization scale in GeV.
            flv (str): flavor filter — ``'All'``, ``'q'``, or ``'g'``.

        Returns:
            np.ndarray: NLO :math:`\mathcal{F}(\xi, t)` (evolved-moment method).
        """
        Para_Forward = ParaAll[..., 0, :, :, :]  # each (N, 5, 1, 5)
        Para_xi2     = ParaAll[..., 1, :, :, :]
        Para_xi4     = ParaAll[..., 2, :, :, :]

        # The contour for Mellin-Barnes integral in terms of j not n.
        reJ = Mellin_Barnes_intercept 
        Max_imJ = Mellin_Barnes_cutoff
        
        def Integrand_Mellin_Barnes_CFF(j: complex):
            
            EvoConf_Wilson = (CFF_Evo_NLO_evMOM(j, NFEFF, self.p, self.Q, self.t, self.xi, Para_Forward, 0, muf) \
                                        +  CFF_Evo_NLO_evMOM(j+2, NFEFF, self.p, self.Q, self.t, self.xi, Para_xi2, 2, muf) \
                                            +  CFF_Evo_NLO_evMOM(j+4, NFEFF, self.p, self.Q, self.t, self.xi, Para_xi4, 4, muf))
            
            fmask = flvmask(flv)
            return np.einsum('j, ...j', fmask, EvoConf_Wilson)
        
        def Integrand_Mellin_Barnes_CFF_evWC(j: complex):
            """
            It might look strange that we use the evolved-Wilson-Coefficient method in the evolved-moment module.
            We only use this to calcuate the j=0 contributions which has been excluded in the double summation formula because there's a pole in the moment F_j near j=0.
            When adding back this term, since j=0 is fixed, we always sum over the moment of Wilson coefficient so it's equivalent to the j=0 term is the evolved-Wilson-Coefficient module
            """
            ConfFlav     = Moment_Sum(j, self.t, Para_Forward) #(N, 5)
            ConfFlav_xi2 = Moment_Sum(j, self.t, Para_xi2)
            ConfFlav_xi4 = Moment_Sum(j, self.t, Para_xi4)

            EvoConf_Wilson = (CFF_Evo_NLO_evWC(j, NFEFF, self.p, self.Q, ConfFlav, muf) \
                                +  CFF_Evo_NLO_evWC(j+2, NFEFF, self.p, self.Q, ConfFlav_xi2, muf) \
                                +  CFF_Evo_NLO_evWC(j+4, NFEFF, self.p, self.Q, ConfFlav_xi4, muf))
                        
            fmask = flvmask(flv)
            return np.einsum('j, ...j', fmask, EvoConf_Wilson)
        
        def tan_factor(j):
            if (self.p==1):
                return  1/(2j)*self.xi ** (-j-1)*(1j+np.tan(j * np.pi / 2))
            else:
                return  1/(2j)*self.xi ** (-j-1)*(1j-1/np.tan(j * np.pi / 2))
        
        eps= 10. **(-6)
        
        # adding back the j=0 contribution using the evolved-Wilson-Coefficient method. Reason explained above
        def CFFj0():
            if self.p==1:
                return 0
            else:
                return self.xi ** (- 1.) * Integrand_Mellin_Barnes_CFF_evWC(np.array([0.+eps]))[0] *(2) # the last factor of 2 is the residual of -1/(2j)*np.cot(j * np.pi / 2) at j=0
        
        # for moment evolution, the j=1 pole is also missed because we choose 1<cj<2. The diagonal piece will be calculated and the off-diagonal piece is zero as expected for C_1.
        def CFFj1():

            if self.p==1:
                return self.xi ** (- 2.) * Integrand_Mellin_Barnes_CFF(np.array([1.+eps]))[0] *(2) # the last factor of 2 is the residual of 1/(2j)*np.tan(j * np.pi / 2) at j=1
            else:
                return 0            
        
        reJ = 2. - 0.6
    
        Max_imJ = 150

        return 1j*fixed_quadvec(lambda imJ: tan_factor(reJ+1j*imJ)*Integrand_Mellin_Barnes_CFF(reJ+1j*imJ)+tan_factor(reJ-1j*imJ)*Integrand_Mellin_Barnes_CFF(reJ-1j*imJ), 0, Max_imJ,n = 400) + CFFj0() + CFFj1()
    
    def TFFNLO(self, ParaAll: np.ndarray, muf: float, meson: int, flv: str = 'All') -> np.ndarray:
        r"""NLO Transition Form Factor :math:`\mathcal{F}(\xi, t)` via evolved Wilson coefficients.

        Implements the NLO TFF using the evolved-Wilson-coefficient (evWC)
        method.  Can be called directly or via :meth:`TFF` with
        ``p_order=2``.

        Args:
            ParaAll (np.ndarray): shape ``(..., 3, 5, n_ansatz, 6)`` parameter
                array with slices:

                * ``[..., 0, ...]`` — :math:`\xi^0` (forward) parameters
                * ``[..., 1, ...]`` — :math:`\xi^2` correction parameters
                * ``[..., 2, ...]`` — :math:`\xi^4` correction parameters

                Flavor axis order: ``[uV, ubar, dV, dbar, g]``.
            muf (float): factorization scale in GeV.
            meson (int): meson code — ``1`` for :math:`\rho^0`, ``2`` for
                :math:`\phi`, ``3`` for :math:`J/\psi`.
            flv (str): flavor filter — ``'All'``, ``'q'``, or ``'g'``.

        Returns:
            np.ndarray: NLO :math:`\mathcal{F}(\xi, t)` for the requested meson.
        """
        Para_Forward = ParaAll[..., 0, :, :, :]  # each (N, 5, 1, 5)
        Para_xi2     = ParaAll[..., 1, :, :, :]
        Para_xi4     = ParaAll[..., 2, :, :, :]

        # The contour for Mellin-Barnes integral in terms of j not n.
        reJ = Mellin_Barnes_intercept 
        Max_imJ = Mellin_Barnes_cutoff
        
        def Integrand_Mellin_Barnes_TFF(j: complex):

            ConfFlav     = Moment_Sum(j, self.t, Para_Forward) #(N, 5)
            ConfFlav_xi2 = Moment_Sum(j, self.t, Para_xi2)
            ConfFlav_xi4 = Moment_Sum(j, self.t, Para_xi4)

            EvoConf_Wilson = (TFF_Evo_NLO_evWC(j, NFEFF, self.p, self.Q, ConfFlav, meson, muf) \
                                +  TFF_Evo_NLO_evWC(j+2, NFEFF, self.p, self.Q, ConfFlav_xi2, meson, muf) \
                                +  TFF_Evo_NLO_evWC(j+4, NFEFF, self.p, self.Q, ConfFlav_xi4, meson, muf))
                        
            fmask = flvmask(flv)
            return np.einsum('j, ...j', fmask, EvoConf_Wilson)
        
        def tan_factor(j):
            if (self.p==1):
                return  1/(2j)*self.xi ** (-j-1)*(1j+np.tan(j * np.pi / 2))
            else:
                return  1/(2j)*self.xi ** (-j-1)*(1j-1/np.tan(j * np.pi / 2))
        
        eps= 10. **(-6)            
        # adding back the j=0 contribution            
        def TFFj0():
            if self.p==1:
                return 0
            else:
                return self.xi ** (- 1.) * Integrand_Mellin_Barnes_TFF(np.array([0.+eps]))[0] *(2) # the last factor of 2 is the residual of -1/(2j)*np.cot(j * np.pi / 2) at j=0
        
        reJ = 1-0.8
        
        Max_imJ = 150
        
        return 1j*fixed_quadvec(lambda imJ: tan_factor(reJ+1j*imJ)*Integrand_Mellin_Barnes_TFF(reJ+1j*imJ)+tan_factor(reJ-1j*imJ)*Integrand_Mellin_Barnes_TFF(reJ-1j*imJ), 0, Max_imJ,n = 300) + TFFj0()

    def TFFNLO_evMom(self, ParaAll: np.ndarray, muf: float, meson: int, flv: str = 'All') -> np.ndarray:
        r"""NLO Transition Form Factor :math:`\mathcal{F}(\xi, t)` via evolved moments.

        Alternative NLO implementation using moment evolution
        (:func:`Evolution.TFF_Evo_NLO_evMOM`) instead of evolved Wilson
        coefficients.  Both methods produce numerically consistent results and
        can be used to cross-check each other.

        The :math:`j = 0` and :math:`j = 1` pole terms are added back
        separately (see :meth:`CFFNLO_evMom` for the rationale).

        Args:
            ParaAll (np.ndarray): shape ``(..., 3, 5, n_ansatz, 6)`` parameter
                array with slices:

                * ``[..., 0, ...]`` — :math:`\xi^0` (forward) parameters
                * ``[..., 1, ...]`` — :math:`\xi^2` correction parameters
                * ``[..., 2, ...]`` — :math:`\xi^4` correction parameters

                Flavor axis order: ``[uV, ubar, dV, dbar, g]``.
            muf (float): factorization scale in GeV.
            meson (int): meson code — ``1`` for :math:`\rho^0`, ``2`` for
                :math:`\phi`, ``3`` for :math:`J/\psi`.
            flv (str): flavor filter — ``'All'``, ``'q'``, or ``'g'``.

        Returns:
            np.ndarray: NLO :math:`\mathcal{F}(\xi, t)` for the requested
            meson (evolved-moment method).
        """

        Para_Forward = ParaAll[..., 0, :, :, :]  # each (N, 5, 1, 5)
        Para_xi2     = ParaAll[..., 1, :, :, :]
        Para_xi4     = ParaAll[..., 2, :, :, :]

        # The contour for Mellin-Barnes integral in terms of j not n.
        reJ = Mellin_Barnes_intercept 
        Max_imJ = Mellin_Barnes_cutoff
        
        def Integrand_Mellin_Barnes_TFF(j: complex):
            
            EvoConf_Wilson = (TFF_Evo_NLO_evMOM(j, NFEFF, self.p, self.Q, self.t, self.xi, Para_Forward, 0, meson, muf) \
                                        +  TFF_Evo_NLO_evMOM(j+2, NFEFF, self.p, self.Q, self.t, self.xi, Para_xi2, 2, meson, muf) \
                                            +  TFF_Evo_NLO_evMOM(j+4, NFEFF, self.p, self.Q, self.t, self.xi, Para_xi4, 4, meson, muf))
            
            fmask = flvmask(flv)
            return np.einsum('j, ...j', fmask, EvoConf_Wilson)
        
        def tan_factor(j):
            if (self.p==1):
                return  1/(2j)*self.xi ** (-j-1)*(1j+np.tan(j * np.pi / 2))
            else:
                return  1/(2j)*self.xi ** (-j-1)*(1j-1/np.tan(j * np.pi / 2))
        
        eps= 10. **(-6)
        
        def Integrand_Mellin_Barnes_TFF_evWC(j: complex):
            """
            It might look strange that we use the evolved-Wilson-Coefficient method in the evolved-moment module.
            We only use this to calcuate the j=0 contributions which has been excluded in the double summation formula because there's a pole in the moment F_j near j=0.
            When adding back this term, since j=0 is fixed, we always sum over the moment of Wilson coefficient so it's equivalent to the j=0 term is the evolved-Wilson-Coefficient module
            """
            ConfFlav     = Moment_Sum(j, self.t, Para_Forward) #(N, 5)
            ConfFlav_xi2 = Moment_Sum(j, self.t, Para_xi2)
            ConfFlav_xi4 = Moment_Sum(j, self.t, Para_xi4)

            EvoConf_Wilson = (TFF_Evo_NLO_evWC(j, NFEFF, self.p, self.Q, ConfFlav, meson, muf) \
                                +  TFF_Evo_NLO_evWC(j+2, NFEFF, self.p, self.Q, ConfFlav_xi2, meson, muf) \
                                +  TFF_Evo_NLO_evWC(j+4, NFEFF, self.p, self.Q, ConfFlav_xi4, meson, muf))
                        
            fmask = flvmask(flv)
            return np.einsum('j, ...j', fmask, EvoConf_Wilson)
        
        # adding back the j=0 contribution using the evolved-Wilson-Coefficient method. Reason explained above
        def TFFj0():
            if self.p==1:
                return 0
            else:
                return self.xi ** (- 1.) * Integrand_Mellin_Barnes_TFF_evWC(np.array([0.+eps]))[0] *(2) # the last factor of 2 is the residual of -1/(2j)*np.cot(j * np.pi / 2) at j=0
        
        # for moment evolution, the j=1 pole is also missed because we choose 1<cj<2. The diagonal piece will be calculated and the off-diagonal piece is zero as expected for C_1.
        def TFFj1():

            if self.p==1:
                return self.xi ** (- 2.) * Integrand_Mellin_Barnes_TFF(np.array([1.+eps]))[0] *(2) # the last factor of 2 is the residual of 1/(2j)*np.tan(j * np.pi / 2) at j=1
            else:
                return 0            
        
        reJ = 2. - 0.5
    
        Max_imJ = 150

        return 1j*fixed_quadvec(lambda imJ: tan_factor(reJ+1j*imJ)*Integrand_Mellin_Barnes_TFF(reJ+1j*imJ)+tan_factor(reJ-1j*imJ)*Integrand_Mellin_Barnes_TFF(reJ-1j*imJ), 0, Max_imJ,n = 400) + TFFj0() + TFFj1()

    def pion_pole_Et_GFF_j0(self, flv: str, N: float, Lambda: float) -> float:
        r"""Return the canonical pion-pole contribution to the :math:`j=0`
        :math:`\widetilde E` generalized form factor.

        The fitted pole residue is

        .. math::

            P_\pi(t) = N\,\frac{2 g_A M_N^2}{m_\pi^2-t}
            \left(\frac{\Lambda^2-m_\pi^2}{\Lambda^2-t}\right)^2.

        A single normalization ``N`` and cutoff ``Lambda`` are shared by the
        two light flavors.  Consequently the contribution is purely
        isovector: :math:`\widetilde E^u_\pi=P_\pi` and
        :math:`\widetilde E^d_\pi=-P_\pi`.

        Args:
            flv: ``'u'``, ``'d'``, ``'NS'``, ``'S'``, or ``'g'``.
            N: Universal pion-pole normalization.
            Lambda: Dipole cutoff in GeV.

        Returns:
            The signed pole contribution for the requested flavor channel.

        Raises:
            ValueError: If this is not an axial observable (``p=-1``), or if
                the flavor label is unknown.
        """
        if self.p != -1:
            raise ValueError("The pion-pole Et GFF requires axial parity p=-1")

        pole = (
            N * 2.0 * PION_POLE_G_A * PION_POLE_NUCLEON_MASS**2 / (PION_MASS**2 - self.t)
            * ( (Lambda**2 - PION_MASS**2) / (Lambda**2 - self.t) )**2
        )

        flavor_factor = {
            "u": 1.0,
            "d": -1.0,
            "NS": 2.0,
            "S": 0.0,
            "g": 0.0,
        }
        if flv not in flavor_factor:
            raise ValueError(f"Unknown flavor for pion-pole Et GFF: {flv}")
        return flavor_factor[flv] * pole

    def pion_pole_Et_CFF_j0(self, N: float, Lambda: float, muf: float, p_order: int = 1, flv: str = "All") -> complex:
        r"""Return the discrete :math:`j=0` pion-pole contribution to
        :math:`\widetilde{\mathcal E}`.

        In GUMP's input flavor basis ``[uV, ubar, dV, dbar, g]`` the fixed
        conformal moment is represented as

        .. math::

            \widetilde E_{0,\pi}
            = [0, P_\pi/2, 0, -P_\pi/2, 0].

        The ``ubar`` and ``dbar`` slots are basis bookkeeping, not a claim
        that the pion pole is an ordinary antiquark PDF: for an axial
        :math:`j=0` moment GUMP reconstructs the physical light-flavor
        moments as ``uV + 2*ubar`` and ``dV + 2*dbar``.  The factors of one
        half therefore recover :math:`(+P_\pi,-P_\pi)` exactly.

        This is a pure axial non-singlet moment after the normal GUMP flavor
        transformation.  It is passed through the same DVCS evolution and
        Wilson-coefficient functions as the regular CFF.  The axial
        Mellin--Barnes residue supplies the final factor :math:`2/\xi`; at LO
        this gives exactly :math:`\widetilde{\mathcal E}_\pi=P_\pi(t)/\xi`.

        Args:
            N: Universal pion-pole normalization.
            Lambda: Dipole cutoff in GeV.
            muf: Factorization scale in GeV.
            p_order: ``1`` for LO or ``2`` for NLO.
            flv: Evolution-basis projection ``'All'``, ``'q'``, or ``'g'``.

        Returns:
            The charge-weighted pion-pole CFF contribution.

        Raises:
            ValueError: If the observable is not axial, if ``xi`` is zero,
                if the perturbative order is unsupported, or if ``flv`` is
                unknown.
        """
        if self.p != -1:
            raise ValueError("The pion-pole Et CFF requires axial parity p=-1")
        if np.any(np.asarray(self.xi) == 0):
            raise ValueError("The pion-pole Et CFF is undefined at xi=0")
        if p_order not in (1, 2):
            raise ValueError("p_order must be 1 (LO) or 2 (NLO)")

        fmask = flvmask(flv)
        if fmask is None:
            raise ValueError(f"Unknown CFF flavor projection: {flv}")

        residue_u = self.pion_pole_Et_GFF_j0("u", N, Lambda)
        residue_d = self.pion_pole_Et_GFF_j0("d", N, Lambda)
        fixed_moment = np.array(
            [[0.0, residue_u / 2.0, 0.0, residue_d / 2.0, 0.0]],
            dtype=float,
        )

        if p_order == 1:
            evolved_with_coefficient = CFF_Evo_LO( np.array([0.0]), NFEFF, self.p, self.Q, fixed_moment )
        else:
            # The NLO evolution kernels are evaluated infinitesimally above
            # j=0, following the existing discrete axial-j=0 CFF treatment.
            evolved_with_coefficient = CFF_Evo_NLO_evWC( np.array([1.0e-6]), NFEFF, self.p, self.Q, fixed_moment, muf)

        projected = np.einsum(
            "i,...i->...", fmask, evolved_with_coefficient
        )[0]
        return 2.0 * projected / self.xi

    def dterm_C_GFF_j1(
            self, flv: str, ParaDterm: np.ndarray,
            multipole: float = 2.0) -> float:
        r"""Return the fixed-:math:`j=1` C form factor
        :math:`C_{20}^{\mathrm{flv}}(t)`.

        .. math::

            C_{20}^{\mathrm{flv}}(t)
            = C_{20}^{\mathrm{flv}}(0)
              (1-t\,\mathrm{invm2})^{-p}.

        ``ParaDterm`` contains one ``[norm, invm2]`` row for each physical
        source flavor in the order ``[u, d, g]``.

        Args:
            flv: Physical source flavor ``'u'``, ``'d'``, or ``'g'``.
            ParaDterm: Shape ``(3, 2)`` with rows
                ``[[N_u, invm2_u], [N_d, invm2_d], [N_g, invm2_g]]``.
            multipole: Positive multipole power; ``2`` (dipole) by default.

        Returns:
            The requested flavor's :math:`C_{20}(t)` form factor.
        """
        if self.p != 1:
            raise ValueError("The C form factor requires vector parity p=1")
        flavor_index = {"u": 0, "d": 1, "g": 2}
        if flv not in flavor_index:
            raise ValueError(f"Unknown C-form-factor flavor: {flv}")
        ParaDterm = np.asarray(ParaDterm)
        if ParaDterm.shape != (3, 2):
            raise ValueError(
                "ParaDterm must have shape (3, 2) ordered as [u, d, g]"
            )
        norm, invm2 = ParaDterm[flavor_index[flv]]
        if not np.isfinite(norm):
            raise ValueError("C-form-factor norm must be finite")
        if not np.isfinite(invm2) or invm2 < 0:
            raise ValueError("invm2 must be finite and nonnegative")
        if not np.isfinite(multipole) or multipole <= 0:
            raise ValueError("multipole must be a positive finite number")

        denominator = 1.0 - self.t * invm2
        if denominator <= 0:
            raise ValueError(
                "The C-form-factor multipole denominator must be positive"
            )
        return norm * denominator ** (-multipole)

    def dterm_CFF_j1(
            self, spe: int, ParaDterm: np.ndarray, muf: float,
            p_order: int = 1,
            multipole: float = 2.0) -> complex:
        r"""Return the full fixed-:math:`j=1` C-form-factor contribution
        to :math:`\mathcal H` or :math:`\mathcal E`.

        The method obtains :math:`C_{20}^{u,d,g}(t)` from
        :meth:`dterm_C_GFF_j1`.  Polynomiality converts them to the fixed
        conformal coefficients :math:`F_{1,2}^{u,d,g}=4C_{20}^{u,d,g}`,
        which are embedded together as

        .. math::

            [0,F_{1,2}^u/2,0,F_{1,2}^d/2,F_{1,2}^g].

        The complete flavor vector is evolved in one call from GUMP's input
        scale at the physical index :math:`j=1`, so singlet--gluon mixing is
        retained before combining with the existing LO or NLO DVCS Wilson
        coefficient.  The fixed vector residue contributes the final factor
        two and no residual power of :math:`\xi`.

        Args:
            spe: ``0`` for :math:`\mathcal H` or ``1`` for
                :math:`\mathcal E=-\mathcal H`.
            ParaDterm: Shape ``(3, 2)`` with one ``[norm, invm2]`` row for
                each physical source flavor in the order ``[u, d, g]``.
            muf: Factorization scale in GeV.
            p_order: ``1`` for LO or ``2`` for NLO.
            multipole: Positive multipole power; ``2`` (dipole) by default.

        Returns:
            The charge-weighted CFF contribution summed over all flavors.
        """
        if self.p != 1:
            raise ValueError("The C-form-factor CFF requires vector parity p=1")
        if spe not in (0, 1):
            raise ValueError("C-form-factor CFF spe must be 0 (H) or 1 (E)")
        if p_order not in (1, 2):
            raise ValueError("p_order must be 1 (LO) or 2 (NLO)")

        C20 = np.array([
            self.dterm_C_GFF_j1(
                flv, ParaDterm, multipole=multipole
            )
            for flv in ("u", "d", "g")
        ])
        F12 = 4.0 * C20
        he_sign = 1.0 if spe == 0 else -1.0
        fixed_moment = np.zeros((1, 5), dtype=float)
        fixed_moment[0, 1] = he_sign * F12[0] / 2.0
        fixed_moment[0, 3] = he_sign * F12[1] / 2.0
        fixed_moment[0, 4] = he_sign * F12[2]

        j1 = np.array([1.0])
        if p_order == 1:
            evolved_with_coefficient = CFF_Evo_LO(
                j1, NFEFF, self.p, self.Q, fixed_moment
            )
        else:
            evolved_with_coefficient = CFF_Evo_NLO_evWC(
                j1, NFEFF, self.p, self.Q, fixed_moment, muf
            )

        projected = np.einsum(
            "i,...i->...", flvmask("All"), evolved_with_coefficient
        )[0]
        return 2.0 * projected

    def dterm_TFF_j1(
            self, spe: int, ParaDterm: np.ndarray, muf: float, meson: int,
            p_order: int = 1,
            multipole: float = 2.0) -> complex:
        r"""Return the full fixed-:math:`j=1` C-form-factor contribution
        to the :math:`H`- or :math:`E`-type transition form factor.

        As in :meth:`dterm_CFF_j1`, the three physical source form factors
        :math:`C_{20}^{u,d,g}(t)` are converted to
        :math:`F_{1,2}^{u,d,g}=4C_{20}^{u,d,g}` and embedded together as

        .. math::

            [0,F_{1,2}^u/2,0,F_{1,2}^d/2,F_{1,2}^g].

        Passing the complete vector through the existing DVMP evolution and
        Wilson-coefficient routines retains both quark evolution and
        singlet--gluon mixing.  At NLO the coefficient is evaluated
        infinitesimally above the physical index :math:`j=1`, following the
        existing discrete-:math:`j=1` TFF prescription.  The fixed-vector
        residue supplies the final factor two and leaves no residual power
        of :math:`\xi`.

        Args:
            spe: ``0`` for the :math:`H` contribution or ``1`` for the
                opposite-sign :math:`E` contribution.
            ParaDterm: Shape ``(3, 2)`` with one ``[norm, invm2]`` row for
                each physical source flavor in the order ``[u, d, g]``.
            muf: Factorization scale in GeV.
            meson: Meson code used by the existing TFF routines.
            p_order: ``1`` for LO or ``2`` for NLO.
            multipole: Positive multipole power; ``2`` (dipole) by default.

        Returns:
            The meson-weighted TFF contribution summed over all flavors.
        """
        if self.p != 1:
            raise ValueError("The C-form-factor TFF requires vector parity p=1")
        if spe not in (0, 1):
            raise ValueError("C-form-factor TFF spe must be 0 (H) or 1 (E)")
        if p_order not in (1, 2):
            raise ValueError("p_order must be 1 (LO) or 2 (NLO)")

        C20 = np.array([
            self.dterm_C_GFF_j1(
                flv, ParaDterm, multipole=multipole
            )
            for flv in ("u", "d", "g")
        ])
        F12 = 4.0 * C20
        he_sign = 1.0 if spe == 0 else -1.0
        fixed_moment = np.zeros((1, 5), dtype=float)
        fixed_moment[0, 1] = he_sign * F12[0] / 2.0
        fixed_moment[0, 3] = he_sign * F12[1] / 2.0
        fixed_moment[0, 4] = he_sign * F12[2]

        j1 = np.array([1.0])
        if p_order == 1:
            evolved_with_coefficient = TFF_Evo_LO(
                j1, NFEFF, self.p, self.Q, fixed_moment, meson
            )
        else:
            # WilsonCoef_DVMP_NLO has a removable singularity at exactly
            # j=1; use the established finite-limit prescription.
            evolved_with_coefficient = TFF_Evo_NLO_evWC(
                j1 + 1.0e-6, NFEFF, self.p, self.Q,
                fixed_moment, meson, muf
            )

        projected = np.einsum(
            "i,...i->...", flvmask("All"), evolved_with_coefficient
        )[0]
        return 2.0 * projected
