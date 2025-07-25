"""
Here we define the GPD ansatz based on the moment space expression norm * x^ alpha  * (1-x)^ beta.
With the GPDs ansatz, observables with LO evolution are calculated
"""
import scipy as sp
import numpy as np
from scipy.integrate import quad_vec, fixed_quad
from scipy.special import gamma
from Evolution import Moment_Evo_LO,Moment_Evo_LO_NSp1, TFF_Evo_LO, CFF_Evo_LO, TFF_Evo_NLO_evWC, TFF_Evo_NLO_evMOM, CFF_Evo_NLO_evWC,CFF_Evo_NLO_evMOM, GPD_Moment_Evo_NLO,tPDF_Moment_Evo_NLO, tPDF_Moment_Evo_NLO_NSp1, fixed_quadvec, inv_flav_trans
from Parameters import Moment_Sum
from Evolution import ConfWaveFuncQ, ConfWaveFuncG, ConfWaveFuncQ_over_sinpij, ConfWaveFuncG_over_sinpij

"""
***********************GPD moments***************************************
"""
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
   

def flv_to_indx(flv:str):
    """flv is the flavor. It is a string
    
    Args:
        flv (str): the flavor in string, e.g., 'u', 'd' , 'g'

    Returns:
        flv (scalar): flavor converted to scalar 
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

def flvs_to_indx(flvs):
    """ Cast :func:`flv_to_indx` for each flv in the list flvs

    Args:
        flvs (list of str): list of flavors
    Returns:
        flvs (list of scalar): flavors converted to scalar 
    """

    output = [flv_to_indx(flv) for flv in flvs]
    return np.array(output, dtype=np.int32)

#The flavor interpreter to return the corresponding flavor combination 
def Flv_Intp(Flv_array: np.array, flv):
    """Return the wave function for each flavor
    
    Flv_array: (N, 3) complex, the wave function of u,d,g
    flv: (N) str

    return result: (N) complex
    """
    _flv_index = flv_to_indx(flv)
    return np.choose(_flv_index, [Flv_array[...,0], Flv_array[..., 1], Flv_array[..., 2],\
                        Flv_array[..., 0]-Flv_array[..., 1], Flv_array[..., 0]+Flv_array[..., 1]])
    # return np.einsum('...j,...j', Flv_array, _helper) # (N)
    
# This is in evolution basis!!!
def flvmask(flv: str):
    if (flv == 'All'):
        return np.array([1,1,1,1,1])
    elif (flv == 'g'):
        return np.array([0,0,0,0,1])
    elif (flv == 'q'):
        return np.array([1,1,1,1,0])
    
def InvMellinWaveFuncQ(s: complex, x: float) -> complex:
    """ Quark wave function for inverse Mellin transformation: x^(-s) for x>0 and 0 for x<0

    Args:
        s: Mellin moment s (= n)
        x: momentum fraction x

    Returns:
        Wave function for inverse Mellin transformation: x^(-s) for x>0 and 0 for x<0

    vectorized version of InvMellinWaveFuncQ
    """ 

    ''' 
    if(x > 0):
        return x ** (-s)
    
    return 0
    '''
    return np.where(x>0, x**(-s), 0)


def InvMellinWaveFuncG(s: complex, x: float) -> complex:
    """ Gluon wave function for inverse Mellin transformation: x^(-s+1) for x>0 and 0 for x<0

    Args:
        s: Mellin moment s (= n)
        x: momentum fraction x

    Returns:
        Gluon wave function for inverse Mellin transformation: x^(-s+1) for x>0 and 0 for x<0
    """  

    '''
    if(x > 0):
        return x ** (-s+1)
    
    return 0
    '''
    return np.where(x>0, x**(-s+1), 0)


"""
***********************Observables***************************************
"""

# Class for observables
class GPDobserv (object) :
    #Initialization of observables. Each is a function of (x, xi ,t, Q), p for parity: p = 1 for vector GPDs (H, E) and p = -1 for axial-vector GPDs (Ht, Et)
    def __init__(self, init_x: float, init_xi: float, init_t: float, init_Q: float, p: int) -> None:
        """Initialization of GPD-related observables

        Args:
            init_x (float): the momentum fraction x
            init_xi (float): the skewness parameter xi
            init_t (float): the momentum transfer square t
            init_Q (float): photon virtuality/the scale of this observables
            p (int): p=1 for vector-like GPDs and p=-1 for axial-vector ones
        """
        self.x = init_x
        self.xi = init_xi
        self.t = init_t
        self.Q = init_Q
        self.p = p

    def tPDF(self, flv, ParaAll, p_order = 1):
        """t-denpendent PDF for given flavor 
        
        Cross-ref: :func:`Evolution.Moment_Evo_LO` and :func:`Evolution.tPDF_Moment_Evo_NLO`
        
        Args:
            flv (str): "u", "d", "S", "NS" or "g"
            ParaAll: array as [Para_Forward, Para_xi2,...] 
            
                - Para_Forward: array as [Para_Forward_uV, Para_Forward_ubar, Para_Forward_dV, Para_Forward_dbar, Para_Forward_g],
                  where Para_Forward_i is parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                - Para_xi2: only matter for non-zero xi (NOT needed here but the parameters are passed for consistency with GPDs)
                 
            p_order: 1 for leading-order evolution (default); 2 for next-to-leading-order evolution ; higher order not implemented yet

        Returns:
            f(x,t) in for the given flavor
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

            '''
            InvMellinWaveC = np.array([[InvMellinWaveFuncQ(s, self.x), InvMellinWaveFuncQ(s, self.x) - self.p * InvMellinWaveFuncQ(s, -self.x),0,0,0],
                                       [0,0,InvMellinWaveFuncQ(s, self.x), InvMellinWaveFuncQ(s, self.x) - self.p * InvMellinWaveFuncQ(s, -self.x),0],
                                       [0,0,0,0,(InvMellinWaveFuncG(s, self.x)+ self.p * InvMellinWaveFuncG(s, -self.x))]]) # (3, 5) matrix
                                       # in my case, I want it to be (N, 3, 5) ndarray
            '''

            # InvMellinWaveFuncQ(s, self.x) # shape (N)
            # self.p * InvMellinWaveFuncQ(s, -self.x) # shape (N)

            helper1 = np.array([[1, 1, 0, 0, 0],
                                [0, 0, 1, 1, 0],
                                [0, 0, 0, 0, 0]])
            helper2 = np.array([[0, -1, 0, 0, 0],
                                [0, 0, 0, -1, 0],
                                [0, 0, 0, 0, 0]])
            helper3 = np.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1]])

            InvMellinWaveC =np.einsum('..., ij->...ij', InvMellinWaveFuncQ(s, self.x), helper1) \
                            + np.einsum('... ,ij->...ij', self.p * InvMellinWaveFuncQ(s, -self.x), helper2) \
                            + np.einsum('... ,ij->...ij', (InvMellinWaveFuncG(s, self.x)+ self.p * InvMellinWaveFuncG(s, -self.x)), helper3)

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

    def GPD(self, flv, ParaAll, p_order = 1):
        """GPD F(x, xi, t) in flavor space (flv = "u", "d", "S", "NS" or "g")
        
        Args:
            flv (str): "u", "d", "S", "NS" or "g"
            ParaAll: array as [Para_Forward, Para_xi2, Para_xi4]
            
                - Para_Forward = array as [Para_Forward_uV, Para_Forward_ubar, Para_Forward_dV, Para_Forward_dbar, Para_Forward_g]
                  where Para_Forward_i is forward parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                - Para_xi2 = array as [Para_xi2_uV, Para_xi2_ubar, Para_xi2_dV, Para_xi2_dbar, Para_xi2_g]
                  where Para_xi2_i is xi^2 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                - Para_xi4 = array as [Para_xi4_uV, Para_xi4_ubar, Para_xi4_dV, Para_xi4_dbar, Para_xi4_g]
                  where Para_xi4_i is xi^4 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
            p_order: 1 for leading-order evolution (default); 2 for next-to-leading-order evolution ; higher order not implemented yet

        Returns:
            f(x,xi,t) for given flavor flv
        """
        #[Para_Forward, Para_xi2, Para_xi4] = ParaAll
        if (p_order == 2):
            return self.GPDNLO_evMom(flv, ParaAll)
        
        Para_Forward = ParaAll[..., 0, :, :, :]  # each (N, 5, 1, 5)
        Para_xi2     = ParaAll[..., 1, :, :, :]
        Para_xi4     = ParaAll[..., 2, :, :, :]

        # The contour for Mellin-Barnes integral in terms of j not n.         
        def ConfWaveConv(j: complex):
            """
            ConfWaveC = np.array([[ConfWaveFuncQ(j, self.x, self.xi), ConfWaveFuncQ(j, self.x, self.xi) - self.p * ConfWaveFuncQ(j, -self.x, self.xi),0,0,0],
                                  [0,0,ConfWaveFuncQ(j, self.x, self.xi), ConfWaveFuncQ(j, self.x, self.xi) - self.p * ConfWaveFuncQ(j, -self.x, self.xi),0],
                                  [0,0,0,0,ConfWaveFuncG(j, self.x, self.xi)+ self.p * ConfWaveFuncG(j, -self.x, self.xi)]])
            """

            helper1 = np.array([[1, 1, 0, 0, 0],
                                [0, 0, 1, 1, 0],
                                [0, 0, 0, 0, 0]])
            helper2 = np.array([[0, -1, 0, 0, 0],
                                [0, 0, 0, -1, 0],
                                [0, 0, 0, 0, 0]])
            helper3 = np.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1]])

            ConfWaveC =np.einsum('..., ij->...ij', ConfWaveFuncQ(j, self.x, self.xi), helper1) \
                     + np.einsum('... ,ij->...ij', self.p * ConfWaveFuncQ(j, -self.x, self.xi), helper2) \
                     + np.einsum('... ,ij->...ij', ConfWaveFuncG(j, self.x, self.xi)+ self.p * ConfWaveFuncG(j, -self.x, self.xi), helper3)

            return ConfWaveC
        
        def ConfWaveConv_over_sinpiji(j: complex):
            """
            This function exists for technical reasons:
            
            The 1/sin(pi*(j+1)) factor is exponentially suppressed on the imaginary axes on both side,
            whereas the conformal partial wave function is exponentiall divergent.
            So we absorb 1/sin(pi*(j+1)) factor into the conformal partial wave function in advance to avoid the overflow in the conformal wave function.
            This is only need for Mellin-Barnes integral!
            """
            helper1 = np.array([[1, 1, 0, 0, 0],
                                [0, 0, 1, 1, 0],
                                [0, 0, 0, 0, 0]])
            helper2 = np.array([[0, -1, 0, 0, 0],
                                [0, 0, 0, -1, 0],
                                [0, 0, 0, 0, 0]])
            helper3 = np.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1]])

            ConfWaveC =np.einsum('..., ij->...ij', ConfWaveFuncQ_over_sinpij(j, self.x, self.xi), helper1) \
                     + np.einsum('... ,ij->...ij', self.p * ConfWaveFuncQ_over_sinpij(j, -self.x, self.xi), helper2) \
                     + np.einsum('... ,ij->...ij', ConfWaveFuncG_over_sinpij(j, self.x, self.xi)+ self.p * ConfWaveFuncG_over_sinpij(j, -self.x, self.xi), helper3)

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
        return 1/2*np.real(fixed_quadvec(lambda imJ : Integrand_Mellin_Barnes(reJ + 1j* imJ) + Integrand_Mellin_Barnes(reJ - 1j* imJ),0, Max_imJ, n=800)) + np.real(GPD0()) 
    
    def GPDNLO_evMom(self, flv, ParaAll):
        """GPD F(x, xi, t) in flavor space (flv = "u", "d", "S", "NS" or "g")
        
        Args:
            flv (str): "u", "d", "S", "NS" or "g"
            ParaAll: array as [Para_Forward, Para_xi2, Para_xi4]
            
                - Para_Forward = array as [Para_Forward_uV, Para_Forward_ubar, Para_Forward_dV, Para_Forward_dbar, Para_Forward_g]
                  where Para_Forward_i is forward parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                - Para_xi2 = array as [Para_xi2_uV, Para_xi2_ubar, Para_xi2_dV, Para_xi2_dbar, Para_xi2_g]
                  where Para_xi2_i is xi^2 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                - Para_xi4 = array as [Para_xi4_uV, Para_xi4_ubar, Para_xi4_dV, Para_xi4_dbar, Para_xi4_g]
                  where Para_xi4_i is xi^4 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
            p_order: 1 for leading-order evolution (default); 2 for next-to-leading-order evolution ; higher order not implemented yet

        Returns:
            f(x,xi,t) for given flavor flv
        """
        
        Para_Forward = ParaAll[..., 0, :, :, :]  # each (N, 5, 1, 5)
        Para_xi2     = ParaAll[..., 1, :, :, :]
        Para_xi4     = ParaAll[..., 2, :, :, :]

        # The contour for Mellin-Barnes integral in terms of j not n.         
        def ConfWaveConv(j: complex):
            """
            ConfWaveC = np.array([[ConfWaveFuncQ(j, self.x, self.xi), ConfWaveFuncQ(j, self.x, self.xi) - self.p * ConfWaveFuncQ(j, -self.x, self.xi),0,0,0],
                                  [0,0,ConfWaveFuncQ(j, self.x, self.xi), ConfWaveFuncQ(j, self.x, self.xi) - self.p * ConfWaveFuncQ(j, -self.x, self.xi),0],
                                  [0,0,0,0,ConfWaveFuncG(j, self.x, self.xi)+ self.p * ConfWaveFuncG(j, -self.x, self.xi)]])
            """

            helper1 = np.array([[1, 1, 0, 0, 0],
                                [0, 0, 1, 1, 0],
                                [0, 0, 0, 0, 0]])
            helper2 = np.array([[0, -1, 0, 0, 0],
                                [0, 0, 0, -1, 0],
                                [0, 0, 0, 0, 0]])
            helper3 = np.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1]])

            ConfWaveC =np.einsum('..., ij->...ij', ConfWaveFuncQ(j, self.x, self.xi), helper1) \
                     + np.einsum('... ,ij->...ij', self.p * ConfWaveFuncQ(j, -self.x, self.xi), helper2) \
                     + np.einsum('... ,ij->...ij', ConfWaveFuncG(j, self.x, self.xi)+ self.p * ConfWaveFuncG(j, -self.x, self.xi), helper3)

            return ConfWaveC
        
        def ConfWaveConv_over_sinpiji(j: complex):
            """
            This function exists for technical reasons:
            
            The 1/sin(pi*(j+1)) factor is exponentially suppressed on the imaginary axes on both side,
            whereas the conformal partial wave function is exponentiall divergent.
            So we absorb 1/sin(pi*(j+1)) factor into the conformal partial wave function in advance to avoid the overflow in the conformal wave function.
            This is only need for Mellin-Barnes integral!
            """
            helper1 = np.array([[1, 1, 0, 0, 0],
                                [0, 0, 1, 1, 0],
                                [0, 0, 0, 0, 0]])
            helper2 = np.array([[0, -1, 0, 0, 0],
                                [0, 0, 0, -1, 0],
                                [0, 0, 0, 0, 0]])
            helper3 = np.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1]])

            ConfWaveC =np.einsum('..., ij->...ij', ConfWaveFuncQ_over_sinpij(j, self.x, self.xi), helper1) \
                     + np.einsum('... ,ij->...ij', self.p * ConfWaveFuncQ_over_sinpij(j, -self.x, self.xi), helper2) \
                     + np.einsum('... ,ij->...ij', ConfWaveFuncG_over_sinpij(j, self.x, self.xi)+ self.p * ConfWaveFuncG_over_sinpij(j, -self.x, self.xi), helper3)

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
        return 1/2*np.real(fixed_quadvec(lambda imJ : Integrand_Mellin_Barnes(reJ + 1j* imJ) + Integrand_Mellin_Barnes(reJ - 1j* imJ),0, Max_imJ, n=800)) + np.real(GPD0()) 
    
    def GFFj0(self, j: int, flv, ParaAll, p_order):
        """Generalized Form Factors A_{j0}(t) which is the xi^0 term of the nth (n= j+1) Mellin moment of GPD int dx x^j F(x,xi,t) for quark and int dx x^(j-1) F(x,xi,t) for gluon
            
            Note for gluon, GPD reduce to x*g(x), not g(x) so the Mellin moment will have a mismatch
            
            Only leading order implemented yet NOT well-maintained (TO BE FIXED)
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
    
    def CFF(self, ParaAll, muf, p_order = 1, flv = 'All'):
        """Charge averged CFF $\mathcal{F}(xi, t) (\mathcal{F} = Q_u^2 F_u + Q_d^2 F_d)$
        
        Args:
            ParaAll: array as [Para_Forward, Para_xi2, Para_xi4]
            
                - Para_Forward = array as [Para_Forward_uV, Para_Forward_ubar, Para_Forward_dV, Para_Forward_dbar, Para_Forward_g]
                  where Para_Forward_i is forward parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                - Para_xi2 = array as [Para_xi2_uV, Para_xi2_ubar, Para_xi2_dV, Para_xi2_dbar, Para_xi2_g]
                  where Para_xi2_i is xi^2 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                - Para_xi4 = array as [Para_xi4_uV, Para_xi4_ubar, Para_xi4_dV, Para_xi4_dbar, Para_xi4_g]
                  where Para_xi4_i is xi^4 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
            muf: factorization scale
            p_order: 1 for leading-order evolution (default); 2 for next-to-leading-order evolution ; higher order not implemented yet
            flv: "q", "g", or "All"

        Returns:
            CFF \mathcal{F}(xi, t) = Q_u^2 F_u + Q_d^2 F_d
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

    def TFF(self, ParaAll, muf , meson, p_order = 1, flv = 'All'):
        """TFF $\mathcal{F}(xi, t) (\mathcal{F}$
        
        Args:
            ParaAll: array as [Para_Forward, Para_xi2, Para_xi4]
        
                - Para_Forward = array as [Para_Forward_uV, Para_Forward_ubar, Para_Forward_dV, Para_Forward_dbar, Para_Forward_g]
                    where Para_Forward_i is forward parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                - Para_xi2 = array as [Para_xi2_uV, Para_xi2_ubar, Para_xi2_dV, Para_xi2_dbar, Para_xi2_g]
                    where Para_xi2_i is xi^2 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                - Para_xi4 = array as [Para_xi4_uV, Para_xi4_ubar, Para_xi4_dV, Para_xi4_dbar, Para_xi4_g]
                    where Para_xi4_i is xi^4 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
            muf: factorization scale
            meson: [1 for rho, 2 for phi, 3 for jpsi]
            p_order: 1 for leading-order evolution (default); 2 for next-to-leading-order evolution ; higher order not implemented yet
            flv: "q", "g", or "All"
            
        Returns:
            TFF \mathcal{F}(xi, t)
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
    
    # !!! Working in progress
    def CFFNLO(self, ParaAll, muf: float, flv = 'All'):
        """CFF $\mathcal{F}(xi, t) (\mathcal{F}$
        
        A separate function for next-to-leading order CFF, it can be called directly or with CFF() by setting p_order = 2 
        
        Args:
            ParaAll: array as [Para_Forward, Para_xi2, Para_xi4]
        
                - Para_Forward = array as [Para_Forward_uV, Para_Forward_ubar, Para_Forward_dV, Para_Forward_dbar, Para_Forward_g]
                    where Para_Forward_i is forward parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                - Para_xi2 = array as [Para_xi2_uV, Para_xi2_ubar, Para_xi2_dV, Para_xi2_dbar, Para_xi2_g]
                    where Para_xi2_i is xi^2 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                - Para_xi4 = array as [Para_xi4_uV, Para_xi4_ubar, Para_xi4_dV, Para_xi4_dbar, Para_xi4_g]
                    where Para_xi4_i is xi^4 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
            muf: factorization scale
            flv: "q", "g", or "All"
            
        Returns:
            CFF \mathcal{F}(xi, t)
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

    # !!! Working in progress
    def CFFNLO_evMom(self, ParaAll, muf: float, flv = 'All'):
        """NLOCFF $\mathcal{F}(xi, t) (\mathcal{F}) $
        
        A different function for next-to-leading order TFF using evolved moment method, it can be checked that it generate the same results as the evolve Wilson coefficient method
        
        Args:
            ParaAll: array as [Para_Forward, Para_xi2, Para_xi4]
        
                - Para_Forward = array as [Para_Forward_uV, Para_Forward_ubar, Para_Forward_dV, Para_Forward_dbar, Para_Forward_g]
                    where Para_Forward_i is forward parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                - Para_xi2 = array as [Para_xi2_uV, Para_xi2_ubar, Para_xi2_dV, Para_xi2_dbar, Para_xi2_g]
                    where Para_xi2_i is xi^2 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                - Para_xi4 = array as [Para_xi4_uV, Para_xi4_ubar, Para_xi4_dV, Para_xi4_dbar, Para_xi4_g]
                    where Para_xi4_i is xi^4 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
            muf: factorization scale
            flv: "q", "g", or "All"
            
        Returns:
            CFF \mathcal{F}(xi, t)
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
    
    def TFFNLO(self, ParaAll, muf: float, meson: int, flv = 'All'):
        """TFF $\mathcal{F}(xi, t) (\mathcal{F}$
        
        A separate function for next-to-leading order TFF, it can be called directly or with TFF() by setting p_order = 2 
        
        Args:
            ParaAll: array as [Para_Forward, Para_xi2, Para_xi4]
        
                - Para_Forward = array as [Para_Forward_uV, Para_Forward_ubar, Para_Forward_dV, Para_Forward_dbar, Para_Forward_g]
                    where Para_Forward_i is forward parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                - Para_xi2 = array as [Para_xi2_uV, Para_xi2_ubar, Para_xi2_dV, Para_xi2_dbar, Para_xi2_g]
                    where Para_xi2_i is xi^2 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                - Para_xi4 = array as [Para_xi4_uV, Para_xi4_ubar, Para_xi4_dV, Para_xi4_dbar, Para_xi4_g]
                    where Para_xi4_i is xi^4 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
            muf: factorization scale
            meson: [1 for rho, 2 for phi, 3 for jpsi]
            flv: "q", "g", or "All"
            
        Returns:
            TFF \mathcal{F}(xi, t)
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

    def TFFNLO_evMom(self, ParaAll, muf: float, meson: int, flv = 'All'):
        """NLOTFF $\mathcal{F}(xi, t) (\mathcal{F}) $
        
        A different function for next-to-leading order TFF using evolved moment method, it can be checked that it generate the same results as the evolve Wilson coefficient method
        
        Args:
            ParaAll: array as [Para_Forward, Para_xi2, Para_xi4]
        
                - Para_Forward = array as [Para_Forward_uV, Para_Forward_ubar, Para_Forward_dV, Para_Forward_dbar, Para_Forward_g]
                    where Para_Forward_i is forward parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                - Para_xi2 = array as [Para_xi2_uV, Para_xi2_ubar, Para_xi2_dV, Para_xi2_dbar, Para_xi2_g]
                    where Para_xi2_i is xi^2 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                - Para_xi4 = array as [Para_xi4_uV, Para_xi4_ubar, Para_xi4_dV, Para_xi4_dbar, Para_xi4_g]
                    where Para_xi4_i is xi^4 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
            muf: factorization scale
            meson: [1 for rho, 2 for phi, 3 for jpsi]
            flv: "q", "g", or "All"
            
        Returns:
            TFF \mathcal{F}(xi, t)
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