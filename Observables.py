"""
Here we define the GPD ansatz based on the moment space expression norm * x^ alpha  * (1-x)^ beta.
With the GPDs ansatz, observables with LO evolution are calculated
"""
import scipy as sp
import numpy as np
from mpmath import mp, hyp2f1
from scipy.integrate import quad_vec, fixed_quad
from scipy.special import gamma
from Evolution import Moment_Evo_LO,TFF_Evo_LO, CFF_Evo_LO, TFF_Evo_NLO_evWC, TFF_Evo_NLO_evMOM, GPD_Moment_Evo_NLO,tPDF_Moment_Evo_NLO, fixed_quadvec, inv_flav_trans

CFF_trans =np.array([1*(2/3)**2, 2*(2/3)**2, 1*(1/3)**2, 2*(1/3)**2, 0])

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

Flavor_Factor = 2 * 2 + 1

def flv_to_indx(flv:str):
    '''
    flv is the flavor. It is a string

    This function will cast each flavor to an auxiliary length 3 array

    Output shape: scalar int    

    '''
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
    '''flvs is an array of strings (N)
    Output: (N)
    '''
    output = [flv_to_indx(flv) for flv in flvs]
    return np.array(output, dtype=np.int32)

#The flavor interpreter to return the corresponding flavor combination 
def Flv_Intp(Flv_array: np.array, flv):

    """
    Flv_array: (N, 3) complex
    flv: (N) str

    return result: (N) complex
    """
    
    '''
    if(flv == "u"):
        return Flv_array[0]
    if(flv == "d"):
        return Flv_array[1]
    if(flv == "g"):
        return Flv_array[2]
    if(flv == "NS"):
        return Flv_array[0] - Flv_array[1]
    if(flv == "S"):
        return Flv_array[0] + Flv_array[1]
    '''
    _flv_index = flvs_to_indx(flv)
    return np.choose(_flv_index, [Flv_array[...,0], Flv_array[..., 1], Flv_array[..., 2],\
                        Flv_array[..., 0]-Flv_array[..., 1], Flv_array[..., 0]+Flv_array[..., 1]])
    # return np.einsum('...j,...j', Flv_array, _helper) # (N)

def flvmask(flv: str):
    if (flv == 'All'):
        return np.array([1,1,1,1,1])
    elif (flv == 'u'):
        return np.array([1,1,0,0,0])
    elif (flv == 'd'):
        return np.array([0,0,1,1,0])
    elif (flv == 'g'):
        return np.array([0,0,0,0,1])
    elif (flv == 'q'):
        return np.array([1,1,1,1,0])
    
# Euler Beta function B(a,b) with complex arguments
def beta_loggamma(a: complex, b: complex) -> complex:
    return np.exp(sp.special.loggamma(a) + sp.special.loggamma(b)-sp.special.loggamma(a + b))

# Conformal moment in j space F(j)
def ConfMoment(j: complex, t: float, ParaSets: np.ndarray):
    """
    Conformal moment in j space F(j)

    Args:
        ParaSet is the array of parameters in the form of (norm, alpha, beta, alphap)
            norm = ParaSet[0]: overall normalization constant
            alpha = ParaSet[1], beta = ParaSet[2]: the two parameters corresponding to x ^ (-alpha) * (1 - x) ^ beta
            alphap = ParaSet[3]: regge trajectory alpha(t) = alpha + alphap * t
        j: conformal spin j (conformal spin is actually j+2 but anyway)
        t: momentum transfer squared t

    Originally, ParaSet have shape (5).
    Output is a scalar
    
    After vectorization, there are a few different usages:
        1. t has shape (N), ParaSet has shape (N, 5). Output have shape (N)
        2. t has shape (N), ParaSet has shape (N, m1, 5). Output have shape (N, m1)
        3. t has shape (N), ParaSet has shape (N, m1, m2, 5). Output have shape (N, m1, m2)
        4. and so on
        
    Recommended usage:
        t has shape (N), ParaSet has shape (N, 5, init_NumofAnsatz, 5)
        output will be (N, 5, init_NumofAnsatz)

    j should only be a scalar. It cannot be an ndarray before I make more changes
    Right now, j can be a vector, but if you want to do integration, then better pass j as scalar.

    Returns:
        Conformal moment in j space F(j,t)
    """
    
    # [norm, alpha, beta, alphap, bexp] = ParaSet
    norm = ParaSets[..., 0]  # in recommended usage, has shape (N, 5, init_NumofAnsatz)
    alpha = ParaSets[..., 1] # in general, can have shape (N), (N, m1), (N, m1, m2), ......
    beta  = ParaSets[..., 2]
    alphap = ParaSets[..., 3]
    bexp = ParaSets[..., 4]
    
    if np.ndim(norm) < np.ndim(t):
        raise ValueError("Input format is wrong.")
    
    t_new_shape = list(np.shape(t)) + [1]*(np.ndim(norm) - np.ndim(t))
    j_new_shape = list(np.shape(j)) + [1]*(np.ndim(norm) - np.ndim(t))  # not a typo, it is np.ndim(norm) - np.ndim(t)
    t = np.reshape(t, t_new_shape) # to make sure t can be broadcasted with norm, alpha, etc.
    # t will have shape (N) or (N, m1) or (N, m1, m2)... depends
    j = np.reshape(j, j_new_shape)

    # Currently with KM ansatz and dipole residual

    return norm * beta_loggamma (j + 1 - alpha, 1 + beta) * (j + 1  - alpha)/ (j + 1 - alpha - alphap * t) * np.exp(t*bexp)
    # (N) or (N, m1) or (N, m1, m2) .... depends on usage
    # For the recommended usage, the output is (N, 5, init_NumofAnsatz)

def Moment_Sum(j: complex, t: float, ParaSets: np.ndarray) -> complex:
    """
    Sum of the conformal moments when the ParaSets contain more than just one set of parameters 

    Args:
        ParaSets : contains [ParaSet1, ParaSet0, ParaSet2,...] with each ParaSet = [norm, alpha, beta ,alphap] for valence and sea distributions repsectively.        
        j: conformal spin j (or j+2 but anyway)
        t: momentum transfer squared t

        Originally, ParaSets have shape (init_NumofAnsatz, 4). In practice, it is (1, 4)
        output is a scalar

        Here, after vectorization, 
            t has shape (N)
            ParaSets has (N, 5, init_NumofAnsatz, 5)
            output will have shape (N, 5)  # here, the 5 means 5 different species [u - ubar, ubar, d - dbar, dbar, g]
        
        More generally, 
            t have shape (N)
            ParaSets has shape (N, i, 5 ) or (N, m1, i, 5) or (N, m1, m2, i, 5) and so on
            output will have shape (N) or (N, m1) or (N, m2)...... etc.

    Returns:
        sum of conformal moments over all the ParaSet
    """

    #return np.sum(np.array( list(map(lambda paraset: ConfMoment(j, t, paraset), ParaSets)) ))

    # ConfMoment_vec(j, t, ParaSets) should have shape (N, 5, init_NumofAnsatz)
    return np.sum(ConfMoment(j, t, ParaSets) ,  axis=-1) # (N, 5)


# precision for the hypergeometric function
mp.dps = 25

hyp2f1_nparray = np.frompyfunc(hyp2f1,4,1)

def InvMellinWaveFuncQ(s: complex, x: float) -> complex:
    """ 
    Quark wave function for inverse Mellin transformation: x^(-s) for x>0 and 0 for x<0

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
    """ 
    Gluon wave function for inverse Mellin transformation: x^(-s+1) for x>0 and 0 for x<0

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

def ConfWaveFuncQ(j: complex, x: float, xi: float) -> complex:
    """ 
    Quark conformal wave function p_j(x,xi) check e.g. https://arxiv.org/pdf/hep-ph/0509204.pdf

    Args:
        j: conformal spin j (conformal spin is actually j+2 but anyway)
        x: momentum fraction x
        xi: skewness parameter xi

    Returns:
        quark conformal wave function p_j(x,xi)
    """  
    """
    if(x > xi):
        pDGLAP = np.sin(np.pi * (j+1))/ np.pi * x**(-j-1) * complex(hyp2f1( (j+1)/2, (j+2)/2, j+5/2, (xi/x) ** 2)) 
        return pDGLAP

    if(x > -xi):
        pERBL = 2 ** (1+j) * gamma(5/2+j) / (gamma(1/2) * gamma(1+j)) * xi ** (-j-1) * (1+x/xi) * complex(hyp2f1(-1-j,j+2,2, (x+xi)/(2*xi)))
        return pERBL
    
    return 0
    """

    pDGLAP = np.where(x >= xi,                np.sin(np.pi * (j+1))/ np.pi * x**(-j-1) * np.array(hyp2f1_nparray( (j+1)/2, (j+2)/2, j+5/2, (xi/x) ** 2), dtype= complex)                           , 0)

    pERBL = np.where(((x > -xi) & (x < xi)), 2 ** (1+j) * gamma(5/2+j) / (gamma(1/2) * gamma(1+j)) * xi ** (-j-1) * (1+x/xi) * np.array(hyp2f1_nparray(-1-j,j+2,2, (x+xi)/(2*xi)), dtype= complex), 0)

    return pDGLAP + pERBL


def ConfWaveFuncG(j: complex, x: float, xi: float) -> complex:
    """ 
    Gluon conformal wave function p_j(x,xi) check e.g. https://arxiv.org/pdf/hep-ph/0509204.pdf

    Args:
        j: conformal spin j (actually conformal spin is j+2 but anyway)
        x: momentum fraction x
        xi: skewness parameter xi

    Returns:
        gluon conformal wave function p_j(x,xi)
    """ 
    # An extra minus sign defined different from the orginal definition to absorb the extra minus sign of MB integral for gluon
    """
    Minus = -1
    if(x > xi):
        pDGLAP = np.sin(np.pi * j)/ np.pi * x**(-j) * complex(hyp2f1( j/2, (j+1)/2, j+5/2, (xi/x) ** 2)) 
        return Minus * pDGLAP

    if(x > -xi):
        pERBL = 2 ** j * gamma(5/2+j) / (gamma(1/2) * gamma(j)) * xi ** (-j) * (1+x/xi) ** 2 * complex(hyp2f1(-1-j,j+2,3, (x+xi)/(2*xi)))
        return Minus * pERBL
    
    return 0
    """
    Minus = -1

    pDGLAP = np.where(x >= xi,                Minus * np.sin(np.pi * j)/ np.pi * x**(-j) * np.array(hyp2f1_nparray( j/2, (j+1)/2, j+5/2, (xi/x) ** 2), dtype= complex)                                   , 0)

    pERBL = np.where(((x > -xi) & (x < xi)), Minus * 2 ** j * gamma(5/2+j) / (gamma(1/2) * gamma(j)) * xi ** (-j) * (1+x/xi) ** 2 * np.array((hyp2f1_nparray(-1-j,j+2,3, (x+xi)/(2*xi))), dtype= complex), 0)

    return pDGLAP + pERBL

"""
***********************Observables***************************************
"""

# Class for observables
class GPDobserv (object) :
    #Initialization of observables. Each is a function of (x, xi ,t, Q), p for parity: p = 1 for vector GPDs (H, E) and p = -1 for axial-vector GPDs (Ht, Et)
    def __init__(self, init_x: float, init_xi: float, init_t: float, init_Q: float, p: int) -> None:
        self.x = init_x
        self.xi = init_xi
        self.t = init_t
        self.Q = init_Q
        self.p = p

    def tPDF(self, flv, ParaAll, p_order = 1):
        """
        t-denpendent PDF for given flavor (flv = "u", "d", "S", "NS" or "g")
        Args:
            ParaAll = [Para_Forward, Para_xi2]
            Para_Forward = [Para_Forward_uV, Para_Forward_ubar, Para_Forward_dV, Para_Forward_dbar, Para_Forward_g]
            Para_Forward_i: parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
            Para_xi2: only matter for non-zero xi (NOT needed here but the parameters are passed for consistency with GPDs)
            
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

        # The contour for inverse Meliin transform. Note that S here is the analytically continued n which is j + 1 not j !
        reS = inv_Mellin_intercept + 1
        Max_imS = inv_Mellin_cutoff 

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
                ConfEv = tPDF_Moment_Evo_NLO(s - 1, NFEFF, self.p, self.Q, ConfFlav, muset = 1)
            
            # Inverse transform the evolved moments back to the flavor basis
            EvoConfFlav = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv) #(N, 5)
    
            # Return the evolved moments with x^(-s) for quark or x^(-s+1) for gluon for the given flavor flv = "u", "d", "S", "NS" or "g"
            #  InvMellinWaveConf(s): (N, 3, 5)            
            # the result of np.einsum will be (N, 3)
            # Flv_Intp  result (N)
            return Flv_Intp(np.einsum('...ij,...j->...i', InvMellinWaveConf(s), EvoConfFlav), flv)

        return 1/(2 * np.pi) * np.real(fixed_quadvec(lambda imS : Integrand_inv_Mellin(reS + 1j * imS) + Integrand_inv_Mellin(reS - 1j * imS) ,0, + Max_imS, n=200))

    def GPD(self, flv, ParaAll, p_order = 1):
        """
        GPD F(x, xi, t) in flavor space (flv = "u", "d", "S", "NS" or "g")
        Args:
            ParaAll = [Para_Forward, Para_xi2, Para_xi4]
            Para_Forward = [Para_Forward_uV, Para_Forward_ubar, Para_Forward_dV, Para_Forward_dbar, Para_Forward_g]
            Para_Forward_i: forward parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
            Para_xi2 = [Para_xi2_uV, Para_xi2_ubar, Para_xi2_dV, Para_xi2_dbar, Para_xi2_g]
            Para_xi2_i: xi^2 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
            Para_xi4 = [Para_xi4_uV, Para_xi4_ubar, Para_xi4_dV, Para_xi4_dbar, Para_xi4_g]
            Para_xi4_i: xi^4 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
            
            p_order: 1 for leading-order evolution (default); 2 for next-to-leading-order evolution ; higher order not implemented yet

        Returns:
            f(x,xi,t) for given flavor flv
        """
        #[Para_Forward, Para_xi2, Para_xi4] = ParaAll

        Para_Forward = ParaAll[..., 0, :, :, :]  # each (N, 5, 1, 5)
        Para_xi2     = ParaAll[..., 1, :, :, :]
        Para_xi4     = ParaAll[..., 2, :, :, :]
        '''
        # Removing xi^4 terms
        Para_xi4     = ParaAll[..., 2, :, :, :]
        '''

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
        
        def Integrand_Mellin_Barnes(j: complex):

            ConfFlav     = Moment_Sum(j, self.t, Para_Forward) #(N, 5)
            ConfFlav_xi2 = Moment_Sum(j, self.t, Para_xi2)
            ConfFlav_xi4 = Moment_Sum(j, self.t, Para_xi4)
            
            #Evolve the conformal moments for different p_order
            if (p_order == 1):
                ConfEv     = Moment_Evo_LO(j, NFEFF, self.p, self.Q, ConfFlav)
                ConfEv_xi2 = Moment_Evo_LO(j+2, NFEFF, self.p, self.Q, ConfFlav_xi2)
                ConfEv_xi4 = Moment_Evo_LO(j+4, NFEFF, self.p, self.Q, ConfFlav_xi4)
                
            elif (p_order == 2):
                ConfEv     = GPD_Moment_Evo_NLO(j, NFEFF, self.p, self.Q, self.t, self.xi, ConfFlav, Para_Forward,0)
                ConfEv_xi2 = GPD_Moment_Evo_NLO(j+2, NFEFF, self.p, self.Q, self.t, self.xi, ConfFlav_xi2, Para_xi2,2)
                ConfEv_xi4 = GPD_Moment_Evo_NLO(j+4, NFEFF, self.p, self.Q, self.t, self.xi, ConfFlav_xi4, Para_xi4,4)
                
            # Inverse transform the evolved moments back to the flavor basis
            ConfFlavEv     = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv) #(N, 5)
            ConfFlavEv_xi2 = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv_xi2)
            ConfFlavEv_xi4 = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv_xi4)
            
            return Flv_Intp(np.einsum('...ij,...j->...i', ConfWaveConv(j), ConfFlavEv) \
                    + self.xi ** 2 * np.einsum('...ij,...j->...i', ConfWaveConv(j+2), ConfFlavEv_xi2) \
                    + self.xi ** 4 * np.einsum('...ij,...j->...i', ConfWaveConv(j+4), ConfFlavEv_xi4),flv)
        
        # Adding a j = 1 term because the contour do not enclose the j = 1 pole which should be the 1th conformal moment.
        def GPD1():
            eps= np.array([0.])
            return (-1)*Integrand_Mellin_Barnes(1.+eps) # Note the residual theorem gives -(2 np.pi *1j)*1/(2*np.sin((j+1)*np.pi)) with residual (-1) at j=1;
  
        # Adding a j = 0 term because the contour do not enclose the j = 0 pole which should be the 0th conformal moment.
        def GPD0():
            '''
            Note: Naively, this function simply returns Integrand_Mellin_Barnes([0.]) like the GPD1() above.
                  However, the zeroth moment is only defined for valence quark not sea quark or gluon
                  Thus there will be divergences in moment when j = 0.       
                  Here we use nan_to_num to set all the divergence to zero, such that the zero moment of the sea quark and gluon do not contribute to GPD.
                  
                  The better choice is to model the leading moment terms separately, and fit them to other quantities since those terms are not well constrained by the CFF/TFF anyway.
            '''
            j0 = 1+np.array([0.])
            
            ConfFlav     = Moment_Sum(j0, self.t, Para_Forward)
            ConfFlav_xi2 = Moment_Sum(j0, self.t, Para_xi2)
            ConfFlav_xi4 = Moment_Sum(j0, self.t, Para_xi4)  

            ConfFlav     = np.nan_to_num(ConfFlav)
            ConfFlav_xi2 = np.nan_to_num(ConfFlav_xi2) 
            ConfFlav_xi4 = np.nan_to_num(ConfFlav_xi4)  
                
            if (p_order == 1):
                ConfEv     = Moment_Evo_LO(j0, NFEFF, self.p, self.Q, ConfFlav)
                ConfEv_xi2 = Moment_Evo_LO(j0+2, NFEFF, self.p, self.Q, ConfFlav_xi2)
                ConfEv_xi4 = Moment_Evo_LO(j0+4, NFEFF, self.p, self.Q, ConfFlav_xi4)
            # Use tPDF_Moment_Evo_NLO since the off-diagonal piece only contribute for j>=2
            elif (p_order == 2):
                ConfEv     = tPDF_Moment_Evo_NLO(j0, NFEFF, self.p, self.Q, ConfFlav)
                ConfEv_xi2 = tPDF_Moment_Evo_NLO(j0+2, NFEFF, self.p, self.Q, ConfFlav_xi2)
                ConfEv_xi4 = tPDF_Moment_Evo_NLO(j0+4, NFEFF, self.p, self.Q, ConfFlav_xi4)
            
            ConfFlavEv     = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv) #(N, 5)
            ConfFlavEv_xi2 = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv_xi2)
            ConfFlavEv_xi4 = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv_xi4)
              
            return Flv_Intp(np.einsum('...ij,...j->...i', ConfWaveConv(j0), ConfFlavEv) \
                    + self.xi ** 2 * np.einsum('...ij,...j->...i', ConfWaveConv(j0+2), ConfFlavEv_xi2) \
                    + self.xi ** 4 * np.einsum('...ij,...j->...i', ConfWaveConv(j0+4), ConfFlavEv_xi4),flv)
       
        reJ = 2 - 0.15
        Max_imJ = Mellin_Barnes_cutoff
        return 1/2*np.real(fixed_quadvec(lambda imJ : Integrand_Mellin_Barnes(reJ + 1j* imJ) / np.sin((reJ + 1j * imJ+1) * np.pi) + Integrand_Mellin_Barnes(reJ - 1j* imJ) / np.sin((reJ - 1j * imJ+1) * np.pi) ,0, Max_imJ, n=200)) + np.real(GPD1())  + np.real(GPD0()) 
          
    def GFFj0(self, j: int, flv, ParaAll, p_order):
        """
            Generalized Form Factors A_{j0}(t) which is the xi^0 term of the nth (n= j+1) Mellin moment of GPD int dx x^j F(x,xi,t) for quark and int dx x^(j-1) F(x,xi,t) for gluon
            Note for gluon, GPD reduce to x*g(x), not g(x) so the Mellin moment will have a mismatch
            
            Only leading order implemented yet
        """

        # j, flv both have shape (N)
        # ParaAll: (N, 3, 5, 1, 5)

        '''
        Para_Forward = ParaAll[0]
        
        GFF_trans = np.array([[1,1 - self.p * (-1) ** j,0,0,0],
                              [0,0,1,1 - self.p * (-1) ** j,0],
                              [0,0,0,0,(1 - self.p * (-1) ** j)/2]])

        ConfFlav = np.array( list(map(lambda paraset: Moment_Sum(j, self.t, paraset), Para_Forward)) )

        if (j == 0):
            if(self.p == 1):
                return Flv_Intp( np.array([ConfFlav[0],ConfFlav[2],ConfFlav[4]]) , flv)
        
        return Flv_Intp(np.einsum('...j,j', GFF_trans, Moment_Evo(j, NFEFF, self.p, self.Q, ConfFlav)), flv)
        '''

        Para_Forward = ParaAll[..., 0, :, :, :]  # (N, 5, 1, 5)
        _helper1 = np.array([[1, 1, 0, 0, 0],
                             [0, 0, 1, 1, 0],
                             [0, 0, 0, 0, 1/2]])
        _helper2 = np.array([[0, -1, 0, 0, 0],
                             [0, 0, 0, -1, 0],
                             [0, 0, 0, 0, -1/2]])
        GFF_trans = np.einsum('... , ij->...ij', self.p * (-1)**j, _helper2) + _helper1  # (N, 3, 5)
        ConfFlav = Moment_Sum(j, self.t, Para_Forward) # (N, 5)

        # the result of np.einsum will be (N, 3)
        # Flv_Intp  result (N)
        mask = ((j==0) & (self.p==1))
        result = np.empty_like(self.Q)

        result[mask] = Flv_Intp(ConfFlav[mask][:, [0,2,4] ] , flv[mask] ) # (N_mask)
        
        if (p_order == 1):
            ConfEv = Moment_Evo_LO(j[~mask], NFEFF, self.p[~mask], self.Q[~mask], ConfFlav[~mask])
        elif (p_order == 2):
            ConfEv = tPDF_Moment_Evo_NLO(j[~mask], NFEFF, self.p[~mask], self.Q[~mask], ConfFlav[~mask], muset = 1)
            
        # Inverse transform the evolved moments back to the flavor basis
        EvoConfFlav = np.einsum('...ij, ...j->...i', inv_flav_trans, ConfEv) #(N, 5)       

        result[~mask] = Flv_Intp(np.einsum('...ij, ...j->...i', GFF_trans[~mask], EvoConfFlav[~mask]), flv[~mask]) # (N_~mask)
                
        return result #(N)
    
    def CFF(self, ParaAll):
        """
        Charge averged CFF \mathcal{F}(xi, t) (\mathcal{F} = Q_u^2 F_u + Q_d^2 F_d)
        Args:
            ParaAll = [Para_Forward, Para_xi2, Para_xi4]

            Para_Forward = [Para_Forward_uV, Para_Forward_ubar, Para_Forward_dV, Para_Forward_dbar, Para_Forward_g]
            Para_Forward_i: forward parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
            Para_xi2 = [Para_xi2_uV, Para_xi2_ubar, Para_xi2_dV, Para_xi2_dbar, Para_xi2_g]
            Para_xi2_i: xi^2 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
            Para_xi4 = [Para_xi4_uV, Para_xi4_ubar, Para_xi4_dV, Para_xi4_dbar, Para_xi4_g]
            Para_xi4_i: xi^4 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)

        Returns:
            CFF \mathcal{F}(xi, t) = Q_u^2 F_u + Q_d^2 F_d
        """
        #[Para_Forward, Para_xi2, Para_xi4] = ParaAll  # each (N, 5, 1, 5)
        Para_Forward = ParaAll[..., 0, :, :, :]  # each (N, 5, 1, 5)
        Para_xi2     = ParaAll[..., 1, :, :, :]
        '''
        # Removing xi^4 terms
        Para_xi4     = ParaAll[..., 2, :, :, :]
        '''

        # The contour for Mellin-Barnes integral in terms of j not n.
        reJ = Mellin_Barnes_intercept 
        Max_imJ = Mellin_Barnes_cutoff 

        def Integrand_Mellin_Barnes_CFF(j: complex):

            '''
            ConfFlav = np.array( list(map(lambda paraset: Moment_Sum(j, self.t, paraset), Para_Forward)) )
            ConfFlav_xi2 = np.array( list(map(lambda paraset: Moment_Sum(j, self.t, paraset), Para_xi2)) )
            ConfFlav_xi4 = np.array( list(map(lambda paraset: Moment_Sum(j, self.t, paraset), Para_xi4)) )
            '''
            ConfFlav     = Moment_Sum(j, self.t, Para_Forward) #(N, 5)
            ConfFlav_xi2 = Moment_Sum(j, self.t, Para_xi2)
            '''
            # Removing xi^4 terms
            ConfFlav_xi4 = Moment_Sum(j, self.t, Para_xi4)
            '''

            # shape (N, 5)
            """
            # Removing xi^4 terms
            EvoConf_Wilson = (CWilson(j) * Moment_Evo(j, NFEFF, self.p, self.Q, ConfFlav) \
                                + CWilson(j+2) * Moment_Evo(j+2, NFEFF, self.p, self.Q, ConfFlav_xi2) \
                                + CWilson(j+4) * Moment_Evo(j+4, NFEFF, self.p, self.Q, ConfFlav_xi4))
            """
            EvoConf_Wilson = (CFF_Evo_LO(j, NFEFF, self.p, self.Q, ConfFlav) \
                                + CFF_Evo_LO(j+2, NFEFF, self.p, self.Q, ConfFlav_xi2))
            
            return np.einsum('j, ...j', CFF_trans, EvoConf_Wilson) # shape (N)

        def Integrand_CFF(imJ: complex):
            # mask = (self.p==1) # assume p can only be either 1 or -1

            result = np.ones_like(self.p) * self.xi ** (-reJ - 1j * imJ - 1) * Integrand_Mellin_Barnes_CFF(reJ + 1j * imJ) / 2

            if self.p==1:
                result *= (1j + np.tan((reJ + 1j * imJ) * np.pi / 2))
            else:
                result *= (1j - 1/np.tan((reJ + 1j * imJ) * np.pi / 2))
            '''
            if np.ndim(result)>0:
                result[mask] *= (1j + np.tan((reJ + 1j * imJ) * np.pi / 2))
                result[~mask] *= (1j - 1/np.tan((reJ + 1j * imJ) * np.pi / 2))
            else: # scalar
                if self.p==1:
                    result *= (1j + np.tan((reJ + 1j * imJ) * np.pi / 2))
                else:
                    result *= (1j - 1/np.tan((reJ + 1j * imJ) * np.pi / 2))
            '''
            return result

        # Adding extra j = 0 term for the axial vector CFFs
        def CFFj0():

            if self.p==1:
                result = np.ones_like(self.p) * 0
            else:
                result = np.ones_like(self.p) * self.xi ** (- 1) * Integrand_Mellin_Barnes_CFF(np.array([0.])) *(2)

            return result

        return fixed_quadvec(lambda imJ: Integrand_CFF(imJ)+Integrand_CFF(-imJ), 0,  Max_imJ, n=200) + CFFj0()

        '''
        if (self.p == 1):
            return quad_vec(lambda imJ : self.xi ** (-reJ - 1j * imJ - 1) * (1j + np.tan((reJ + 1j * imJ) * np.pi / 2)) *Integrand_Mellin_Barnes_CFF(reJ + 1j * imJ) / 2, - Max_imJ, + Max_imJ, epsrel = Prec_Goal)[0]
        
        if (self.p == -1):
            return quad_vec(lambda imJ : self.xi ** (-reJ - 1j * imJ - 1) * (1j - 1/np.tan((reJ + 1j * imJ) * np.pi / 2)) *Integrand_Mellin_Barnes_CFF(reJ + 1j * imJ) / 2, - Max_imJ, + Max_imJ, epsrel = Prec_Goal)[0]
        '''
    
    def TFF(self, ParaAll, meson, p_order = 1, muset = 1, flv = 'All'):
            """
            TFF \mathcal{F}(xi, t) (\mathcal{F} 
            Args:
                ParaAll = [Para_Forward, Para_xi2, Para_xi4]

                Para_Forward = [Para_Forward_uV, Para_Forward_ubar, Para_Forward_dV, Para_Forward_dbar, Para_Forward_g]
                Para_Forward_i: forward parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                Para_xi2 = [Para_xi2_uV, Para_xi2_ubar, Para_xi2_dV, Para_xi2_dbar, Para_xi2_g]
                Para_xi2_i: xi^2 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                Para_xi4 = [Para_xi4_uV, Para_xi4_ubar, Para_xi4_dV, Para_xi4_dbar, Para_xi4_g]
                Para_xi4_i: xi^4 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                
                meson = [1 for rho, 2 for phi, 3 for jpsi]
                
                p_order: 1 for leading-order evolution (default); 2 for next-to-leading-order evolution ; higher order not implemented yet
                
                (flv = "u", "d", "g", "All" or "q")
                
            Returns:
                TFF \mathcal{F}(xi, t)
            """
            if (p_order == 2):
                return self.TFFNLO(ParaAll, meson, muset, flv)
            
            #[Para_Forward, Para_xi2, Para_xi4] = ParaAll  # each (N, 5, 1, 5)
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
                
                EvoConf_Wilson = (TFF_Evo_LO(j, NFEFF, self.p, self.Q, ConfFlav, meson, muset) \
                                    + TFF_Evo_LO(j+2, NFEFF, self.p, self.Q, ConfFlav_xi2, meson, muset) \
                                        + TFF_Evo_LO(j+4, NFEFF, self.p, self.Q, ConfFlav_xi4, meson, muset))
                
                fmask = flvmask(flv)
                return np.einsum('j, ...j', fmask, EvoConf_Wilson)  
         
            def Integrand_TFF(imJ: complex):
                # mask = (self.p==1) # assume p can only be either 1 or -1

                result = np.ones_like(self.p) * self.xi ** (-reJ - 1j * imJ - 1) * Integrand_Mellin_Barnes_TFF(reJ + 1j * imJ) / 2

                if self.p==1:
                    result *= (1j + np.tan((reJ + 1j * imJ) * np.pi / 2))
                else:
                    result *= (1j - 1/np.tan((reJ + 1j * imJ) * np.pi / 2))
                '''
                if np.ndim(result)>0:
                    result[mask] *= (1j + np.tan((reJ + 1j * imJ) * np.pi / 2))
                    result[~mask] *= (1j - 1/np.tan((reJ + 1j * imJ) * np.pi / 2))
                else: # scalar
                    if self.p==1:
                        result *= (1j + np.tan((reJ + 1j * imJ) * np.pi / 2))
                    else:
                        result *= (1j - 1/np.tan((reJ + 1j * imJ) * np.pi / 2))
                '''
                return result

            # Adding extra j = 0 term for the axial vector CFFs
            def TFFj0():

                if self.p==1:
                    result = np.ones_like(self.p) * 0
                else:
                    result = np.ones_like(self.p) * self.xi ** (- 1) * Integrand_Mellin_Barnes_TFF(0) *(2)

                return result

            return fixed_quad(Integrand_TFF, - Max_imJ, + Max_imJ, n=1000)[0] + TFFj0()

            '''
            if (self.p == 1):
                return quad_vec(lambda imJ : self.xi ** (-reJ - 1j * imJ - 1) * (1j + np.tan((reJ + 1j * imJ) * np.pi / 2)) *Integrand_Mellin_Barnes_CFF(reJ + 1j * imJ) / 2, - Max_imJ, + Max_imJ, epsrel = Prec_Goal)[0]
            
            if (self.p == -1):
                return quad_vec(lambda imJ : self.xi ** (-reJ - 1j * imJ - 1) * (1j - 1/np.tan((reJ + 1j * imJ) * np.pi / 2)) *Integrand_Mellin_Barnes_CFF(reJ + 1j * imJ) / 2, - Max_imJ, + Max_imJ, epsrel = Prec_Goal)[0]
            '''

    # A separate function for next-to-leading order TFF, it can be called directly or with TFF() by setting p_order = 2 
    def TFFNLO(self, ParaAll, meson, muset: float = 1, flv = 'All'):
            """
            TFF \mathcal{F}(xi, t) (\mathcal{F} 
            Args:
                ParaAll = [Para_Forward, Para_xi2, Para_xi4]

                Para_Forward = [Para_Forward_uV, Para_Forward_ubar, Para_Forward_dV, Para_Forward_dbar, Para_Forward_g]
                Para_Forward_i: forward parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                Para_xi2 = [Para_xi2_uV, Para_xi2_ubar, Para_xi2_dV, Para_xi2_dbar, Para_xi2_g]
                Para_xi2_i: xi^2 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                Para_xi4 = [Para_xi4_uV, Para_xi4_ubar, Para_xi4_dV, Para_xi4_dbar, Para_xi4_g]
                Para_xi4_i: xi^4 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                
                meson = [1 for rho, 2 for phi, 3 for jpsi]
                                
                muset = offset of scale mu to study the scale dependence of the results
                
                (flv = "u", "d", "g", "All" or "q")
                
            Returns:
                TFF \mathcal{F}(xi, t)
            """
            #[Para_Forward, Para_xi2, Para_xi4] = ParaAll  # each (N, 5, 1, 5)
            Para_Forward = ParaAll[..., 0, :, :, :]  # each (N, 5, 1, 5)
            Para_xi2     = ParaAll[..., 1, :, :, :]
            Para_xi4     = ParaAll[..., 2, :, :, :]
            '''
            # Removing xi^4 terms
            Para_xi4     = ParaAll[..., 2, :, :, :]
            '''

            # The contour for Mellin-Barnes integral in terms of j not n.
            reJ = Mellin_Barnes_intercept 
            Max_imJ = Mellin_Barnes_cutoff
            
            def Integrand_Mellin_Barnes_TFF(j: complex):
                # j is a scalar

                '''
                ConfFlav = np.array( list(map(lambda paraset: Moment_Sum(j, self.t, paraset), Para_Forward)) )
                ConfFlav_xi2 = np.array( list(map(lambda paraset: Moment_Sum(j, self.t, paraset), Para_xi2)) )
                ConfFlav_xi4 = np.array( list(map(lambda paraset: Moment_Sum(j, self.t, paraset), Para_xi4)) )
                '''
                ConfFlav     = Moment_Sum(j, self.t, Para_Forward) #(N, 5)
                ConfFlav_xi2 = Moment_Sum(j, self.t, Para_xi2)
                ConfFlav_xi4 = Moment_Sum(j, self.t, Para_xi4)
                '''
                # Removing xi^4 terms
                ConfFlav_xi4 = Moment_Sum(j, self.t, Para_xi4)
                '''

                # shape (N, 5)
                """
                # Removing xi^4 terms
                EvoConf_Wilson = (CWilson(j) * Moment_Evo(j, NFEFF, self.p, self.Q, ConfFlav) \
                                    + CWilson(j+2) * Moment_Evo(j+2, NFEFF, self.p, self.Q, ConfFlav_xi2) \
                                    + CWilson(j+4) * Moment_Evo(j+4, NFEFF, self.p, self.Q, ConfFlav_xi4))
                """
                EvoConf_Wilson = (TFF_Evo_NLO_evWC(j, NFEFF, self.p, self.Q, ConfFlav, meson, muset) \
                                 +  TFF_Evo_NLO_evWC(j+2, NFEFF, self.p, self.Q, ConfFlav_xi2, meson, muset) \
                                 +  TFF_Evo_NLO_evWC(j+4, NFEFF, self.p, self.Q, ConfFlav_xi4, meson, muset))
                
                '''
                if(meson == 1):
                    return np.einsum('j, ...j', TFF_rho_trans, EvoConf_Wilson)                
                if(meson== 2):
                    return np.einsum('j, ...j', TFF_phi_trans, EvoConf_Wilson)                
                if(meson == 3):
                    return np.einsum('j, ...j', TFF_jpsi_trans, EvoConf_Wilson)
                '''                
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
                    return self.xi ** (- 1.) * Integrand_Mellin_Barnes_TFF(np.array([0.+eps])) *(2) # the last factor of 2 is the residual of -1/(2j)*np.cot(j * np.pi / 2) at j=0
            
            reJ = 1-0.8
            
            Max_imJ = 150
            
            return 1j*fixed_quadvec(lambda imJ: tan_factor(reJ+1j*imJ)*Integrand_Mellin_Barnes_TFF(reJ+1j*imJ)+tan_factor(reJ-1j*imJ)*Integrand_Mellin_Barnes_TFF(reJ-1j*imJ), 0, Max_imJ,n = 300) + TFFj0()

    # A different function for next-to-leading order TFF using evolved moment method, it can be checked that it generate the same results as the evolve Wilson coefficient method
    def TFFNLO_evMom(self, ParaAll, meson, muset: float = 1, flv = 'All'):
            """
            NLOTFF \mathcal{F}(xi, t) (\mathcal{F} 
            Args:
                ParaAll = [Para_Forward, Para_xi2, Para_xi4]

                Para_Forward = [Para_Forward_uV, Para_Forward_ubar, Para_Forward_dV, Para_Forward_dbar, Para_Forward_g]
                Para_Forward_i: forward parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                Para_xi2 = [Para_xi2_uV, Para_xi2_ubar, Para_xi2_dV, Para_xi2_dbar, Para_xi2_g]
                Para_xi2_i: xi^2 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                Para_xi4 = [Para_xi4_uV, Para_xi4_ubar, Para_xi4_dV, Para_xi4_dbar, Para_xi4_g]
                Para_xi4_i: xi^4 parameter sets for valence u quark (uV), sea u quark (ubar), valence d quark (dV), sea d quark (dbar) and gluon (g)
                
                meson = [1 for rho, 2 for phi, 3 for jpsi]
                
                muset = offset of scale mu to study the scale dependence of the results
                
                (flv = "u", "d", "g", "All" or "q")
                
            Returns:
                TFF \mathcal{F}(xi, t)
            """
            #[Para_Forward, Para_xi2, Para_xi4] = ParaAll  # each (N, 5, 1, 5)
            Para_Forward = ParaAll[..., 0, :, :, :]  # each (N, 5, 1, 5)
            Para_xi2     = ParaAll[..., 1, :, :, :]
            Para_xi4     = ParaAll[..., 2, :, :, :]
            '''
            # Removing xi^4 terms
            Para_xi4     = ParaAll[..., 2, :, :, :]
            '''

            # The contour for Mellin-Barnes integral in terms of j not n.
            reJ = Mellin_Barnes_intercept 
            Max_imJ = Mellin_Barnes_cutoff
            
            def Integrand_Mellin_Barnes_TFF(j: complex):
                # j is a scalar

                '''
                ConfFlav = np.array( list(map(lambda paraset: Moment_Sum(j, self.t, paraset), Para_Forward)) )
                ConfFlav_xi2 = np.array( list(map(lambda paraset: Moment_Sum(j, self.t, paraset), Para_xi2)) )
                ConfFlav_xi4 = np.array( list(map(lambda paraset: Moment_Sum(j, self.t, paraset), Para_xi4)) )
                '''
                ConfFlav     = Moment_Sum(j, self.t, Para_Forward) #(N, 5)
                ConfFlav_xi2 = Moment_Sum(j, self.t, Para_xi2)
                ConfFlav_xi4 = Moment_Sum(j, self.t, Para_xi4)
                '''
                # Removing xi^4 terms
                ConfFlav_xi4 = Moment_Sum(j, self.t, Para_xi4)
                '''

                # shape (N, 5)
                """
                # Removing xi^4 terms
                EvoConf_Wilson = (CWilson(j) * Moment_Evo(j, NFEFF, self.p, self.Q, ConfFlav) \
                                    + CWilson(j+2) * Moment_Evo(j+2, NFEFF, self.p, self.Q, ConfFlav_xi2) \
                                    + CWilson(j+4) * Moment_Evo(j+4, NFEFF, self.p, self.Q, ConfFlav_xi4))
                """
                EvoConf_Wilson = (TFF_Evo_NLO_evMOM(j, NFEFF, self.p, self.Q, self.t, self.xi, ConfFlav, Para_Forward, 0, meson, muset) \
                                          +  TFF_Evo_NLO_evMOM(j+2, NFEFF, self.p, self.Q, self.t, self.xi, ConfFlav_xi2, Para_xi2, 2, meson, muset) \
                                              +  TFF_Evo_NLO_evMOM(j+4, NFEFF, self.p, self.Q, self.t, self.xi, ConfFlav_xi4, Para_xi4, 4, meson, muset))
                '''
                if(meson == 1):
                    return np.einsum('j, ...j', TFF_rho_trans, EvoConf_Wilson)                
                if(meson== 2):
                    return np.einsum('j, ...j', TFF_phi_trans, EvoConf_Wilson)                
                if(meson == 3):
                    return np.einsum('j, ...j', TFF_jpsi_trans, EvoConf_Wilson)                
                '''
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
                    return self.xi ** (- 1.) * Integrand_Mellin_Barnes_TFF(np.array([0.+eps])) *(2) # the last factor of 2 is the residual of -1/(2j)*np.cot(j * np.pi / 2) at j=0
            
            #for moment evolution, the j=1 pole is also missed because we choose 1<cj<2.
            def TFFj1():

                if self.p==1:
                    return self.xi ** (- 2.) * Integrand_Mellin_Barnes_TFF(np.array([1.+eps])) *(2) # the last factor of 2 is the residual of 1/(2j)*np.tan(j * np.pi / 2) at j=1
                else:
                    return 0            
            
            reJ = 2. - 0.5
        
            Max_imJ = 150

            return 1j*fixed_quadvec(lambda imJ: tan_factor(reJ+1j*imJ)*Integrand_Mellin_Barnes_TFF(reJ+1j*imJ)+tan_factor(reJ-1j*imJ)*Integrand_Mellin_Barnes_TFF(reJ-1j*imJ), 0, Max_imJ,n = 400) + TFFj0() + TFFj1()