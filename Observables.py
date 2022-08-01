"""
Observables calculation with LO evolution and (u,d,g) GPDs ansatz
"""

import scipy as sp
import numpy as np
import MomentAnsatz

# number of ansatz,  combinations of multiple ansatz x^alpha (1-x)^beta might be needed for more flexiblity but 1 should be good to start with
init_NumofAnsatz = 1
# initial parameters for the non-singlet quark distributions
Para_NS

# initial parameters for the singlet quark distributions
Para_S

# initial parameters for the gluon distributions
Para_G

# 
class GPDobserv (object) :
    #Initialization of observables. Each is a function of (x, \xi ,t, Q) 
    def __init__(self, init_x , init_xi, init_t, init_Q) -> None:
        self.x = init_x
        self.xi = init_xi
        self.t = init_t
        self.Q = init_Q
        # number of ansatz,  combinations of multiple ansatz x^alpha (1-x)^beta might be needed for more flexiblity but 1 should be good to start with
        self.NumofAnsatz = init_NumofAnsatz

    #Non-singlet quark distributions in the moment space 
    def F_NS(self):

    def tPDFevo(self):
        return F_NS, F_S, Fg
