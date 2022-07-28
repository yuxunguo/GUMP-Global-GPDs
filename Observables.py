"""
Observables calculation with LO evolution and (u,d,g) GPDs ansatz
"""

import scipy as sp
import numpy as np
import MomentAnsatz

class GPDobserv (object) :
    def __init__(self, init_x , init_xi, init_t, init_Q) -> None:
        self.x = init_x
        self.xi = init_xi
        self.t = init_t
        self.Q = init_Q
    
    def tPDFevo(Fu: MomentAnsatz.GPDMoment, Fd : MomentAnsatz.GPDMoment,Fg : MomentAnsatz.GPDMoment):
        return Fu, Fd, Fg
