"""

Here we define the GPD ansatz based on the moment space expression norm * x^ alpha  * (1-x)^ beta.

Actual GPDs require different GPD types (H, E, Htilde, Etilde), flavor (u, d, g), sea/valence and more xi dependence (xi^2, xi^4 ...),
which can be obtained by defining different ansatz for each of them. 

"""

import scipy as sp
import numpy as np

# reference scale to be 2 GeV

FactorizationScaleMu0 = 2

# Define beta function with complex arguments

def beta_loggamma(a, b):
    return np.exp(sp.special.loggamma(a) + sp.special.loggamma(b)-sp.special.loggamma(a + b))

class GPDMoment(object) :
    # Class for conformal moment in j space
    def __init__(self, init_norm, init_alpha, init_beta, init_alphap) -> None:
        # Initial values for the PDF parameters from the input
        self.norm = init_norm
        self.alpha = init_alpha
        self.beta = init_beta
        self.alphap = init_alphap
        # Initialization of scale as the reference scale mu0 = 2 GeV
        self.mu0 = FactorizationScaleMu0

    def ConfMoment(self , j , t):
        return self.norm * beta_loggamma ( j + 1 - self.alpha, 1 + self.beta) * (j + 1 - self.alpha)/ (j + 1 - self.alpha - self.alphap * t)
