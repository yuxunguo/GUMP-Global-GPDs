"""

Here we define the GPD ansatz based on the moment space expression norm * x^ alpha  * (1-x)^ beta.

Actual GPDs require different GPD types (H, E, Htilde, Etilde), flavor (u, d, g), sea/valence and more xi dependence (xi^2, xi^4 ...),
which can be obtained by defining different ansatz for each of them. 

"""

import array as ary
import scipy as sp
import numpy as np
import time

# reference scale to be 2 GeV

FactorizationScaleMu0 = 2

# Define beta function with complex arguments

def beta_loggamma(a, b):
    return np.exp(sp.special.loggamma(a) + sp.special.loggamma(b)-sp.special.loggamma(a + b))

# Define integral of complex function (only real parts are needed)

def quadratureC(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    return (real_integral[0], real_integral[1:])

class GPDAnsatzPlus(object) :
    # Class for Generalized parton distributions (GPDs), parton distributions functions (PDFs), Compton form factors (CFFs) and more
     def __init__(self) -> None:
        # More TBA for the initialization of class
        self.norm = 1        
        self.alpha = 0.5
        self.beta = 3
        self.alphap = 1.5
        # Initialization of scale
        self.mu0 = FactorizationScaleMu0 

    def ConfMomentvalue(self , n , t , Q):
        return 0

    def tPDFvalue(self , x , t , Q):
        return 0

    def GPDvalue(self , x , xi , t , Q):
        return 0    

    def CFFvalue(self , xi , t , Q):
        return 0

    



