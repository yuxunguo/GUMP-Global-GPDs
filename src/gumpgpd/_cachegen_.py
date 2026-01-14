from .Minimizer import Paralst_Unp_Names, Paralst_Pol_Names,Paralst_Aux_Names, cost_off_forward_withH_withHt, Paralst_Unp_off_forward, Paralst_Pol_off_forward
from ._helper_ import gump_msg
if __name__ == '__main__':
    
    # Retrieve the best-fit parameters
    Paralst_Unp = Paralst_Unp_off_forward
    # The last parameter of Paralst_Pol_off_forward is the auxiliary parameter not used in the analysis
    Paralst_Pol = Paralst_Pol_off_forward[:-1]
    Paralst_Aux = [Paralst_Pol_off_forward[-1]]

    # Sythesis them with the parameter names into a dictionary
    params_unp = dict(zip(Paralst_Unp_Names, Paralst_Unp))
    params_pol = dict(zip(Paralst_Pol_Names, Paralst_Pol))
    params_aux = dict(zip(Paralst_Aux_Names, Paralst_Aux))

    # Put all of them together
    params = {**params_unp, **params_pol, **params_aux}
    gump_msg("Calling the cost function to generate needed cached data. This could take several minutes...", level="INFO")

    gump_msg("Large memory usage is expected during this process, especially if no cache exists yet.", level="INFO")

    cost_off_forward_withH_withHt(**params)
    
    gump_msg("Cached data generation completed! No need to run this again unless the evolution or Wilson coefficients module changes.", level="INFO")