# ==========================================================================================
# In this example, we show how to call the gumpgpd package to 
# generate all the data to be used in the analysis as well as the theoretical predictions
#
# This package is written in a half-integrated way purposely, so the parameters can be varied if necessary
# ===========================================================================================
import pandas as pd
import os, csv
import time

# Import the package
import gumpgpd as gp

# Setting the config before importing gumpgpd.Minimizer to ensure correct initialization
gp.config.Export_Mode = True

# Some functions from the Minimizer modules that we used to call the saved best-fit parameters and data
<<<<<<< Updated upstream
from gumpgpd.Minimizer import Paralst_Unp_Names, Paralst_Pol_Names,Paralst_Aux_Names, cost_off_forward_withH_withHt, Paralst_Unp_off_forward, Paralst_Pol_off_forward
=======
from gumpgpd.Minimizer import Paralst_Unp_Names, Paralst_Pol_Names, Paralst_PionPole_Names, Paralst_Dterm_Names, Paralst_Aux_Names, cost_off_forward_withH_withHt, Paralst_Unp_off_forward, Paralst_Pol_off_forward, Paralst_PionPole_off_forward, Paralst_Dterm_off_forward
>>>>>>> Stashed changes

# Path to save the output into csv files
dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':

    # Set SAVE_TO_FILE to True if exported csv files are needed
    gp.Minimizer.SAVE_TO_FILE = True
    # Set the path for the export file, the default is the package directory 
    # gp.Minimizer.SAVE_TO_FILE_PATH = dir_path
    
    # Retrieve the best-fit parameters
    Paralst_Unp = Paralst_Unp_off_forward
    # The last parameter of Paralst_Pol_off_forward is the auxiliary parameter not used in the analysis
    Paralst_Pol = Paralst_Pol_off_forward[:-1]
    Paralst_Aux = [Paralst_Pol_off_forward[-1]]
<<<<<<< Updated upstream
=======
    Paralst_PionPole = Paralst_PionPole_off_forward
    Paralst_Dterm = Paralst_Dterm_off_forward
>>>>>>> Stashed changes

    # Sythesis them with the parameter names into a dictionary
    params_unp = dict(zip(Paralst_Unp_Names, Paralst_Unp))
    params_pol = dict(zip(Paralst_Pol_Names, Paralst_Pol))
<<<<<<< Updated upstream
    params_aux = dict(zip(Paralst_Aux_Names, Paralst_Aux))

    # Put all of them together
    params = {**params_unp, **params_pol, **params_aux}
=======
    params_pionpole = dict(zip(Paralst_PionPole_Names, Paralst_PionPole))
    params_dterm = dict(zip(Paralst_Dterm_Names, Paralst_Dterm))
    params_aux = dict(zip(Paralst_Aux_Names, Paralst_Aux))

    # Put all of them together
    params = {**params_unp, **params_pol, **params_pionpole, **params_dterm, **params_aux}
>>>>>>> Stashed changes
    print("Running code to generate all results, could take a few minutes")
    t0 = time.perf_counter()
    grouped_results = cost_off_forward_withH_withHt(**params)
    t1 = time.perf_counter()
    print(f"Elapsed time: {t1 - t0:.6f} s")
    '''
    for name, df in grouped_results.items():
        print(f"\n===== {name} =====")
        print(df.head())
    '''

