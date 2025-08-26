import time, os, csv
import numpy as np
import pandas as pd
from .Minimizer import off_forward_fit_withH_withHt, cost_off_forward_withH_withHt, Paralst_Unp_Names, Paralst_Pol_Names,Paralst_Aux_Names
from . import config 
dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    
    print('Test16')
    time_start = time.time()

    #'''
    Paralst_Unp=pd.read_csv(os.path.join(dir_path,f'GUMP_Params/Para_Unp_Off_forward_withH_withHt_NLO.csv'), header=None).to_numpy()[0]
    Paralst_Pol=pd.read_csv(os.path.join(dir_path,f'GUMP_Params/Para_Pol_Off_forward_withH_withHt_NLO.csv'), header=None).to_numpy()[0]
    
    config.INC_gGFF = True
    config.INC_JPSI = True
    str = '_withH_withHt_NLO_withJpsi'
    
    fit_off_forward = off_forward_fit_withH_withHt(Paralst_Unp, Paralst_Pol)
    
    FitVals = list([*fit_off_forward.values])
    FitErrs = list([*fit_off_forward.errors])
    UnpLength = len(Paralst_Unp)
    
    with open(os.path.join(dir_path,f"GUMP_Params/Para_Unp_Off_forward{str}.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerow(FitVals[:UnpLength])
        csvWriter.writerow(FitErrs[:UnpLength])
        print(f"off-forward fit unpolarized parameters saved to Para_Unp_Off_forward{str}.csv")
    
    with open(os.path.join(dir_path,f"GUMP_Params/Para_Pol_Off_forward{str}.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerow(FitVals[UnpLength:])
        csvWriter.writerow(FitErrs[UnpLength:])
        print(f"off-forward fit polarized parameters saved to Para_Pol_Off_forward{str}.csv")
    #'''

    #
    # Below is for testing, set Export_Mode to True in config.py and run through to generate the outputs
    #

    '''
    str = '_withH_withHt_NLO'
    config.Export_Mode = True
    config.INC_gGFF = True
    config.INC_JPSI = False
    
    Paralst_Unp=pd.read_csv(os.path.join(dir_path,f'GUMP_Params/Para_Unp_Off_forward{str}.csv'), header=None).to_numpy()[0]
    Paralst_Pol=pd.read_csv(os.path.join(dir_path,f'GUMP_Params/Para_Pol_Off_forward{str}.csv'), header=None).to_numpy()[0]
    Paralst_Aux = [1.0] * len(Paralst_Aux_Names)

    params_unp = dict(zip(Paralst_Unp_Names, Paralst_Unp))
    params_pol = dict(zip(Paralst_Pol_Names, Paralst_Pol))
    params_aux = dict(zip(Paralst_Aux_Names, Paralst_Aux))
    
    params = {**params_unp, **params_pol, **params_aux}

    print(cost_off_forward_withH_withHt(**params))
    '''