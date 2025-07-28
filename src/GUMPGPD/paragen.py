import time, os, csv
import numpy as np
import pandas as pd
from .Minimizer import forward_H_fit, forward_Ht_fit, off_forward_fit,off_forward_fit_withHt, off_forward_fit_withH_withHt, cost_forward_H, cost_forward_Ht, cost_off_forward, cost_off_forward_withHt, Paralst_Unp, Paralst_Pol, Paralst_Unp_Names, Paralst_Pol_Names, Paralst_Unp_off_forward, Paralst_Pol_off_forward
from . import config 
dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    
    print('Test12')
    time_start = time.time()
    #'''
    fit_forward_H   = forward_H_fit(Paralst_Unp)
    Paralst_Unp     = np.array(fit_forward_H.values)
    
    with open(os.path.join(dir_path,"GUMP_Params/Para_Unp.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerow(list([*fit_forward_H.values]))
        csvWriter.writerow(list([*fit_forward_H.errors]))
        print("H fit parameters saved to Para_Unp.csv")
    #'''
    #'''
    fit_forward_Ht  = forward_Ht_fit(Paralst_Pol)
    Paralst_Pol     = np.array(fit_forward_Ht.values)

    with open(os.path.join(dir_path,"GUMP_Params/Para_Pol.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerow(list([*fit_forward_Ht.values]))
        csvWriter.writerow(list([*fit_forward_Ht.errors]))
        print("Ht fit parameters saved to Para_Pol.csv")
    #'''
    #'''
    str = ''
    fit_off_forward = off_forward_fit(Paralst_Unp, Paralst_Pol)
    
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
    config.Export_Mode = True
    
    params_unp = dict(zip(Paralst_Unp_Names, Paralst_Unp_off_forward))
    params_pol = dict(zip(Paralst_Pol_Names, Paralst_Pol_off_forward))
    params = {**params_unp, **params_pol}

    print(cost_forward_H(**params_unp))
    print(cost_forward_Ht(**params_pol))
    print(cost_off_forward(**params))
    '''