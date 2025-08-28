import time, os, csv
import numpy as np
import pandas as pd
from . import config 
config.INC_gGFF = False
config.INC_JPSI = False
from .Minimizer import off_forward_fit_withH_withHt

dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    
    print('Fit will be running, parameters will be generated to Minimizer.Paralst_Unp_off_forward and Minimizer.Paralst_Pol_off_forward')
    str = '_withH_withHt_NLO'
    Paralst_Unp=pd.read_csv(os.path.join(dir_path,f'GUMP_Params/Para_Unp_Off_forward{str}.csv'), header=None).to_numpy()[0]
    Paralst_Pol=pd.read_csv(os.path.join(dir_path,f'GUMP_Params/Para_Pol_Off_forward{str}.csv'), header=None).to_numpy()[0]

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
