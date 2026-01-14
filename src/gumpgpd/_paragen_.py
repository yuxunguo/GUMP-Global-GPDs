import time, os, csv
import numpy as np
import pandas as pd
from . import config 
config.INC_gGFF = False
config.INC_JPSI = False
from .Minimizer import off_forward_fit_withH_withHt
from ._helper_ import gump_msg

dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    
    gump_msg("Fit is starting. Parameters will be generated in Minimizer.Paralst_Unp_off_forward and Minimizer.Paralst_Pol_off_forward.", "INFO")

    gump_msg("Strongly recommended: run the _cachegen_ module first to generate cached data if not already done.", "NOTE")

    gump_msg("Performance may be lower and memory usage MUCH higher if no cache exists yet.", "NOTE")



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
        gump_msg(f"off-forward fit unpolarized parameters saved to Para_Unp_Off_forward{str}.csv", level="INFO")

    with open(os.path.join(dir_path,f"GUMP_Params/Para_Pol_Off_forward{str}.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerow(FitVals[UnpLength:])
        csvWriter.writerow(FitErrs[UnpLength:])
        gump_msg(f"off-forward fit polarized parameters saved to Para_Pol_Off_forward{str}.csv", level="INFO")
