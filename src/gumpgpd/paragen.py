import time, os, csv
import numpy as np
import pandas as pd
from .Minimizer import forward_H_fit, forward_Ht_fit, off_forward_fit,off_forward_fit_withHt, off_forward_fit_withH_withHt, cost_forward_H, cost_forward_Ht, cost_off_forward, cost_off_forward_withHt, cost_off_forward_withH_withHt,Paralst_Unp_Names, Paralst_Pol_Names,Paralst_Aux_Names
from . import config 
dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    
    print('Test15')
    time_start = time.time()
    '''
    Paralst_Unp=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Unp.csv'), header=None).to_numpy()[0]
    Paralst_Pol=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Pol.csv'), header=None).to_numpy()[0]
    
    fit_forward_H   = forward_H_fit(Paralst_Unp)
    Paralst_Unp     = np.array(fit_forward_H.values)
    
    with open(os.path.join(dir_path,"GUMP_Params/Para_Unp.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerow(list([*fit_forward_H.values]))
        csvWriter.writerow(list([*fit_forward_H.errors]))
        print("H fit parameters saved to Para_Unp.csv")
    '''
    '''
    fit_forward_Ht  = forward_Ht_fit(Paralst_Pol)
    Paralst_Pol     = np.array(fit_forward_Ht.values)

    with open(os.path.join(dir_path,"GUMP_Params/Para_Pol.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerow(list([*fit_forward_Ht.values]))
        csvWriter.writerow(list([*fit_forward_Ht.errors]))
        print("Ht fit parameters saved to Para_Pol.csv")
    '''
    str = '_withH_withHt_NLO'
    
    #'''
    Paralst_Unp=pd.read_csv(os.path.join(dir_path,f'GUMP_Params/Para_Unp_Off_forward_withH_withHt_NLO.csv'), header=None).to_numpy()[0]
    Paralst_Pol=pd.read_csv(os.path.join(dir_path,f'GUMP_Params/Para_Pol_Off_forward_withH_withHt_NLO.csv'), header=None).to_numpy()[0]
    
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
    config.Export_Mode = True
    
    #Paralst_Unp=pd.read_csv(os.path.join(dir_path,f'GUMP_Params/Para_Unp_Off_forward{str}.csv'), header=None).to_numpy()[0]
    #Paralst_Pol=pd.read_csv(os.path.join(dir_path,f'GUMP_Params/Para_Pol_Off_forward{str}.csv'), header=None).to_numpy()[0]
    #Paralst_Aux = [0.66] * len(Paralst_Aux_Names)

    Paralst_All=[4.974479052380454, 0.2150740792451562, 3.2411071645005896, 0.7496135271861655, 1.6947259889765753e-05, 0.11397791632915487, 1.1992637937435942, 6.101928951846839, 0.15, 0.56176370993929, 0.2602545960564436, 19.86049090618029, 3.200856770199667, 0.1923942322369725, 4.23490683380349, 0.3509828512979401, 0.6524807422831795, 0.11378159455167741, 1.1999688672271176, 19.396037904383444, 2.1039093980642294, 0.2504396318783725, 10.36130493034568, 0.7185298137875703, 1.1569834142182356, 19.783375624216465, 0.15, 0.0, 6.07209920602559, 0.7992782162183695, 9.549779272485518, 0.6452181146949506, 0.7423088506961779, 3.161920842921677, 0.5193801599320266, -0.7902356106259096, 1.794888871874573, -0.7664761506460352, -5.8805023474187115, -0.1453428363296144, 0.26071968781294463, 41.83195270678209, -1.042690251016659, 0.10311130414566934, 1.4671476396361256, 0.27389257701426256, -0.33683378796898206, -9.609156920833957, 0.1338870091646237, 3.079834783046851, 3.8070666591268107, 4.041975114635693, -0.23700061079989787, 2.6921598672233475, 0.2762739055109903, 0.17088370214337661, 0.24473620738654267, 6.016319029866385, 0.15, -0.6766553673895538, 0.13204251635388609, 2.72966155398141, 0.3390801186480352, -0.2383352108503581, 0.3053415087598208, 6.001105006259753, 0.31955455306925157, 0.4379594809814069, 1.6546572541123585, 0.15, 0.8144115783739172, 0.7535311769783539, 4.209046221113367, 0.6778360060308288, -0.6663422193456814, 134.13045636650617, 0.13995168646493417, 7.83809328363342, 0.0, 4.607388240015277, 21.255753307346755, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.980789233400085, 0.6650903319731375]

    Paralst_Unp = Paralst_All[:51]
    Paralst_Pol = Paralst_All[51:-1]
    Paralst_Aux = Paralst_All[-1:]

    params_unp = dict(zip(Paralst_Unp_Names, Paralst_Unp))
    params_pol = dict(zip(Paralst_Pol_Names, Paralst_Pol))
    params_aux = dict(zip(Paralst_Aux_Names, Paralst_Aux))
    
    params = {**params_unp, **params_pol, **params_aux}

    print(cost_off_forward_withH_withHt(**params))
    '''