from Parameters import ParaManager
from Observables import GPDobserv
from multiprocessing import Pool
from functools import partial
from iminuit import Minuit
import numpy as np
import pandas as pd

Minuit_Counter = 1

def tPDF_theo(tPDF_input: np.array, Para: np.array):

    [x, t, Q, f, delta_f, spe, flv] = tPDF_input

    xi = 0

    Para_spe = Para[spe]

    if(spe == 0 or spe == 1):
        p = 1
    if(spe == 2 or spe == 3):
        p = -1

    tPDF_theo = GPDobserv(x, xi, t, Q, p)
    tPDF_lst = tPDF_theo.tPDF(Para_spe)        
    if(flv == "S"):
        return tPDF_lst[0] + tPDF_lst[1]
    if(flv == "NS"):
        return tPDF_lst[0] - tPDF_lst[1]
    if(flv == "g"):
        return tPDF_lst[2]

def GFF_theo(GFF_input: np.array, Para):
        
    [j, t, Q, f, delta_f, spe, flv] = GFF_input
    x = 0
    xi = 0        
    Para_spe = Para[spe]
    if(spe == 0 or spe == 1):
        p = 1        
    if(spe == 2 or spe == 3):
        p = -1

    GFF_theo = GPDobserv(x, xi, t, Q, p)
    GFF_lst = GFF_theo.GFFj0(j, Para_spe)
        
    if(flv == "S"):
        return GFF_lst[0] + GFF_lst[1]

    if(flv == "NS"):
        return GFF_lst[0] - GFF_lst[1]

    if(flv == "g"):
        return GFF_lst[2]

def cost_GUMP(alphapV_H, alphapS_H, R_H_xi2, R_E_u, R_E_d, R_E_g, R_E_xi2, alphapV_Ht, alphapS_Ht, R_Ht_xi2, R_Et_u, R_Et_d, R_Et_g, R_Et_xi2):

    global Minuit_Counter
    print("Cost function called ", Minuit_Counter, " times")
    Minuit_Counter = Minuit_Counter + 1

    Paralst = [alphapV_H, alphapS_H, R_H_xi2, R_E_u, R_E_d, R_E_g, R_E_xi2, alphapV_Ht, alphapS_Ht, R_Ht_xi2, R_Et_u, R_Et_d, R_Et_g, R_Et_xi2]

    Para_all = ParaManager(Paralst)
    #pool = Pool(6)
    #[H_para, E_para, Ht_Para, Et_para] = Para_all

    tPDF_data = pd.read_csv("GUMPDATA/tPDFdata.csv", names = ["x", "t", "Q", "f", "delta f", "spe", "flv"])    
    tPDF_pred = np.array(list(map(partial(tPDF_theo, Para = Para_all), np.array(tPDF_data))))
    cost_tPDF = np.sum(((tPDF_pred - tPDF_data["f"])/ tPDF_data["delta f"]) ** 2 )

    GFF_data = pd.read_csv("GUMPDATA/GFFdata.csv", names = ["j", "t", "Q", "f", "delta f", "spe", "flv"])
    GFF_pred = np.array(list(map(partial(GFF_theo, Para = Para_all), np.array(GFF_data))))
    cost_GFF = np.sum(((GFF_pred - GFF_data["f"])/ GFF_data["delta f"]) ** 2 )

    #pool.close()
    #pool.join()
    return cost_tPDF + cost_GFF

if __name__ == '__main__':
    NDOF = 313 + 12  - 8
    fit_gump = Minuit(cost_GUMP, alphapV_H = 1, alphapS_H = 1, R_H_xi2 = 1, R_E_u = 1, R_E_d = 1, R_E_g = 1, R_E_xi2 = 1, alphapV_Ht = 1, alphapS_Ht = 1, R_Ht_xi2 = 1, R_Et_u = 1, R_Et_d = 1, R_Et_g = 1, R_Et_xi2 = 1)
    fit_gump.errordef = 1
    fit_gump.fixed["R_H_xi2"] = True
    fit_gump.fixed["R_E_g"] = True
    fit_gump.fixed["R_E_xi2"] = True
    fit_gump.fixed["R_Ht_xi2"] = True
    fit_gump.fixed["R_Et_xi2"] = True
    fit_gump.fixed["R_Et_g"] = True
    fit_gump.migrad()
    fit_gump.hesse()
    print(fit_gump.fval/NDOF)
    print(fit_gump.params)


