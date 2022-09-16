from Parameters import ParaManager
from Observables import GPDobserv
import numpy as np
import pandas as pd
from iminuit import Minuit

def cost_GUMP(alphapV_H, alphapS_H, R_H_xi2, R_E, R_E_xi2, alphapV_Ht, alphapS_Ht, R_Ht_xi2, R_Et, R_Et_xi2):

    Paralst = [alphapV_H, alphapS_H, R_H_xi2, R_E, R_E_xi2, alphapV_Ht, alphapS_Ht, R_Ht_xi2, R_Et, R_Et_xi2]

    Para_all = ParaManager(Paralst)

    [H_para, E_para, Ht_Para, Et_para] = Para_all

    tPDF_data = pd.read_csv("GUMPDATA/tPDF_0.csv", names = ["x", "t", "f", "delta f", "spe", "flv"])

    def tPDF_theo():
        return 0
    
    cost_tPDF = np.sum(((0 - tPDF_data["f"])/ tPDF_data["delta f"]) ** 2 )




fit_gump = Minuit(cost_GUMP, alphapV_H = 1, alphapS_H = 1, R_H_xi2 = 1, R_E = 1, R_E_xi2 = 1, alphapV_Ht = 1, alphapS_Ht = 1, R_Ht_xi2 = 1, R_Et = 1, R_Et_xi2 = 1)
fit_gump.errordef = 1
fit_gump.migrad()
fit_gump.hesse()

print(fit_gump.params)