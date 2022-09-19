from Parameters import ParaManager
from Observables import GPDobserv
from multiprocessing import Pool
from functools import partial
from iminuit import Minuit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Minuit_Counter = 1
PDF_data = pd.read_csv("GUMPDATA/PDFdata.csv", names = ["x", "t", "Q", "f", "delta f", "spe", "flv"])
tPDF_data = pd.read_csv("GUMPDATA/tPDFdata.csv", names = ["x", "t", "Q", "f", "delta f", "spe", "flv"])
GFF_data = pd.read_csv("GUMPDATA/GFFdata.csv", names = ["j", "t", "Q", "f", "delta f", "spe", "flv"])

def PDF_theo(PDF_input: np.array, Para: np.array):
    [x, t, Q, f, delta_f, spe, flv] = PDF_input
    xi = 0
    Para_spe = Para[spe]
    if(spe == 0 or spe == 1):
        p = 1
    if(spe == 2 or spe == 3):
        p = -1

    PDF_theo = GPDobserv(x, xi, t, Q, p)
    PDF_lst = PDF_theo.tPDF(Para_spe)        
    if(flv == "u"):
        return PDF_lst[0]
    if(flv == "d"):
        return PDF_lst[1]
    if(flv == "g"):
        return PDF_lst[2]

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

def cost_GUMP(Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
              Norm_Hubar,  alpha_Hubar,  beta_Hubar,   
              Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
              Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar,  
              Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_HS,
              R_H_xi2,     R_E_u,        R_E_d,       R_E_g,       R_E_xi2,
              Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV, 
              Norm_Htubar, alpha_Htubar, beta_Htubar, 
              Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,
              Norm_Htdbar, alpha_Htdbar, beta_Htdbar, 
              Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_HtS,
              R_Ht_xi2,    R_Et_u,       R_Et_d,      R_Et_g,      R_Et_xi2):

    global Minuit_Counter
    print("Cost function called ", Minuit_Counter, " times")
    Minuit_Counter = Minuit_Counter + 1

    Paralst = [Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
               Norm_Hubar,  alpha_Hubar,  beta_Hubar,   
               Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
               Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar,  
               Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_HS,
               R_H_xi2,     R_E_u,        R_E_d,       R_E_g,       R_E_xi2,
               Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV, 
               Norm_Htubar, alpha_Htubar, beta_Htubar, 
               Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,
               Norm_Htdbar, alpha_Htdbar, beta_Htdbar, 
               Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_HtS,
               R_Ht_xi2,    R_Et_u,       R_Et_d,      R_Et_g,      R_Et_xi2]

    Para_all = ParaManager(np.array(Paralst))
    #[H_para, E_para, Ht_Para, Et_para] = Para_all
    
    PDF_pred = np.array(list(pool.map(partial(PDF_theo, Para = Para_all), np.array(PDF_data))))
    cost_PDF = np.sum(((PDF_pred - PDF_data["f"])/ PDF_data["delta f"]) ** 2 )

    tPDF_pred = np.array(list(pool.map(partial(tPDF_theo, Para = Para_all), np.array(tPDF_data))))
    cost_tPDF = np.sum(((tPDF_pred - tPDF_data["f"])/ tPDF_data["delta f"]) ** 2 )

    GFF_pred = np.array(list(pool.map(partial(GFF_theo, Para = Para_all), np.array(GFF_data))))
    cost_GFF = np.sum(((GFF_pred - GFF_data["f"])/ GFF_data["delta f"]) ** 2 )

    return  cost_GFF + cost_PDF + cost_tPDF

if __name__ == '__main__':

    pool = Pool()
    NDOF = 310 + 313 + 12   - 4 
    fit_gump = Minuit(cost_GUMP, Norm_HuV = 1,    alpha_HuV = 1,    beta_HuV = 1,    alphap_HuV = 1, 
                                 Norm_Hubar = 1,  alpha_Hubar = 1,  beta_Hubar = 1,
                                 Norm_HdV = 1,    alpha_HdV = 1,    beta_HdV = 1,    alphap_HdV = 1,
                                 Norm_Hdbar = 1,  alpha_Hdbar = 1,  beta_Hdbar = 1,
                                 Norm_Hg = 1,     alpha_Hg = 1,     beta_Hg = 1,     alphap_HS = 1,
                                 R_H_xi2 = 1,     R_E_u = 1,        R_E_d = 1,       R_E_g = 1,       R_E_xi2 = 1,
                                 Norm_HtuV = 1,   alpha_HtuV = 1,   beta_HtuV = 1,   alphap_HtuV = 1, 
                                 Norm_Htubar = 1, alpha_Htubar = 1, beta_Htubar = 1,
                                 Norm_HtdV = 1,   alpha_HtdV = 1,   beta_HtdV = 1,   alphap_HtdV = 1,
                                 Norm_Htdbar = 1, alpha_Htdbar = 1, beta_Htdbar = 1,
                                 Norm_Htg = 1,    alpha_Htg = 1,    beta_Htg = 1,    alphap_HtS = 1,
                                 R_Ht_xi2 = 1,    R_Et_u = 1,       R_Et_d = 1,      R_Et_g = 1,      R_Et_xi2 = 1)
    fit_gump.errordef = 1
    fit_gump.fixed["R_H_xi2"] = True
    fit_gump.fixed["alphap_HS"] = True
    fit_gump.fixed["R_E_g"] = True
    fit_gump.fixed["R_E_xi2"] = True
    fit_gump.fixed["alphap_HtuV"] = True
    fit_gump.fixed["alphap_HtdV"] = True
    fit_gump.fixed["alphap_HtS"] = True
    fit_gump.fixed["R_Ht_xi2"] = True    
    fit_gump.fixed["R_Et_u"] = True
    fit_gump.fixed["R_Et_d"] = True
    fit_gump.fixed["R_Et_g"] = True
    fit_gump.fixed["R_Et_xi2"] = True
    fit_gump.migrad()
    fit_gump.hesse()
    print(fit_gump.fval/NDOF)
    print(fit_gump.params)

    """
    Para_all = ParaManager([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    print(Para_all[0].shape)
    
    
    filt = ((GFF_data['j'] == 0) & (GFF_data['spe'] == 0) &  (GFF_data['flv'] == "NS") )
    filt2 = ((GFF_data['j'] == 0) & (GFF_data['spe'] == 0) &  (GFF_data['flv'] == "S") )
    GFFj0NS0 = GFF_data[filt]
    GFFj0S0 = GFF_data[filt2]
    GFF_pred = np.array(list(pool.map(partial(GFF_theo, Para = Para_all), np.array(GFFj0NS0))))
    GFF_pred2 = np.array(list(pool.map(partial(GFF_theo, Para = Para_all), np.array(GFFj0S0))))
    print(GFFj0NS0)
    print(GFF_pred)

    print(GFFj0S0)
    print(GFF_pred2)
    fig, ax = plt.subplots()
    ax.errorbar(-GFFj0NS0['t'], GFFj0NS0['f'], yerr = GFFj0NS0['delta f'], fmt='.', c='k')
    ax.plot(-GFFj0NS0['t'],GFF_pred, label='fit')
    plt.show()
    
    fig, ax = plt.subplots()
    ax.errorbar(-GFFj0S0['t'], GFFj0S0['f'], yerr = GFFj0S0['delta f'], fmt='.', c='k')
    ax.plot(-GFFj0S0['t'],GFF_pred2, label='fit')
    plt.show()
    """
    pool.close()
    pool.join()
