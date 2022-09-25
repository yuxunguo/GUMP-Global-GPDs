from Parameters import ParaManager
from Observables import GPDobserv
from DVCS_xsec import dsigma_TOT, M
from multiprocessing import Pool
from functools import partial
from iminuit import Minuit
import numpy as np
import pandas as pd
import time

Minuit_Counter = 0

Time_Counter = 1

Q_threshold = 2

xB_Cut = 0.5

PDF_data = pd.read_csv('GUMPDATA/PDFdata.csv',       header = None, names = ['x', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'x': float, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})
tPDF_data = pd.read_csv('GUMPDATA/tPDFdata.csv',     header = None, names = ['x', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'x': float, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})
GFF_data = pd.read_csv('GUMPDATA/GFFdata.csv',       header = None, names = ['j', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'j': int, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})
DVCSxsec_data = pd.read_csv('GUMPDATA/DVCSxsec.csv', header = None, names = ['y', 'xB', 't', 'Q', 'phi', 'f', 'delta f', 'pol'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'phi': float, 'f': float, 'delta f': float, 'pol': str})

DVCSxsec_data_invalid = DVCSxsec_data[DVCSxsec_data['t']*(DVCSxsec_data['xB']-1) - M ** 2 * DVCSxsec_data['xB'] ** 2 < 0]
DVCSxsec_data = DVCSxsec_data[(DVCSxsec_data['Q'] > Q_threshold) & (DVCSxsec_data['xB'] < xB_Cut) & (DVCSxsec_data['t']*(DVCSxsec_data['xB']-1) - M ** 2 * DVCSxsec_data['xB'] ** 2 > 0)]
xBtQlst = DVCSxsec_data.drop_duplicates(subset = ['xB', 't', 'Q'], keep = 'first')[['xB','t','Q']].values.tolist()
DVCSxsec_group_data = list(map(lambda set: DVCSxsec_data[(DVCSxsec_data['xB'] == set[0]) & (DVCSxsec_data['t'] == set[1]) & ((DVCSxsec_data['Q'] == set[2]))], xBtQlst))

def PDF_theo(PDF_input: np.array, Para: np.array):
    [x, t, Q, f, delta_f, spe, flv] = PDF_input
    xi = 0
    Para_spe = Para[spe]
    if(spe == 0 or spe == 1):
        p = 1
    if(spe == 2 or spe == 3):
        p = -1

    PDF_theo = GPDobserv(x, xi, t, Q, p)
    return PDF_theo.tPDF(flv, Para_spe)     

def tPDF_theo(tPDF_input: np.array, Para: np.array):
    [x, t, Q, f, delta_f, spe, flv] = tPDF_input
    xi = 0
    Para_spe = Para[spe]
    if(spe == 0 or spe == 1):
        p = 1
    if(spe == 2 or spe == 3):
        p = -1

    tPDF_theo = GPDobserv(x, xi, t, Q, p)
    return tPDF_theo.tPDF(flv, Para_spe)        

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
    return GFF_theo.GFFj0(j, flv, Para_spe)

def CFF_theo(xB, t, Q, Para):
    x = 0
    xi = (1/(2 - xB) - (2*t*(-1 + xB))/(Q**2*(-2 + xB)**2))*xB
    H_E = GPDobserv(x, xi, t, Q, 1)
    Ht_Et = GPDobserv(x, xi, t, Q, -1)
    HCFF = H_E.CFF(Para[0])
    ECFF = H_E.CFF(Para[1])
    HtCFF = Ht_Et.CFF(Para[2])
    EtCFF = Ht_Et.CFF(Para[3])
    return [HCFF, ECFF, HtCFF, EtCFF]

def DVCSxsec_theo(DVCSxsec_input: np.array, CFF_input: np.array):
    [y, xB, t, Q, phi, f, delta_f, pol] = DVCSxsec_input    
    [HCFF, ECFF, HtCFF, EtCFF] = CFF_input
    return dsigma_TOT(y, xB, t, Q, phi, pol, HCFF, ECFF, HtCFF, EtCFF)

def DVCSxsec_cost_xBtQ(DVCSxsec_data_xBtQ: np.array, Para):
    [xB, t, Q] = [DVCSxsec_data_xBtQ['xB'].iat[0], DVCSxsec_data_xBtQ['t'].iat[0], DVCSxsec_data_xBtQ['Q'].iat[0]]
    [HCFF, ECFF, HtCFF, EtCFF] = CFF_theo(xB, t, Q, Para)
    DVCS_pred_xBtQ = np.array(list(map(partial(DVCSxsec_theo, CFF_input = [HCFF, ECFF, HtCFF, EtCFF]), np.array(DVCSxsec_data_xBtQ))))
    return np.sum(((DVCS_pred_xBtQ - DVCSxsec_data_xBtQ['f'])/ DVCSxsec_data_xBtQ['delta f']) ** 2 )

def cost_GUMP(Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
              Norm_Hubar,  alpha_Hubar,  beta_Hubar,   
              Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
              Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar,  
              Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_HS,
              R_H_u_xi2,   R_H_d_xi2,    R_H_g_xi2,
              R_E_u,        R_E_d,       R_E_g,       R_E_xi2,
              Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV, 
              Norm_Htubar, alpha_Htubar, beta_Htubar, 
              Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,
              Norm_Htdbar, alpha_Htdbar, beta_Htdbar, 
              Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_HtS,
              R_Ht_u_xi2,  R_Ht_d_xi2,   R_Ht_g_xi2,
              R_Et_u,       R_Et_d,      R_Et_g,      R_Et_xi2):

    global Minuit_Counter, Time_Counter

    time_now = time.time() -time_start
    
    if(time_now > Time_Counter * 600):
        print('Runing Time:',round(time_now/60),'minutes. Cost function called total', Minuit_Counter, 'times.')
        Time_Counter = Time_Counter + 1
    
    Minuit_Counter = Minuit_Counter + 1
    
    Paralst = [Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
               Norm_Hubar,  alpha_Hubar,  beta_Hubar,   
               Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
               Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar,  
               Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_HS,
               R_H_u_xi2,   R_H_d_xi2,    R_H_g_xi2,
               R_E_u,        R_E_d,       R_E_g,       R_E_xi2,
               Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV, 
               Norm_Htubar, alpha_Htubar, beta_Htubar, 
               Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,
               Norm_Htdbar, alpha_Htdbar, beta_Htdbar, 
               Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_HtS,
               R_Ht_u_xi2,  R_Ht_d_xi2,   R_Ht_g_xi2,
               R_Et_u,       R_Et_d,      R_Et_g,      R_Et_xi2]

    Para_all = ParaManager(np.array(Paralst))

    PDF_pred = np.array(list(pool.map(partial(PDF_theo, Para = Para_all), np.array(PDF_data))))
    cost_PDF = np.sum(((PDF_pred - PDF_data['f'])/ PDF_data['delta f']) ** 2 )

    tPDF_pred = np.array(list(pool.map(partial(tPDF_theo, Para = Para_all), np.array(tPDF_data))))
    cost_tPDF = np.sum(((tPDF_pred - tPDF_data['f'])/ tPDF_data['delta f']) ** 2 )

    GFF_pred = np.array(list(pool.map(partial(GFF_theo, Para = Para_all), np.array(GFF_data))))
    cost_GFF = np.sum(((GFF_pred - GFF_data['f'])/ GFF_data['delta f']) ** 2 )
    
    cost_DVCS_xBtQ = np.array(list(pool.map(partial(DVCSxsec_cost_xBtQ, Para = Para_all), DVCSxsec_group_data)))
    cost_DVCSxsec = np.sum(cost_DVCS_xBtQ)

    return  cost_PDF + cost_tPDF + cost_GFF + cost_DVCSxsec

def set_GUMP():
    fit = Minuit(cost_GUMP, Norm_HuV = 4.1,    alpha_HuV = 0.3,     beta_HuV = 3.0,    alphap_HuV = 1.1, 
                            Norm_Hubar = 0.2,  alpha_Hubar = 1.1,   beta_Hubar = 7.6,
                            Norm_HdV = 1.4,    alpha_HdV = 0.5,     beta_HdV = 3.7,    alphap_HdV = 1.3,
                            Norm_Hdbar = 0.2,  alpha_Hdbar = 1.1,   beta_Hdbar = 5.5,
                            Norm_Hg = 2.4,     alpha_Hg = 0.1,      beta_Hg = 6.8,     alphap_HS = 0.5,
                            R_H_u_xi2 = 1.0,   R_H_d_xi2 = 1.0,     R_H_g_xi2 = 1.0,
                            R_E_u = 1.0,       R_E_d = 1.0,         R_E_g = 1.0,       R_E_xi2 = 1.0,
                            Norm_HtuV = 11,    alpha_HtuV = -0.5,   beta_HtuV = 3.7,   alphap_HtuV = 1.0, 
                            Norm_Htubar = -30, alpha_Htubar = -1.8, beta_Htubar = 7.8,
                            Norm_HtdV = -0.9,  alpha_HtdV = 0.4,    beta_HtdV = 11,    alphap_HtdV = 1.0,
                            Norm_Htdbar = -30, alpha_Htdbar = -1.8, beta_Htdbar = 7.8,
                            Norm_Htg = 0.4,    alpha_Htg = -0.4,    beta_Htg = 1.5,    alphap_HtS = 1,
                            R_Ht_u_xi2 = 1.0,  R_Ht_d_xi2 = 1.0,    R_Ht_g_xi2 = 1.0,
                            R_Et_u = 1.0,      R_Et_d = 1.0,        R_Et_g = 1.0,      R_Et_xi2 = 1.0)
    fit.errordef = 1

    fit.limits['alpha_HuV'] = (-2, 1.2)
    fit.limits['alpha_Hubar'] = (-2, 1.2)
    fit.limits['alpha_HdV'] = (-2, 1.2)
    fit.limits['alpha_Hdbar'] = (-2, 1.2)
    fit.limits['alpha_Hg'] = (-2, 1.2)

    fit.limits['beta_HuV'] = (0, 15)
    fit.limits['beta_Hubar'] = (0, 15)
    fit.limits['beta_HdV'] = (0, 15)
    fit.limits['beta_Hdbar'] = (0, 15)
    fit.limits['beta_Hg'] = (0, 15)

    fit.limits['alpha_HtuV'] = (-2, 1.2)
    fit.limits['alpha_Htubar'] = (-2, 1.2)
    fit.limits['alpha_HtdV'] = (-2, 1.2)
    fit.limits['alpha_Htdbar'] = (-2, 1.2)
    fit.limits['alpha_Htg'] = (-2, 1.2)

    fit.limits['beta_HtuV'] = (0, 15)
    fit.limits['beta_Htubar'] = (0, 15)
    fit.limits['beta_HtdV'] = (0, 15)
    fit.limits['beta_Htdbar'] = (0, 15)
    fit.limits['beta_Htg'] = (0, 15)

    return fit

if __name__ == '__main__':

    fit_GUMP = set_GUMP()
    pool = Pool()
    time_start = time.time()

    fit_GUMP.migrad(ncall = 15000)
    time_migrad = time.time() 
    print('The migard runs for: ', round((time_migrad - time_start)/60, 1), 'minutes.')

    fit_GUMP.hesse()
    time_hesse = time.time()
    print('The hesse runs for: ', round((time_hesse - time_migrad)/60, 1), 'minutes.')

    pool.close()
    pool.join()

    ndof = len(PDF_data.index) + len(tPDF_data.index) + len(GFF_data.index) + len(DVCSxsec_data.index) - fit_GUMP.npar

    time_end = time.time() -time_start    
    with open('Output.txt', 'w') as f:
        print('Total running time:',round(time_end/60, 1), 'minutes. Total call of cost function:',Minuit_Counter,'(or', fit_GUMP.nfcn, 'from Minuit).\n', file=f)
        print('The chi squared per d.o.f. is:', round(fit_GUMP.fval/ndof, 3),'.\n', file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(fit_GUMP.params, file = f)
        print('Below are the output covariance from iMinuit:', file = f)
        print(fit_GUMP.covariance, file = f)