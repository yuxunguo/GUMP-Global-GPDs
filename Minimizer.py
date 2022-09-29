from Parameters import ParaManager_Unp, ParaManager_Pol
from Observables import GPDobserv
from DVCS_xsec import dsigma_TOT, M
from multiprocessing import Pool
from functools import partial
from iminuit import Minuit
import numpy as np
import pandas as pd
import time
from  Plotting import plot_compare

Minuit_Counter = 0

Time_Counter = 1

Q_threshold = 2

xB_Cut = 0.5

PDF_data = pd.read_csv('GUMPDATA/PDFdata.csv',       header = None, names = ['x', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'x': float, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})
PDF_data_H  = PDF_data[PDF_data['spe'] == 0]
PDF_data_E  = PDF_data[PDF_data['spe'] == 1]
PDF_data_Ht = PDF_data[PDF_data['spe'] == 2]
PDF_data_Et = PDF_data[PDF_data['spe'] == 3]

tPDF_data = pd.read_csv('GUMPDATA/tPDFdata.csv',     header = None, names = ['x', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'x': float, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})
tPDF_data_H  = tPDF_data[tPDF_data['spe'] == 0]
tPDF_data_E  = tPDF_data[tPDF_data['spe'] == 1]
tPDF_data_Ht = tPDF_data[tPDF_data['spe'] == 2]
tPDF_data_Et = tPDF_data[tPDF_data['spe'] == 3]

GFF_data = pd.read_csv('GUMPDATA/GFFdata.csv',       header = None, names = ['j', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'j': int, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})
GFF_data_H  = GFF_data[GFF_data['spe'] == 0]
GFF_data_E  = GFF_data[GFF_data['spe'] == 1]
GFF_data_Ht = GFF_data[GFF_data['spe'] == 2]
GFF_data_Et = GFF_data[GFF_data['spe'] == 3]

DVCSxsec_data = pd.read_csv('GUMPDATA/DVCSxsec.csv', header = None, names = ['y', 'xB', 't', 'Q', 'phi', 'f', 'delta f', 'pol'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'phi': float, 'f': float, 'delta f': float, 'pol': str})
DVCSxsec_data_invalid = DVCSxsec_data[DVCSxsec_data['t']*(DVCSxsec_data['xB']-1) - M ** 2 * DVCSxsec_data['xB'] ** 2 < 0]
DVCSxsec_data = DVCSxsec_data[(DVCSxsec_data['Q'] > Q_threshold) & (DVCSxsec_data['xB'] < xB_Cut) & (DVCSxsec_data['t']*(DVCSxsec_data['xB']-1) - M ** 2 * DVCSxsec_data['xB'] ** 2 > 0)]
xBtQlst = DVCSxsec_data.drop_duplicates(subset = ['xB', 't', 'Q'], keep = 'first')[['xB','t','Q']].values.tolist()
DVCSxsec_group_data = list(map(lambda set: DVCSxsec_data[(DVCSxsec_data['xB'] == set[0]) & (DVCSxsec_data['t'] == set[1]) & ((DVCSxsec_data['Q'] == set[2]))], xBtQlst))

def PDF_theo(PDF_input: np.array, Para: np.array):
    [x, t, Q, f, delta_f, spe, flv] = PDF_input
    xi = 0
    if(spe == 0 or spe == 1):
        spe, p = spe, 1

    if(spe == 2 or spe == 3):
        spe, p = spe - 2 , -1

    Para_spe = Para[spe]
    PDF_theo = GPDobserv(x, xi, t, Q, p)
    return PDF_theo.tPDF(flv, Para_spe)     

def tPDF_theo(tPDF_input: np.array, Para: np.array):
    [x, t, Q, f, delta_f, spe, flv] = tPDF_input
    xi = 0
    if(spe == 0 or spe == 1):
        spe, p = spe, 1

    if(spe == 2 or spe == 3):
        spe, p = spe - 2 , -1

    Para_spe = Para[spe]
    tPDF_theo = GPDobserv(x, xi, t, Q, p)
    return tPDF_theo.tPDF(flv, Para_spe)        

def GFF_theo(GFF_input: np.array, Para):
    [j, t, Q, f, delta_f, spe, flv] = GFF_input
    x = 0
    xi = 0   
    if(spe == 0 or spe == 1):
        spe, p = spe, 1

    if(spe == 2 or spe == 3):
        spe, p = spe - 2 , -1

    Para_spe = Para[spe]
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

def cost_forward_H(Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
                   Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
                   Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
                   Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar, 
                   Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg,
                   Norm_EuV,    alpha_EuV,    beta_EuV,    alphap_EuV,
                   Norm_EdV,    alpha_EdV,    beta_EdV,    alphap_EdV,
                   R_E_Sea,     R_H_xi2,      R_E_xi2):

    global Minuit_Counter, Time_Counter

    time_now = time.time() -time_start
    
    if(time_now > Time_Counter * 600):
        print('Runing Time:',round(time_now/60),'minutes. Cost function called total', Minuit_Counter, 'times.')
        Time_Counter = Time_Counter + 1
    
    Minuit_Counter = Minuit_Counter + 1

    Paralst = [Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
               Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
               Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
               Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar, 
               Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg,
               Norm_EuV,    alpha_EuV,    beta_EuV,    alphap_EuV,
               Norm_EdV,    alpha_EdV,    beta_EdV,    alphap_EdV,
               R_E_Sea,     R_H_xi2,      R_E_xi2]
    
    Para_all = ParaManager_Unp(Paralst)
    PDF_H_pred = np.array(list(pool.map(partial(PDF_theo, Para = Para_all), np.array(PDF_data_H))))
    cost_PDF_H = np.sum(((PDF_H_pred - PDF_data_H['f'])/ PDF_data_H['delta f']) ** 2 )

    tPDF_H_pred = np.array(list(pool.map(partial(tPDF_theo, Para = Para_all), np.array(tPDF_data_H))))
    cost_tPDF_H = np.sum(((tPDF_H_pred - tPDF_data_H['f'])/ tPDF_data_H['delta f']) ** 2 )

    GFF_H_pred = np.array(list(pool.map(partial(GFF_theo, Para = Para_all), np.array(GFF_data_H))))
    cost_GFF_H = np.sum(((GFF_H_pred - GFF_data_H['f'])/ GFF_data_H['delta f']) ** 2 )

    return cost_PDF_H + cost_tPDF_H + cost_GFF_H

def cost_forward_Ht(Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV, 
                    Norm_Htubar, alpha_Htubar, beta_Htubar, alphap_Htqbar,
                    Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,
                    Norm_Htdbar, alpha_Htdbar, beta_Htdbar, 
                    Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_Htg,
                    Norm_EtuV,   alpha_EtuV,   beta_EtuV,   alphap_EtuV, 
                    Norm_EtdV,   alpha_EtdV,   beta_EtdV,   alphap_EtdV,
                    R_Et_Sea,    R_Ht_xi2,     R_Et_xi2):

    global Minuit_Counter, Time_Counter

    time_now = time.time() -time_start
    
    if(time_now > Time_Counter * 600):
        print('Runing Time:',round(time_now/60),'minutes. Cost function called total', Minuit_Counter, 'times.')
        Time_Counter = Time_Counter + 1
    
    Minuit_Counter = Minuit_Counter + 1

    Paralst = [Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV, 
               Norm_Htubar, alpha_Htubar, beta_Htubar, alphap_Htqbar,
               Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,
               Norm_Htdbar, alpha_Htdbar, beta_Htdbar, 
               Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_Htg,
               Norm_EtuV,   alpha_EtuV,   beta_EtuV,   alphap_EtuV, 
               Norm_EtdV,   alpha_EtdV,   beta_EtdV,   alphap_EtdV,
               R_Et_Sea,    R_Ht_xi2,     R_Et_xi2]
    
    Para_all = ParaManager_Pol(Paralst)
    PDF_Ht_pred = np.array(list(pool.map(partial(PDF_theo, Para = Para_all), np.array(PDF_data_Ht))))
    cost_PDF_Ht = np.sum(((PDF_Ht_pred - PDF_data_Ht['f'])/ PDF_data_Ht['delta f']) ** 2 )

    tPDF_Ht_pred = np.array(list(pool.map(partial(tPDF_theo, Para = Para_all), np.array(tPDF_data_Ht))))
    cost_tPDF_Ht = np.sum(((tPDF_Ht_pred - tPDF_data_Ht['f'])/ tPDF_data_Ht['delta f']) ** 2 )

    GFF_Ht_pred = np.array(list(pool.map(partial(GFF_theo, Para = Para_all), np.array(GFF_data_Ht))))
    cost_GFF_Ht = np.sum(((GFF_Ht_pred - GFF_data_Ht['f'])/ GFF_data_Ht['delta f']) ** 2 )

    return cost_PDF_Ht + cost_tPDF_Ht + cost_GFF_Ht

def cost_forward_E(Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
                   Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
                   Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
                   Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar, 
                   Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg,
                   Norm_EuV,    alpha_EuV,    beta_EuV,    alphap_EuV,
                   Norm_EdV,    alpha_EdV,    beta_EdV,    alphap_EdV,
                   R_E_Sea,     R_H_xi2,      R_E_xi2):

    global Minuit_Counter, Time_Counter

    time_now = time.time() -time_start
    
    if(time_now > Time_Counter * 600):
        print('Runing Time:',round(time_now/60),'minutes. Cost function called total', Minuit_Counter, 'times.')
        Time_Counter = Time_Counter + 1
    
    Minuit_Counter = Minuit_Counter + 1

    Paralst = [Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
               Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
               Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
               Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar, 
               Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg,
               Norm_EuV,    alpha_EuV,    beta_EuV,    alphap_EuV,
               Norm_EdV,    alpha_EdV,    beta_EdV,    alphap_EdV,
               R_E_Sea,     R_H_xi2,      R_E_xi2]
    
    Para_all = ParaManager_Unp(Paralst)
    PDF_E_pred = np.array(list(pool.map(partial(PDF_theo, Para = Para_all), np.array(PDF_data_E))))
    cost_PDF_E = np.sum(((PDF_E_pred - PDF_data_E['f'])/ PDF_data_E['delta f']) ** 2 )

    tPDF_E_pred = np.array(list(pool.map(partial(tPDF_theo, Para = Para_all), np.array(tPDF_data_E))))
    cost_tPDF_E = np.sum(((tPDF_E_pred - tPDF_data_E['f'])/ tPDF_data_E['delta f']) ** 2 )

    GFF_E_pred = np.array(list(pool.map(partial(GFF_theo, Para = Para_all), np.array(GFF_data_E))))
    cost_GFF_E = np.sum(((GFF_E_pred - GFF_data_E['f'])/ GFF_data_E['delta f']) ** 2 )

    return cost_PDF_E + cost_tPDF_E + cost_GFF_E

def cost_forward_Et(Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV, 
                    Norm_Htubar, alpha_Htubar, beta_Htubar, alphap_Htqbar,
                    Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,
                    Norm_Htdbar, alpha_Htdbar, beta_Htdbar, 
                    Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_Htg,
                    Norm_EtuV,   alpha_EtuV,   beta_EtuV,   alphap_EtuV, 
                    Norm_EtdV,   alpha_EtdV,   beta_EtdV,   alphap_EtdV,
                    R_Et_Sea,    R_Ht_xi2,     R_Et_xi2):

    global Minuit_Counter, Time_Counter

    time_now = time.time() -time_start
    
    if(time_now > Time_Counter * 600):
        print('Runing Time:',round(time_now/60),'minutes. Cost function called total', Minuit_Counter, 'times.')
        Time_Counter = Time_Counter + 1
    
    Minuit_Counter = Minuit_Counter + 1

    Paralst = [Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV, 
               Norm_Htubar, alpha_Htubar, beta_Htubar, alphap_Htqbar,
               Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,
               Norm_Htdbar, alpha_Htdbar, beta_Htdbar, 
               Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_Htg,
               Norm_EtuV,   alpha_EtuV,   beta_EtuV,   alphap_EtuV, 
               Norm_EtdV,   alpha_EtdV,   beta_EtdV,   alphap_EtdV,
               R_Et_Sea,    R_Ht_xi2,     R_Et_xi2]
    
    Para_all = ParaManager_Pol(Paralst)
    PDF_Et_pred = np.array(list(pool.map(partial(PDF_theo, Para = Para_all), np.array(PDF_data_Et))))
    cost_PDF_Et = np.sum(((PDF_Et_pred - PDF_data_Et['f'])/ PDF_data_Et['delta f']) ** 2 )

    tPDF_Et_pred = np.array(list(pool.map(partial(tPDF_theo, Para = Para_all), np.array(tPDF_data_Et))))
    cost_tPDF_Et = np.sum(((tPDF_Et_pred - tPDF_data_Et['f'])/ tPDF_data_Et['delta f']) ** 2 )

    GFF_Et_pred = np.array(list(pool.map(partial(GFF_theo, Para = Para_all), np.array(GFF_data_Et))))
    cost_GFF_Et = np.sum(((GFF_Et_pred - GFF_data_Et['f'])/ GFF_data_Et['delta f']) ** 2 )

    return cost_PDF_Et + cost_tPDF_Et + cost_GFF_Et

def forward_H_fit():
    fit_forw_H = Minuit(cost_forward_H, Norm_HuV = 4.1,    alpha_HuV = 0.3,     beta_HuV = 3.0,    alphap_HuV = 1.1, 
                                        Norm_Hubar = 0.2,  alpha_Hubar = 1.1,   beta_Hubar = 7.6,  alphap_Hqbar = 0.15,
                                        Norm_HdV = 1.4,    alpha_HdV = 0.5,     beta_HdV = 3.7,    alphap_HdV = 1.3,
                                        Norm_Hdbar = 0.2,  alpha_Hdbar = 1.1,   beta_Hdbar = 5.5, 
                                        Norm_Hg = 2.4,     alpha_Hg = 0.1,      beta_Hg = 6.8,     alphap_Hg = 1.1,
                                        Norm_EuV = 4.1,    alpha_EuV = 0.3,     beta_EuV = 3.0,    alphap_EuV = 1.1, 
                                        Norm_EdV = 1.4,    alpha_EdV = 0.5,     beta_EdV = 3.7,    alphap_EdV = 1.3,
                                        R_E_Sea = 1,       R_H_xi2 = 1,         R_E_xi2 = 1)
    fit_forw_H.errordef = 1

    fit_forw_H.limits['alpha_HuV'] = (-2, 1.2)
    fit_forw_H.limits['alpha_Hubar'] = (-2, 1.2)
    fit_forw_H.limits['alpha_HdV'] = (-2, 1.2)
    fit_forw_H.limits['alpha_Hdbar'] = (-2, 1.2)
    fit_forw_H.limits['alpha_Hg'] = (-2, 1.2)
    fit_forw_H.limits['alpha_EuV'] = (-2, 1.2)
    fit_forw_H.limits['alpha_EdV'] = (-2, 1.2)

    fit_forw_H.limits['beta_HuV'] = (0, 15)
    fit_forw_H.limits['beta_Hubar'] = (0, 15)
    fit_forw_H.limits['beta_HdV'] = (0, 15)
    fit_forw_H.limits['beta_Hdbar'] = (0, 15)
    fit_forw_H.limits['beta_Hg'] = (0, 15)    
    fit_forw_H.limits['beta_EuV'] = (0, 15)
    fit_forw_H.limits['beta_EdV'] = (0, 15)

    #fit_forw_Unp.fixed['alphap_Hqbar'] = True

    fit_forw_H.fixed['Norm_EuV'] = True
    fit_forw_H.fixed['alpha_EuV'] = True
    fit_forw_H.fixed['beta_EuV'] = True
    fit_forw_H.fixed['alphap_EuV'] = True

    fit_forw_H.fixed['Norm_EdV'] = True
    fit_forw_H.fixed['alpha_EdV'] = True
    fit_forw_H.fixed['beta_EdV'] = True
    fit_forw_H.fixed['alphap_EdV'] = True

    fit_forw_H.fixed['R_E_Sea'] = True
    fit_forw_H.fixed['R_H_xi2'] = True    
    fit_forw_H.fixed['R_E_xi2'] = True

    fit_forw_H.migrad()
    fit_forw_H.hesse()

    ndof_H = len(PDF_data_H.index) + len(tPDF_data_H.index) + len(GFF_data_H.index)  - fit_forw_H.npar 

    time_end = time.time() -time_start

    with open('GUMP_Output/H_forward_fit.txt', 'w') as f:
        print('Total running time:',round(time_end/60, 1), 'minutes. Total call of cost function:', fit_forw_H.nfcn, '.\n', file=f)
        print('The chi squared per d.o.f. is:', round(fit_forw_H.fval/ndof_H, 3),'.\n', file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*fit_forw_H.values, sep=", ", file = f)
        print(*fit_forw_H.errors, sep=", ", file = f)
        print(fit_forw_H.params, file = f)

    return fit_forw_H

def forward_Ht_fit():
    fit_forw_Ht = Minuit(cost_forward_Ht, Norm_HtuV = 11,    alpha_HtuV = -0.5,   beta_HtuV = 3.7,   alphap_HtuV = 1.0, 
                                          Norm_Htubar = -30, alpha_Htubar = -1.8, beta_Htubar = 7.8, alphap_Htqbar = 0.15,
                                          Norm_HtdV = -0.9,  alpha_HtdV = 0.4,    beta_HtdV = 11,    alphap_HtdV = 1.0,
                                          Norm_Htdbar = -30, alpha_Htdbar = -1.8, beta_Htdbar = 7.8,
                                          Norm_Htg = 0.4,    alpha_Htg = -0.4,    beta_Htg = 1.5,    alphap_Htg = 1.1,
                                          Norm_EtuV = 11,    alpha_EtuV = -0.5,   beta_EtuV = 3.7,   alphap_EtuV = 1.0,
                                          Norm_EtdV = -0.9,  alpha_EtdV = 0.4,    beta_EtdV = 11,    alphap_EtdV = 1.0,
                                          R_Et_Sea = 1,      R_Ht_xi2 = 1,        R_Et_xi2 = 1)
    fit_forw_Ht.errordef = 1

    fit_forw_Ht.limits['alpha_HtuV'] = (-2, 1.2)
    fit_forw_Ht.limits['alpha_Htubar'] = (-2, 1.2)
    fit_forw_Ht.limits['alpha_HtdV'] = (-2, 1.2)
    fit_forw_Ht.limits['alpha_Htdbar'] = (-2, 1.2)
    fit_forw_Ht.limits['alpha_Htg'] = (-2, 1.2)
    fit_forw_Ht.limits['alpha_EtuV'] = (-2, 1.2)
    fit_forw_Ht.limits['alpha_EtdV'] = (-2, 1.2)

    fit_forw_Ht.limits['beta_HtuV'] = (0, 15)
    fit_forw_Ht.limits['beta_Htubar'] = (0, 15)
    fit_forw_Ht.limits['beta_HtdV'] = (0, 15)
    fit_forw_Ht.limits['beta_Htdbar'] = (0, 15)
    fit_forw_Ht.limits['beta_Htg'] = (0, 15)
    fit_forw_Ht.limits['beta_EtuV'] = (0, 15)
    fit_forw_Ht.limits['beta_EtdV'] = (0, 15)

    #fit_forw_Pol.fixed['alphap_Htqbar'] = True

    fit_forw_Ht.fixed['Norm_EtuV'] = True
    fit_forw_Ht.fixed['alpha_EtuV'] = True
    fit_forw_Ht.fixed['beta_EtuV'] = True
    fit_forw_Ht.fixed['alphap_EtuV'] = True

    fit_forw_Ht.fixed['Norm_EtdV'] = True
    fit_forw_Ht.fixed['alpha_EtdV'] = True
    fit_forw_Ht.fixed['beta_EtdV'] = True
    fit_forw_Ht.fixed['alphap_EtdV'] = True

    fit_forw_Ht.fixed['R_Et_Sea'] = True
    fit_forw_Ht.fixed['R_Ht_xi2'] = True
    fit_forw_Ht.fixed['R_Et_xi2'] = True

    fit_forw_Ht.migrad()
    fit_forw_Ht.hesse()

    ndof_Ht = len(PDF_data_Ht.index) + len(tPDF_data_Ht.index) + len(GFF_data_Ht.index)  - fit_forw_Ht.npar

    time_end = time.time() -time_start    
    with open('GUMP_Output/Ht_forward_fit.txt', 'w') as f:
        print('Total running time:',round(time_end/60, 1), 'minutes. Total call of cost function:', fit_forw_Ht.nfcn, '.\n', file=f)
        print('The chi squared per d.o.f. is:', round(fit_forw_Ht.fval/ndof_Ht, 3),'.\n', file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*fit_forw_Ht.values, sep=", ", file = f)
        print(*fit_forw_Ht.errors, sep=", ", file = f)
        print(fit_forw_Ht.params, file = f)

    return fit_forw_Ht

def forward_E_fit(Paralst_Unp):

    [Norm_HuV_Init,    alpha_HuV_Init,    beta_HuV_Init,    alphap_HuV_Init, 
     Norm_Hubar_Init,  alpha_Hubar_Init,  beta_Hubar_Init,  alphap_Hqbar_Init,
     Norm_HdV_Init,    alpha_HdV_Init,    beta_HdV_Init,    alphap_HdV_Init,
     Norm_Hdbar_Init,  alpha_Hdbar_Init,  beta_Hdbar_Init, 
     Norm_Hg_Init,     alpha_Hg_Init,     beta_Hg_Init,     alphap_Hg_Init,
     Norm_EuV_Init,    alpha_EuV_Init,    beta_EuV_Init,    alphap_EuV_Init,
     Norm_EdV_Init,    alpha_EdV_Init,    beta_EdV_Init,    alphap_EdV_Init,
     R_E_Sea_Init,     R_H_xi2_Init,      R_E_xi2_Init] = Paralst_Unp

    fit_forw_E = Minuit(cost_forward_E, Norm_HuV = Norm_HuV_Init,     alpha_HuV = alpha_HuV_Init,      beta_HuV = beta_HuV_Init,     alphap_HuV = alphap_HuV_Init, 
                                        Norm_Hubar = Norm_Hubar_Init, alpha_Hubar = alpha_Hubar_Init,  beta_Hubar = beta_Hubar_Init, alphap_Hqbar = alphap_Hqbar_Init,
                                        Norm_HdV = Norm_HdV_Init,     alpha_HdV = alpha_HdV_Init,      beta_HdV = beta_HdV_Init,     alphap_HdV = alphap_HdV_Init,
                                        Norm_Hdbar = Norm_Hdbar_Init, alpha_Hdbar = alpha_Hdbar_Init,  beta_Hdbar = beta_Hdbar_Init, 
                                        Norm_Hg = Norm_Hg_Init,       alpha_Hg = alpha_Hg_Init,        beta_Hg = beta_Hg_Init,       alphap_Hg = alphap_Hg_Init,
                                        Norm_EuV = Norm_EuV_Init,     alpha_EuV = alpha_EuV_Init,      beta_EuV = beta_EuV_Init,     alphap_EuV = alphap_EuV_Init, 
                                        Norm_EdV = Norm_EdV_Init,     alpha_EdV = alpha_EdV_Init,      beta_EdV = beta_EdV_Init,     alphap_EdV = alphap_EdV_Init,
                                        R_E_Sea = R_E_Sea_Init,       R_H_xi2 = R_H_xi2_Init,          R_E_xi2 = R_E_xi2_Init)
    fit_forw_E.errordef = 1

    fit_forw_E.limits['alpha_HuV'] = (-2, 1.2)
    fit_forw_E.limits['alpha_Hubar'] = (-2, 1.2)
    fit_forw_E.limits['alpha_HdV'] = (-2, 1.2)
    fit_forw_E.limits['alpha_Hdbar'] = (-2, 1.2)
    fit_forw_E.limits['alpha_Hg'] = (-2, 1.2)
    fit_forw_E.limits['alpha_EuV'] = (-2, 1.2)
    fit_forw_E.limits['alpha_EdV'] = (-2, 1.2)

    fit_forw_E.limits['beta_HuV'] = (0, 15)
    fit_forw_E.limits['beta_Hubar'] = (0, 15)
    fit_forw_E.limits['beta_HdV'] = (0, 15)
    fit_forw_E.limits['beta_Hdbar'] = (0, 15)
    fit_forw_E.limits['beta_Hg'] = (0, 15)    
    fit_forw_E.limits['beta_EuV'] = (0, 15)
    fit_forw_E.limits['beta_EdV'] = (0, 15)

    fit_forw_E.fixed['Norm_HuV'] = True
    fit_forw_E.fixed['alpha_HuV'] = True
    fit_forw_E.fixed['beta_HuV'] = True
    fit_forw_E.fixed['alphap_HuV'] = True

    fit_forw_E.fixed['Norm_Hubar'] = True
    fit_forw_E.fixed['alpha_Hubar'] = True
    fit_forw_E.fixed['beta_Hubar'] = True
    fit_forw_E.fixed['alphap_Hqbar'] = True

    fit_forw_E.fixed['Norm_HdV'] = True
    fit_forw_E.fixed['alpha_HdV'] = True
    fit_forw_E.fixed['beta_HdV'] = True
    fit_forw_E.fixed['alphap_HdV'] = True

    fit_forw_E.fixed['Norm_Hdbar'] = True
    fit_forw_E.fixed['alpha_Hdbar'] = True
    fit_forw_E.fixed['beta_Hdbar'] = True

    fit_forw_E.fixed['Norm_Hg'] = True
    fit_forw_E.fixed['alpha_Hg'] = True
    fit_forw_E.fixed['beta_Hg'] = True
    fit_forw_E.fixed['alphap_Hg'] = True

    fit_forw_E.fixed['R_H_xi2'] = True    
    fit_forw_E.fixed['R_E_xi2'] = True

    fit_forw_E.migrad()
    fit_forw_E.hesse()

    ndof_E = len(PDF_data_E.index) + len(tPDF_data_E.index) + len(GFF_data_E.index)  - fit_forw_E.npar 

    time_end = time.time() -time_start

    with open('GUMP_Output/E_forward_fit.txt', 'w') as f:
        print('Total running time:',round(time_end/60, 1), 'minutes. Total call of cost function:', fit_forw_E.nfcn, '.\n', file=f)
        print('The chi squared per d.o.f. is:', round(fit_forw_E.fval/ndof_E, 3),'.\n', file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*fit_forw_E.values, sep=", ", file = f)
        print(*fit_forw_E.errors, sep=", ", file = f)
        print(fit_forw_E.params, file = f)

    return fit_forw_E

def forward_Et_fit(Paralst_Pol):
    
    [Norm_HtuV_Init,   alpha_HtuV_Init,   beta_HtuV_Init,   alphap_HtuV_Init, 
     Norm_Htubar_Init, alpha_Htubar_Init, beta_Htubar_Init, alphap_Htqbar_Init,
     Norm_HtdV_Init,   alpha_HtdV_Init,   beta_HtdV_Init,   alphap_HtdV_Init,
     Norm_Htdbar_Init, alpha_Htdbar_Init, beta_Htdbar_Init, 
     Norm_Htg_Init,    alpha_Htg_Init,    beta_Htg_Init,    alphap_Htg_Init,
     Norm_EtuV_Init,   alpha_EtuV_Init,   beta_EtuV_Init,   alphap_EtuV_Init, 
     Norm_EtdV_Init,   alpha_EtdV_Init,   beta_EtdV_Init,   alphap_EtdV_Init,
     R_Et_Sea_Init,    R_Ht_xi2_Init,     R_Et_xi2_Init] = Paralst_Pol

    fit_forw_Et = Minuit(cost_forward_Et, Norm_HtuV = Norm_HtuV_Init,     alpha_HtuV = alpha_HtuV_Init,      beta_HtuV = beta_HtuV_Init,     alphap_HtuV = alphap_HtuV_Init, 
                                          Norm_Htubar = Norm_Htubar_Init, alpha_Htubar = alpha_Htubar_Init,  beta_Htubar = beta_Htubar_Init, alphap_Htqbar = alphap_Htqbar_Init,
                                          Norm_HtdV = Norm_HtdV_Init,     alpha_HtdV = alpha_HtdV_Init,      beta_HtdV = beta_HtdV_Init,     alphap_HtdV = alphap_HtdV_Init,
                                          Norm_Htdbar = Norm_Htdbar_Init, alpha_Htdbar = alpha_Htdbar_Init,  beta_Htdbar = beta_Htdbar_Init, 
                                          Norm_Htg = Norm_Htg_Init,       alpha_Htg = alpha_Htg_Init,        beta_Htg = beta_Htg_Init,       alphap_Htg = alphap_Htg_Init,
                                          Norm_EtuV = Norm_EtuV_Init,     alpha_EtuV = alpha_EtuV_Init,      beta_EtuV = beta_EtuV_Init,     alphap_EtuV = alphap_EtuV_Init, 
                                          Norm_EtdV = Norm_EtdV_Init,     alpha_EtdV = alpha_EtdV_Init,      beta_EtdV = beta_EtdV_Init,     alphap_EtdV = alphap_EtdV_Init,
                                          R_Et_Sea = R_Et_Sea_Init,       R_Ht_xi2 = R_Ht_xi2_Init,          R_Et_xi2 = R_Et_xi2_Init)
    fit_forw_Et.errordef = 1

    fit_forw_Et.limits['alpha_HtuV'] = (-2, 1.2)
    fit_forw_Et.limits['alpha_Htubar'] = (-2, 1.2)
    fit_forw_Et.limits['alpha_HtdV'] = (-2, 1.2)
    fit_forw_Et.limits['alpha_Htdbar'] = (-2, 1.2)
    fit_forw_Et.limits['alpha_Htg'] = (-2, 1.2)
    fit_forw_Et.limits['alpha_EtuV'] = (-2, 1.2)
    fit_forw_Et.limits['alpha_EtdV'] = (-2, 1.2)

    fit_forw_Et.limits['beta_HtuV'] = (0, 15)
    fit_forw_Et.limits['beta_Htubar'] = (0, 15)
    fit_forw_Et.limits['beta_HtdV'] = (0, 15)
    fit_forw_Et.limits['beta_Htdbar'] = (0, 15)
    fit_forw_Et.limits['beta_Htg'] = (0, 15)
    fit_forw_Et.limits['beta_EtuV'] = (0, 15)
    fit_forw_Et.limits['beta_EtdV'] = (0, 15)

    fit_forw_Et.fixed['Norm_HtuV'] = True
    fit_forw_Et.fixed['alpha_HtuV'] = True
    fit_forw_Et.fixed['beta_HtuV'] = True
    fit_forw_Et.fixed['alphap_HtuV'] = True

    fit_forw_Et.fixed['Norm_Htubar'] = True
    fit_forw_Et.fixed['alpha_Htubar'] = True
    fit_forw_Et.fixed['beta_Htubar'] = True
    fit_forw_Et.fixed['alphap_Htqbar'] = True

    fit_forw_Et.fixed['Norm_HtdV'] = True
    fit_forw_Et.fixed['alpha_HtdV'] = True
    fit_forw_Et.fixed['beta_HtdV'] = True
    fit_forw_Et.fixed['alphap_HtdV'] = True

    fit_forw_Et.fixed['Norm_Htdbar'] = True
    fit_forw_Et.fixed['alpha_Htdbar'] = True
    fit_forw_Et.fixed['beta_Htdbar'] = True

    fit_forw_Et.fixed['Norm_Htg'] = True
    fit_forw_Et.fixed['alpha_Htg'] = True
    fit_forw_Et.fixed['beta_Htg'] = True
    fit_forw_Et.fixed['alphap_Htg'] = True

    fit_forw_Et.fixed['R_Ht_xi2'] = True    
    fit_forw_Et.fixed['R_Et_xi2'] = True

    fit_forw_Et.migrad()
    fit_forw_Et.hesse()

    ndof_Et = len(PDF_data_Et.index) + len(tPDF_data_Et.index) + len(GFF_data_Et.index)  - fit_forw_Et.npar

    time_end = time.time() -time_start    
    with open('GUMP_Output/Et_forward_fit.txt', 'w') as f:
        print('Total running time:',round(time_end/60, 1), 'minutes. Total call of cost function:', fit_forw_Et.nfcn, '.\n', file=f)
        print('The chi squared per d.o.f. is:', round(fit_forw_Et.fval/ndof_Et, 3),'.\n', file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*fit_forw_Et.values, sep=", ", file = f)
        print(*fit_forw_Et.errors, sep=", ", file = f)
        print(fit_forw_Et.params, file = f)

    return fit_forw_Et

"""
def cost_GUMP(Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
              Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
              Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
              Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar,  
              Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg,
              R_H_u_xi2,   R_H_d_xi2,    R_H_g_xi2,
              R_E_u,        R_E_d,       R_E_g,       R_E_xi2,
              Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV, 
              Norm_Htubar, alpha_Htubar, beta_Htubar, alphap_Htqbar,
              Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,
              Norm_Htdbar, alpha_Htdbar, beta_Htdbar, 
              Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_Htg,
              R_Ht_u_xi2,  R_Ht_d_xi2,   R_Ht_g_xi2,
              R_Et_u,       R_Et_d,      R_Et_g,      R_Et_xi2):

    global Minuit_Counter, Time_Counter

    time_now = time.time() -time_start
    
    if(time_now > Time_Counter * 600):
        print('Runing Time:',round(time_now/60),'minutes. Cost function called total', Minuit_Counter, 'times.')
        Time_Counter = Time_Counter + 1
    
    Minuit_Counter = Minuit_Counter + 1
    
    Paralst = [Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
               Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
               Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
               Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar,  
               Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg,
               R_H_u_xi2,   R_H_d_xi2,    R_H_g_xi2,
               R_E_u,        R_E_d,       R_E_g,       R_E_xi2,
               Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV, 
               Norm_Htubar, alpha_Htubar, beta_Htubar, alphap_Htqbar,
               Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,
               Norm_Htdbar, alpha_Htdbar, beta_Htdbar, 
               Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_Htg,
               R_Ht_u_xi2,  R_Ht_d_xi2,   R_Ht_g_xi2,
               R_Et_u,       R_Et_d,      R_Et_g,      R_Et_xi2]

    Para_all = ParaManager(np.array(Paralst))

    PDF_pred = np.array(list(pool.map(partial(PDF_theo, Para = Para_all), np.array(PDF_data))))
    cost_PDF = np.sum(((PDF_pred - PDF_data['f'])/ PDF_data['delta f']) ** 2 )

    tPDF_pred = np.array(list(pool.map(partial(tPDF_theo, Para = Para_all), np.array(tPDF_data))))
    cost_tPDF = np.sum(((tPDF_pred - tPDF_data['f'])/ tPDF_data['delta f']) ** 2 )

    GFF_pred = np.array(list(pool.map(partial(GFF_theo, Para = Para_all), np.array(GFF_data))))
    cost_GFF = np.sum(((GFF_pred - GFF_data['f'])/ GFF_data['delta f']) ** 2 )
    
    #cost_DVCS_xBtQ = np.array(list(pool.map(partial(DVCSxsec_cost_xBtQ, Para = Para_all), DVCSxsec_group_data)))
    #cost_DVCSxsec = np.sum(cost_DVCS_xBtQ)

    return  cost_PDF + cost_tPDF + cost_GFF #+ cost_DVCSxsec

def set_GUMP():
    fit = Minuit(cost_GUMP, Norm_HuV = 4.1,    alpha_HuV = 0.3,     beta_HuV = 3.0,    alphap_HuV = 1.1, 
                            Norm_Hubar = 0.2,  alpha_Hubar = 1.1,   beta_Hubar = 7.6,  alphap_Hqbar = 0.15,
                            Norm_HdV = 1.4,    alpha_HdV = 0.5,     beta_HdV = 3.7,    alphap_HdV = 1.3,
                            Norm_Hdbar = 0.2,  alpha_Hdbar = 1.1,   beta_Hdbar = 5.5, 
                            Norm_Hg = 2.4,     alpha_Hg = 0.1,      beta_Hg = 6.8,     alphap_Hg = 1.1,
                            R_H_u_xi2 = 1.0,   R_H_d_xi2 = 1.0,     R_H_g_xi2 = 1.0,
                            R_E_u = 1.0,       R_E_d = 1.0,         R_E_g = 1.0,       R_E_xi2 = 1.0,
                            Norm_HtuV = 11,    alpha_HtuV = -0.5,   beta_HtuV = 3.7,   alphap_HtuV = 1.0, 
                            Norm_Htubar = -30, alpha_Htubar = -1.8, beta_Htubar = 7.8, alphap_Htqbar = 0.15,
                            Norm_HtdV = -0.9,  alpha_HtdV = 0.4,    beta_HtdV = 11,    alphap_HtdV = 1.0,
                            Norm_Htdbar = -30, alpha_Htdbar = -1.8, beta_Htdbar = 7.8,
                            Norm_Htg = 0.4,    alpha_Htg = -0.4,    beta_Htg = 1.5,    alphap_Htg = 1.1,
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

    fit.fixed['R_H_u_xi2'] = True
    fit.fixed['R_H_d_xi2'] = True
    fit.fixed['R_H_g_xi2'] = True
    fit.fixed['R_E_xi2'] = True
    fit.fixed['R_Ht_u_xi2'] = True
    fit.fixed['R_Ht_d_xi2'] = True
    fit.fixed['R_Ht_g_xi2'] = True
    fit.fixed['R_Et_xi2'] = True

    return fit
"""

if __name__ == '__main__':
    pool = Pool()
    time_start = time.time()
    """
    fit_forward_H = forward_H_fit()
    #Paralst_Unp = np.array(fit_forward_H.values)

    Minuit_Counter = 0
    Time_Counter = 1
    time_start = time.time()

    fit_forward_Ht = forward_Ht_fit()
    #Paralst_Pol = np.array(fit_forward_Ht.values)
    """

    Paralst_Unp = [4.069414842032496,   0.2854310655700574, 2.984801319163314,  2.313918289050004,
                   0.15416110374949857, 1.1275378406600627, 7.050379970938083,  8.460138845142314,
                   1.5528552443284596,  0.4494176682500317, 3.9117862194545543, 2.6933791859780487,
                   0.16512984220694593, 1.0830871264105184, 4.9719259240405895,
                   2.4045560425974233,  1.0962757503174405, 6.799743430343084,  1.3477968478412048,
                   4.1,                 0.3,                3.0,                1.1,
                   1.4,                 0.5,                3.7,                1.3,
                   1.0,                 1.0,                1.0]

    Minuit_Counter = 0
    Time_Counter = 1
    time_start = time.time()

    fit_forward_E = forward_E_fit(Paralst_Unp)

    #Paralst_Unp = np.array(fit_forward_E.values)

    Paralst_Pol = [8.603302685534663,   -0.3439437793836446, 3.823339217692947,   2.3930611979605114,
                   -48.18161077749349,  -1.7491336880623545, 14.999994517244774,  14.943027051512095,
                   -1.596476811170268,  0.2732431661236512,  14.023399232686286,  0.8132610624539507,
                   -44.836841779691504, -1.9999964628876339, 8.267812298540168,
                   0.439190177655261,   0.5623448077955673,  1.5558573126417579,  1.1,
                   11.0,                -0.5,                3.7,                 1.0,
                   -0.9,                0.4,                 11.0,                1.0,
                   1.0,                 1.0,                 1.0]

    Minuit_Counter = 0
    Time_Counter = 1
    time_start = time.time()
    
    fit_forward_Et = forward_Et_fit(Paralst_Pol)

    #Paralst_Pol = np.array(fit_forward_Et.values)

    """
    Para_Unp_All = ParaManager_Unp(Paralst_Unp)
    FLV = 'g'
    PDF_data_H_FLV = PDF_data_H[(PDF_data_H['flv'] == FLV)]
    PDF_data_H_FLV_pred = np.array(list(pool.map(partial(PDF_theo, Para = Para_Unp_All), np.array(PDF_data_H_FLV))))
    plot_compare(PDF_data_H_FLV['x'],PDF_data_H_FLV['f'], PDF_data_H_FLV['delta f'], PDF_data_H_FLV_pred)
    """

    """
    FLV = 'NS'
    J = 1
    GFF_data_H_FLV = GFF_data_H[(GFF_data_H['flv'] == FLV) & (GFF_data_H['j'] == J)]
    GFF_data_H_FLV_pred = np.array(list(pool.map(partial(GFF_theo, Para = Para_Unp_All), np.array(GFF_data_H_FLV))))
    plot_compare(GFF_data_H_FLV['t'], GFF_data_H_FLV['f'], GFF_data_H_FLV['delta f'], GFF_data_H_FLV_pred)
    """

    """
    FLV = 'NS'
    SPE = 0
    T = -0.39
    tPDF_data_H_FLV = tPDF_data_H[(tPDF_data_H['flv'] == FLV) & (tPDF_data_H['spe'] == SPE) & (tPDF_data_H['t'] == T)]
    tPDF_data_H_FLV_pred = np.array(list(pool.map(partial(tPDF_theo, Para = Para_Unp_All), np.array(tPDF_data_H_FLV))))
    plot_compare(tPDF_data_H_FLV['x'], tPDF_data_H_FLV['f'], tPDF_data_H_FLV['delta f'], tPDF_data_H_FLV_pred)
    """

    """
    Para_Pol_All = ParaManager_Pol(Paralst_Pol)
    FLV = 'u'
    PDF_data_Ht_FLV = PDF_data_Ht[(PDF_data_Ht['flv'] == FLV)]
    PDF_data_Ht_FLV_pred = np.array(list(pool.map(partial(PDF_theo, Para = Para_Pol_All), np.array(PDF_data_Ht_FLV))))
    plot_compare(PDF_data_Ht_FLV['x'],PDF_data_Ht_FLV['f'], PDF_data_Ht_FLV['delta f'], PDF_data_Ht_FLV_pred)
    """

    """
    FLV = 'NS'
    T = -0.69
    tPDF_data_Ht_FLV = tPDF_data_Ht[(tPDF_data_Ht['flv'] == FLV) & (tPDF_data_Ht['t'] == T)]
    tPDF_data_Ht_FLV_pred = np.array(list(pool.map(partial(tPDF_theo, Para = Para_Pol_All), np.array(tPDF_data_Ht_FLV))))
    plot_compare(tPDF_data_Ht_FLV['x'], tPDF_data_Ht_FLV['f'], tPDF_data_Ht_FLV['delta f'], tPDF_data_Ht_FLV_pred)
    """

    """
    FLV = 'NS'
    J = 1
    GFF_data_Ht_FLV = GFF_data_Ht[(GFF_data_Ht['flv'] == FLV) & (GFF_data_Ht['j'] == J)]
    GFF_data_Ht_FLV_pred = np.array(list(pool.map(partial(GFF_theo, Para = Para_Pol_All), np.array(GFF_data_Ht_FLV))))
    plot_compare(GFF_data_Ht_FLV['t'], GFF_data_Ht_FLV['f'], GFF_data_Ht_FLV['delta f'], GFF_data_Ht_FLV_pred)
    """

    """
    fit_GUMP.migrad()
    time_migrad = time.time() 
    print('The migard runs for: ', round((time_migrad - time_start)/60, 1), 'minutes.')

    fit_GUMP.hesse()
    time_hesse = time.time()
    print('The hesse runs for: ', round((time_hesse - time_migrad)/60, 1), 'minutes.')

    pool.close()
    pool.join()

    ndof = len(PDF_data.index) + len(tPDF_data.index) + len(GFF_data.index)  - fit_GUMP.npar #+ len(DVCSxsec_data.index)

    time_end = time.time() -time_start    
    with open('Output.txt', 'w') as f:
        print('Total running time:',round(time_end/60, 1), 'minutes. Total call of cost function:',Minuit_Counter,'(or', fit_GUMP.nfcn, 'from Minuit).\n', file=f)
        print('The chi squared per d.o.f. is:', round(fit_GUMP.fval/ndof, 3),'.\n', file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(fit_GUMP.params, file = f)
        print('Below are the output covariance from iMinuit:', file = f)
        print(fit_GUMP.covariance, file = f)
    """