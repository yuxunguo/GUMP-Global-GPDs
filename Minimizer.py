from Parameters import ParaManager_Unp, ParaManager_Pol
from Observables import GPDobserv
from DVCS_xsec import dsigma_TOT, dsigma_DVCS_HERA, M
from multiprocessing import Pool
from functools import partial
from iminuit import Minuit
import numpy as np
import pandas as pd
import time

Minuit_Counter = 0

Time_Counter = 1

Q_threshold = 1.9

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

GFF_data = pd.read_csv('GUMPDATA/GFFdata_Quark.csv',       header = None, names = ['j', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'j': int, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})
GFF_data_H  = GFF_data[GFF_data['spe'] == 0]
GFF_data_E  = GFF_data[GFF_data['spe'] == 1]
GFF_data_Ht = GFF_data[GFF_data['spe'] == 2]
GFF_data_Et = GFF_data[GFF_data['spe'] == 3]

GFF_Gluon_data = pd.read_csv('GUMPDATA/GFFdata_Gluon.csv',       header = None, names = ['j', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'j': int, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})
GFF_Gluon_data_H  = GFF_Gluon_data[GFF_Gluon_data['spe'] == 0]
GFF_Gluon_data_E  = GFF_Gluon_data[GFF_Gluon_data['spe'] == 1]
GFF_Gluon_data_Ht = GFF_Gluon_data[GFF_Gluon_data['spe'] == 2]
GFF_Gluon_data_Et = GFF_Gluon_data[GFF_Gluon_data['spe'] == 3]

DVCSxsec_data = pd.read_csv('GUMPDATA/DVCSxsec.csv', header = None, names = ['y', 'xB', 't', 'Q', 'phi', 'f', 'delta f', 'pol'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'phi': float, 'f': float, 'delta f': float, 'pol': str})
DVCSxsec_data_invalid = DVCSxsec_data[DVCSxsec_data['t']*(DVCSxsec_data['xB']-1) - M ** 2 * DVCSxsec_data['xB'] ** 2 < 0]
DVCSxsec_data = DVCSxsec_data[(DVCSxsec_data['Q'] > Q_threshold) & (DVCSxsec_data['xB'] < xB_Cut) & (DVCSxsec_data['t']*(DVCSxsec_data['xB']-1) - M ** 2 * DVCSxsec_data['xB'] ** 2 > 0)]
xBtQlst = DVCSxsec_data.drop_duplicates(subset = ['xB', 't', 'Q'], keep = 'first')[['xB','t','Q']].values.tolist()
DVCSxsec_group_data = list(map(lambda set: DVCSxsec_data[(DVCSxsec_data['xB'] == set[0]) & (DVCSxsec_data['t'] == set[1]) & ((DVCSxsec_data['Q'] == set[2]))], xBtQlst))

DVCS_HERA_data = pd.read_csv('GUMPDATA/DVCSxsec_HERA.csv', header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f', 'pol'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float, 'pol': str})


def PDF_theo(PDF_input: pd.DataFrame, Para: np.array):
    # [x, t, Q, f, delta_f, spe, flv] = PDF_input
    xs = PDF_input['x'].to_numpy()
    ts = PDF_input['t'].to_numpy()
    Qs = PDF_input['Q'].to_numpy()
    flvs = PDF_input['flv'].to_numpy()
    spes = PDF_input['spe'].to_numpy()
 
    xi = 0

    ps = np.where(spes<=1, 1, -1)
    spes = np.where(spes<=1, spes, spes-2)

    '''
    if(spe == 0 or spe == 1):
        spe, p = spe, 1

    if(spe == 2 or spe == 3):
        spe, p = spe - 2 , -1
    '''
    # Para: (4, 2, 5, 1, 4)

    Para_spe = Para[spes] # fancy indexing. Output (N, 3, 5, 1, 5)
    PDF_theo = GPDobserv(xs, xi, ts, Qs, ps)
    return PDF_theo.tPDF(flvs, Para_spe)  # array length N

tPDF_theo = PDF_theo
      

def GFF_theo(GFF_input: np.array, Para):
    # [j, t, Q, f, delta_f, spe, flv] = GFF_input
    js = GFF_input['j'].to_numpy()
    ts = GFF_input['t'].to_numpy()
    Qs = GFF_input['Q'].to_numpy()
    #fs = GFF_input['f'].to_numpy()
    #delta_fs = GFF_input['delta f'].to_numpy()
    flvs = GFF_input['flv'].to_numpy()
    spes = GFF_input['spe'].to_numpy()
    x = 0
    xi = 0   
    '''
    if(spe == 0 or spe == 1):
        spe, p = spe, 1

    if(spe == 2 or spe == 3):
        spe, p = spe - 2 , -1
    '''
    ps = np.where(spes<=1, 1, -1)
    spes = np.where(spes<=1, spes, spes-2)

    Para_spe = Para[spes] # fancy indexing. Output (N, 3, 5, 1, 5)
    GFF_theo = GPDobserv(x, xi, ts, Qs, ps)
    return GFF_theo.GFFj0(js, flvs, Para_spe) # (N)

def CFF_theo(xB, t, Q, Para_Unp, Para_Pol):
    x = 0
    xi = (1/(2 - xB) - (2*t*(-1 + xB))/(Q**2*(-2 + xB)**2))*xB
    H_E = GPDobserv(x, xi, t, Q, 1)
    Ht_Et = GPDobserv(x, xi, t, Q, -1)
    HCFF = H_E.CFF(Para_Unp[..., 0, :, :, :, :])
    ECFF = H_E.CFF(Para_Unp[..., 1, :, :, :, :])
    HtCFF = Ht_Et.CFF(Para_Pol[..., 0, :, :, :, :])
    EtCFF = Ht_Et.CFF(Para_Pol[..., 1, :, :, :, :])

    return [ HCFF, ECFF, HtCFF, EtCFF ] # this can be a list of arrays of shape (N)
    # return np.stack([HCFF, ECFF, HtCFF, EtCFF], axis=-1)

def DVCSxsec_theo(DVCSxsec_input: pd.DataFrame, CFF_input: np.array):
    # CFF_input is a list of np.arrays
    # [y, xB, t, Q, phi, f, delta_f, pol] = DVCSxsec_input    

    y = DVCSxsec_input['y'].to_numpy()
    xB = DVCSxsec_input['xB'].to_numpy()
    t = DVCSxsec_input['t'].to_numpy()
    Q = DVCSxsec_input['Q'].to_numpy()
    phi = DVCSxsec_input['phi'].to_numpy()
    #f = DVCSxsec_input['f'].to_numpy()
    pol = DVCSxsec_input['pol'].to_numpy()

    [HCFF, ECFF, HtCFF, EtCFF] = CFF_input # each of them have shape (N); scalar is also OK if we use 
    return dsigma_TOT(y, xB, t, Q, phi, pol, HCFF, ECFF, HtCFF, EtCFF)

def DVCSxsec_cost_xBtQ(DVCSxsec_data_xBtQ: pd.DataFrame, Para_Unp, Para_Pol):
    [xB, t, Q] = [DVCSxsec_data_xBtQ['xB'].iat[0], DVCSxsec_data_xBtQ['t'].iat[0], DVCSxsec_data_xBtQ['Q'].iat[0]] 
    [HCFF, ECFF, HtCFF, EtCFF] = CFF_theo(xB, t, Q, Para_Unp, Para_Pol) # scalar for each of them
    # DVCS_pred_xBtQ = np.array(list(map(partial(DVCSxsec_theo, CFF_input = [HCFF, ECFF, HtCFF, EtCFF]), np.array(DVCSxsec_data_xBtQ))))
    DVCS_pred_xBtQ = DVCSxsec_theo(DVCSxsec_data_xBtQ, CFF_input = [HCFF, ECFF, HtCFF, EtCFF])
    return np.sum(((DVCS_pred_xBtQ - DVCSxsec_data_xBtQ['f'])/ DVCSxsec_data_xBtQ['delta f']) ** 2 )

def DVCSxsec_HERA_theo(DVCSxsec_data_HERA: pd.DataFrame, Para_Unp, Para_Pol):
    # [y, xB, t, Q, f, delta_f, pol]  = DVCSxsec_data_HERA
    y = DVCSxsec_data_HERA['y'].to_numpy()
    xB = DVCSxsec_data_HERA['xB'].to_numpy()
    t = DVCSxsec_data_HERA['t'].to_numpy()
    Q = DVCSxsec_data_HERA['Q'].to_numpy()
    #f = DVCSxsec_data_HERA['f'].to_numpy()
    #delta_f = DVCSxsec_data_HERA['delta f'].to_numpy()
    pol = DVCSxsec_data_HERA['pol'].to_numpy()

    [HCFF, ECFF, HtCFF, EtCFF] = CFF_theo(xB, t, Q, np.expand_dims(Para_Unp, axis=0), np.expand_dims(Para_Pol, axis=0))
    return dsigma_DVCS_HERA(y, xB, t, Q, pol, HCFF, ECFF, HtCFF, EtCFF)

def cost_forward_H(Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
                   Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
                   Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
                   Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar, 
                   Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg,
                   Norm_EuV,    alpha_EuV,    beta_EuV,    alphap_EuV,
                   Norm_EdV,    R_E_Sea,      R_Hu_xi2,    R_Hd_xi2,    R_Hg_xi2,
                   R_Eu_xi2,    R_Ed_xi2,     R_Eg_xi2,
                   R_Hu_xi4,    R_Hd_xi4,     R_Hg_xi4,
                   R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea):

    Paralst = [Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
               Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
               Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
               Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar, 
               Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg,
               Norm_EuV,    alpha_EuV,    beta_EuV,    alphap_EuV,
               Norm_EdV,    R_E_Sea,      R_Hu_xi2,    R_Hd_xi2,    R_Hg_xi2,
               R_Eu_xi2,    R_Ed_xi2,     R_Eg_xi2,
               R_Hu_xi4,    R_Hd_xi4,     R_Hg_xi4,
               R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea]
    
    Para_all = ParaManager_Unp(Paralst)
    # PDF_H_pred = np.array(list(pool.map(partial(PDF_theo, Para = Para_all), np.array(PDF_data_H))))
    PDF_H_pred = PDF_theo(PDF_data_H, Para=Para_all)
    cost_PDF_H = np.sum(((PDF_H_pred - PDF_data_H['f'])/ PDF_data_H['delta f']) ** 2 )

    # tPDF_H_pred = np.array(list(pool.map(partial(tPDF_theo, Para = Para_all), np.array(tPDF_data_H))))
    tPDF_H_pred = tPDF_theo(tPDF_data_H, Para=Para_all)
    cost_tPDF_H = np.sum(((tPDF_H_pred - tPDF_data_H['f'])/ tPDF_data_H['delta f']) ** 2 )

    # GFF_H_pred = np.array(list(pool.map(partial(GFF_theo, Para = Para_all), np.array(GFF_data_H))))
    GFF_H_pred = GFF_theo(GFF_data_H, Para=Para_all)
    cost_GFF_H = np.sum(((GFF_H_pred - GFF_data_H['f'])/ GFF_data_H['delta f']) ** 2 )

    return cost_PDF_H + cost_tPDF_H + cost_GFF_H

def cost_forward_E(Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
                   Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
                   Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
                   Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar, 
                   Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg,
                   Norm_EuV,    alpha_EuV,    beta_EuV,    alphap_EuV,
                   Norm_EdV,    R_E_Sea,      R_Hu_xi2,    R_Hd_xi2,    R_Hg_xi2,
                   R_Eu_xi2,    R_Ed_xi2,     R_Eg_xi2,
                   R_Hu_xi4,    R_Hd_xi4,     R_Hg_xi4,
                   R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea):

    Paralst = [Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
               Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
               Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
               Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar, 
               Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg,
               Norm_EuV,    alpha_EuV,    beta_EuV,    alphap_EuV,
               Norm_EdV,    R_E_Sea,      R_Hu_xi2,    R_Hd_xi2,    R_Hg_xi2,
               R_Eu_xi2,    R_Ed_xi2,     R_Eg_xi2,
               R_Hu_xi4,    R_Hd_xi4,     R_Hg_xi4,
               R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea]
    
    Para_all = ParaManager_Unp(Paralst)
    # PDF_E_pred = np.array(list(pool.map(partial(PDF_theo, Para = Para_all), np.array(PDF_data_E))))
    PDF_E_pred = PDF_theo(PDF_data_E, Para=Para_all)
    cost_PDF_E = np.sum(((PDF_E_pred - PDF_data_E['f'])/ PDF_data_E['delta f']) ** 2 )

    # tPDF_E_pred = np.array(list(pool.map(partial(tPDF_theo, Para = Para_all), np.array(tPDF_data_E))))
    tPDF_E_pred = tPDF_theo(tPDF_data_E, Para=Para_all)
    cost_tPDF_E = np.sum(((tPDF_E_pred - tPDF_data_E['f'])/ tPDF_data_E['delta f']) ** 2 )

    # GFF_E_pred = np.array(list(pool.map(partial(GFF_theo, Para = Para_all), np.array(GFF_data_E))))
    GFF_E_pred = GFF_theo(GFF_data_E, Para=Para_all)
    cost_GFF_E = np.sum(((GFF_E_pred - GFF_data_E['f'])/ GFF_data_E['delta f']) ** 2 )

    return cost_PDF_E + cost_tPDF_E + cost_GFF_E

def forward_H_fit(Paralst_Unp):

    [Norm_HuV_Init,    alpha_HuV_Init,    beta_HuV_Init,    alphap_HuV_Init, 
     Norm_Hubar_Init,  alpha_Hubar_Init,  beta_Hubar_Init,  alphap_Hqbar_Init,
     Norm_HdV_Init,    alpha_HdV_Init,    beta_HdV_Init,    alphap_HdV_Init,
     Norm_Hdbar_Init,  alpha_Hdbar_Init,  beta_Hdbar_Init, 
     Norm_Hg_Init,     alpha_Hg_Init,     beta_Hg_Init,     alphap_Hg_Init,
     Norm_EuV_Init,    alpha_EuV_Init,    beta_EuV_Init,    alphap_EuV_Init,
     Norm_EdV_Init,    R_E_Sea_Init,      R_Hu_xi2_Init,    R_Hd_xi2_Init,    R_Hg_xi2_Init,
     R_Eu_xi2_Init,    R_Ed_xi2_Init,     R_Eg_xi2_Init,
     R_Hu_xi4_Init,    R_Hd_xi4_Init,     R_Hg_xi4_Init,
     R_Eu_xi4_Init,    R_Ed_xi4_Init,     R_Eg_xi4_Init,    bexp_HSea_Init] = Paralst_Unp

    fit_forw_H = Minuit(cost_forward_H, Norm_HuV = Norm_HuV_Init,     alpha_HuV = alpha_HuV_Init,      beta_HuV = beta_HuV_Init,     alphap_HuV = alphap_HuV_Init, 
                                        Norm_Hubar = Norm_Hubar_Init, alpha_Hubar = alpha_Hubar_Init,  beta_Hubar = beta_Hubar_Init, alphap_Hqbar = alphap_Hqbar_Init,
                                        Norm_HdV = Norm_HdV_Init,     alpha_HdV = alpha_HdV_Init,      beta_HdV = beta_HdV_Init,     alphap_HdV = alphap_HdV_Init,
                                        Norm_Hdbar = Norm_Hdbar_Init, alpha_Hdbar = alpha_Hdbar_Init,  beta_Hdbar = beta_Hdbar_Init, 
                                        Norm_Hg = Norm_Hg_Init,       alpha_Hg = alpha_Hg_Init,        beta_Hg = beta_Hg_Init,       alphap_Hg = alphap_Hg_Init,
                                        Norm_EuV = Norm_EuV_Init,     alpha_EuV = alpha_EuV_Init,      beta_EuV = beta_EuV_Init,     alphap_EuV = alphap_EuV_Init, 
                                        Norm_EdV = Norm_EdV_Init,     R_E_Sea = R_E_Sea_Init,          R_Hu_xi2 = R_Hu_xi2_Init,     R_Hd_xi2 = R_Hd_xi2_Init,     R_Hg_xi2 = R_Hg_xi2_Init,
                                        R_Eu_xi2 = R_Eu_xi2_Init,     R_Ed_xi2 = R_Ed_xi2_Init,        R_Eg_xi2 = R_Eg_xi2_Init,
                                        R_Hu_xi4 = R_Hu_xi4_Init,     R_Hd_xi4 = R_Hd_xi4_Init,        R_Hg_xi4 = R_Hg_xi4_Init,
                                        R_Eu_xi4 = R_Eu_xi4_Init,     R_Ed_xi4 = R_Ed_xi4_Init,        R_Eg_xi4 = R_Eg_xi4_Init,     bexp_HSea = bexp_HSea_Init)
    fit_forw_H.errordef = 1

    fit_forw_H.limits['alpha_HuV'] = (-2, 1.2)
    fit_forw_H.limits['alpha_Hubar'] = (-2, 1.2)
    fit_forw_H.limits['alpha_HdV'] = (-2, 1.2)
    fit_forw_H.limits['alpha_Hdbar'] = (-2, 1.2)
    fit_forw_H.limits['alpha_Hg'] = (-2, 1.2)
    fit_forw_H.limits['alpha_EuV'] = (-2, 1.2)

    fit_forw_H.limits['beta_HuV'] = (0, 15)
    fit_forw_H.limits['beta_Hubar'] = (0, 15)
    fit_forw_H.limits['beta_HdV'] = (0, 15)
    fit_forw_H.limits['beta_Hdbar'] = (0, 15)
    fit_forw_H.limits['beta_Hg'] = (0, 15)    
    fit_forw_H.limits['beta_EuV'] = (0, 15)

    fit_forw_H.fixed['alphap_Hqbar'] = True
    fit_forw_H.fixed['alphap_Hg'] = True

    fit_forw_H.fixed['Norm_EuV'] = True
    fit_forw_H.fixed['alpha_EuV'] = True
    fit_forw_H.fixed['beta_EuV'] = True
    fit_forw_H.fixed['alphap_EuV'] = True

    fit_forw_H.fixed['Norm_EdV'] = True

    fit_forw_H.fixed['R_E_Sea'] = True
    fit_forw_H.fixed['R_Hu_xi2'] = True
    fit_forw_H.fixed['R_Hd_xi2'] = True 
    fit_forw_H.fixed['R_Hg_xi2'] = True 
    fit_forw_H.fixed['R_Eu_xi2'] = True
    fit_forw_H.fixed['R_Ed_xi2'] = True
    fit_forw_H.fixed['R_Eg_xi2'] = True

    fit_forw_H.fixed['R_Hu_xi4'] = True
    fit_forw_H.fixed['R_Hd_xi4'] = True 
    fit_forw_H.fixed['R_Hg_xi4'] = True 
    fit_forw_H.fixed['R_Eu_xi4'] = True
    fit_forw_H.fixed['R_Ed_xi4'] = True
    fit_forw_H.fixed['R_Eg_xi4'] = True

    fit_forw_H.fixed['bexp_HSea'] = True

    global time_start
    time_start = time.time()

    fit_forw_H.migrad()
    fit_forw_H.hesse()

    ndof_H = len(PDF_data_H.index) + len(tPDF_data_H.index) + len(GFF_data_H.index)  - fit_forw_H.nfit 

    time_end = time.time() -time_start

    with open('GUMP_Output/H_forward_fit.txt', 'w') as f:
        print('Total running time: %3.1f minutes. Total call of cost function: %3d.\n' % ( time_end/60, fit_forw_H.nfcn), file=f)
        print('The chi squared/d.o.f. is: %3.1f / %3d ( = %3.1f ).\n' % (fit_forw_H.fval, ndof_H, fit_forw_H.fval/ndof_H), file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*fit_forw_H.values, sep=", ", file = f)
        print(*fit_forw_H.errors, sep=", ", file = f)
        print(fit_forw_H.params, file = f)
        print(fit_forw_H.covariance, file = f)
    print("H fit finished...")
    return fit_forw_H

def forward_E_fit(Paralst_Unp):

    [Norm_HuV_Init,    alpha_HuV_Init,    beta_HuV_Init,    alphap_HuV_Init, 
     Norm_Hubar_Init,  alpha_Hubar_Init,  beta_Hubar_Init,  alphap_Hqbar_Init,
     Norm_HdV_Init,    alpha_HdV_Init,    beta_HdV_Init,    alphap_HdV_Init,
     Norm_Hdbar_Init,  alpha_Hdbar_Init,  beta_Hdbar_Init, 
     Norm_Hg_Init,     alpha_Hg_Init,     beta_Hg_Init,     alphap_Hg_Init,
     Norm_EuV_Init,    alpha_EuV_Init,    beta_EuV_Init,    alphap_EuV_Init,
     Norm_EdV_Init,    R_E_Sea_Init,      R_Hu_xi2_Init,    R_Hd_xi2_Init,    R_Hg_xi2_Init,
     R_Eu_xi2_Init,    R_Ed_xi2_Init,     R_Eg_xi2_Init,
     R_Hu_xi4_Init,    R_Hd_xi4_Init,     R_Hg_xi4_Init,
     R_Eu_xi4_Init,    R_Ed_xi4_Init,     R_Eg_xi4_Init,    bexp_HSea_Init] = Paralst_Unp

    fit_forw_E = Minuit(cost_forward_E, Norm_HuV = Norm_HuV_Init,     alpha_HuV = alpha_HuV_Init,      beta_HuV = beta_HuV_Init,     alphap_HuV = alphap_HuV_Init, 
                                        Norm_Hubar = Norm_Hubar_Init, alpha_Hubar = alpha_Hubar_Init,  beta_Hubar = beta_Hubar_Init, alphap_Hqbar = alphap_Hqbar_Init,
                                        Norm_HdV = Norm_HdV_Init,     alpha_HdV = alpha_HdV_Init,      beta_HdV = beta_HdV_Init,     alphap_HdV = alphap_HdV_Init,
                                        Norm_Hdbar = Norm_Hdbar_Init, alpha_Hdbar = alpha_Hdbar_Init,  beta_Hdbar = beta_Hdbar_Init, 
                                        Norm_Hg = Norm_Hg_Init,       alpha_Hg = alpha_Hg_Init,        beta_Hg = beta_Hg_Init,       alphap_Hg = alphap_Hg_Init,
                                        Norm_EuV = Norm_EuV_Init,     alpha_EuV = alpha_EuV_Init,      beta_EuV = beta_EuV_Init,     alphap_EuV = alphap_EuV_Init, 
                                        Norm_EdV = Norm_EdV_Init,     R_E_Sea = R_E_Sea_Init,          R_Hu_xi2 = R_Hu_xi2_Init,     R_Hd_xi2 = R_Hd_xi2_Init,     R_Hg_xi2 = R_Hg_xi2_Init,
                                        R_Eu_xi2 = R_Eu_xi2_Init,     R_Ed_xi2 = R_Ed_xi2_Init,        R_Eg_xi2 = R_Eg_xi2_Init,
                                        R_Hu_xi4 = R_Hu_xi4_Init,     R_Hd_xi4 = R_Hd_xi4_Init,        R_Hg_xi4 = R_Hg_xi4_Init,
                                        R_Eu_xi4 = R_Eu_xi4_Init,     R_Ed_xi4 = R_Ed_xi4_Init,        R_Eg_xi4 = R_Eg_xi4_Init,     bexp_HSea = bexp_HSea_Init)
    fit_forw_E.errordef = 1

    fit_forw_E.limits['alpha_HuV'] = (-2, 1.2)
    fit_forw_E.limits['alpha_Hubar'] = (-2, 1.2)
    fit_forw_E.limits['alpha_HdV'] = (-2, 1.2)
    fit_forw_E.limits['alpha_Hdbar'] = (-2, 1.2)
    fit_forw_E.limits['alpha_Hg'] = (-2, 1.2)
    fit_forw_E.limits['alpha_EuV'] = (-2, 1.2)

    fit_forw_E.limits['beta_HuV'] = (0, 15)
    fit_forw_E.limits['beta_Hubar'] = (0, 15)
    fit_forw_E.limits['beta_HdV'] = (0, 15)
    fit_forw_E.limits['beta_Hdbar'] = (0, 15)
    fit_forw_E.limits['beta_Hg'] = (0, 15)    
    fit_forw_E.limits['beta_EuV'] = (0, 15)

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

    fit_forw_E.fixed['R_Hu_xi2'] = True
    fit_forw_E.fixed['R_Hd_xi2'] = True 
    fit_forw_E.fixed['R_Hg_xi2'] = True 
    fit_forw_E.fixed['R_Eu_xi2'] = True
    fit_forw_E.fixed['R_Ed_xi2'] = True
    fit_forw_E.fixed['R_Eg_xi2'] = True

    fit_forw_E.fixed['R_Hu_xi4'] = True
    fit_forw_E.fixed['R_Hd_xi4'] = True 
    fit_forw_E.fixed['R_Hg_xi4'] = True 
    fit_forw_E.fixed['R_Eu_xi4'] = True
    fit_forw_E.fixed['R_Ed_xi4'] = True
    fit_forw_E.fixed['R_Eg_xi4'] = True

    fit_forw_E.fixed['bexp_HSea'] = True

    global time_start
    time_start = time.time()
    
    fit_forw_E.migrad()
    fit_forw_E.hesse()

    ndof_E = len(PDF_data_E.index) + len(tPDF_data_E.index) + len(GFF_data_E.index)  - fit_forw_E.nfit 

    time_end = time.time() -time_start

    with open('GUMP_Output/E_forward_fit.txt', 'w') as f:
        print('Total running time: %3.1f minutes. Total call of cost function: %3d.\n' % ( time_end/60, fit_forw_E.nfcn), file=f)
        print('The chi squared/d.o.f. is: %3.1f / %3d ( = %3.1f ).\n' % (fit_forw_E.fval, ndof_E, fit_forw_E.fval/ndof_E), file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*fit_forw_E.values, sep=", ", file = f)
        print(*fit_forw_E.errors, sep=", ", file = f)
        print(fit_forw_E.params, file = f)
        print(fit_forw_E.covariance, file = f)
    print("E fit finished...")
    return fit_forw_E

def cost_forward_Ht(Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV, 
                    Norm_Htubar, alpha_Htubar, beta_Htubar, alphap_Htqbar,
                    Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,
                    Norm_Htdbar, alpha_Htdbar, beta_Htdbar, 
                    Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_Htg,
                    Norm_EtuV,   alpha_EtuV,   beta_EtuV,   alphap_EtuV,
                    Norm_EtdV,   R_Et_Sea,     R_Htu_xi2,   R_Htd_xi2,    R_Htg_xi2,
                    R_Etu_xi2,   R_Etd_xi2,    R_Etg_xi2,
                    R_Htu_xi4,   R_Htd_xi4,    R_Htg_xi4,
                    R_Etu_xi4,   R_Etd_xi4,    R_Etg_xi4,   bexp_HtSea):

    Paralst = [Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV, 
               Norm_Htubar, alpha_Htubar, beta_Htubar, alphap_Htqbar,
               Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,
               Norm_Htdbar, alpha_Htdbar, beta_Htdbar, 
               Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_Htg,
               Norm_EtuV,   alpha_EtuV,   beta_EtuV,   alphap_EtuV,
               Norm_EtdV,   R_Et_Sea,     R_Htu_xi2,   R_Htd_xi2,    R_Htg_xi2,
               R_Etu_xi2,   R_Etd_xi2,    R_Etg_xi2,
               R_Htu_xi4,   R_Htd_xi4,    R_Htg_xi4,
               R_Etu_xi4,   R_Etd_xi4,    R_Etg_xi4,   bexp_HtSea]
    
    Para_all = ParaManager_Pol(Paralst)
    # PDF_Ht_pred = np.array(list(pool.map(partial(PDF_theo, Para = Para_all), np.array(PDF_data_Ht))))
    PDF_Ht_pred = PDF_theo(PDF_data_Ht, Para=Para_all)
    cost_PDF_Ht = np.sum(((PDF_Ht_pred - PDF_data_Ht['f'])/ PDF_data_Ht['delta f']) ** 2 )

    # tPDF_Ht_pred = np.array(list(pool.map(partial(tPDF_theo, Para = Para_all), np.array(tPDF_data_Ht))))
    tPDF_Ht_pred = tPDF_theo(tPDF_data_Ht, Para=Para_all)
    cost_tPDF_Ht = np.sum(((tPDF_Ht_pred - tPDF_data_Ht['f'])/ tPDF_data_Ht['delta f']) ** 2 )

    # GFF_Ht_pred = np.array(list(pool.map(partial(GFF_theo, Para = Para_all), np.array(GFF_data_Ht))))
    GFF_Ht_pred = GFF_theo(GFF_data_Ht, Para=Para_all)
    cost_GFF_Ht = np.sum(((GFF_Ht_pred - GFF_data_Ht['f'])/ GFF_data_Ht['delta f']) ** 2 )

    return cost_PDF_Ht + cost_tPDF_Ht + cost_GFF_Ht

def cost_forward_Et(Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV, 
                    Norm_Htubar, alpha_Htubar, beta_Htubar, alphap_Htqbar,
                    Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,
                    Norm_Htdbar, alpha_Htdbar, beta_Htdbar, 
                    Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_Htg,
                    Norm_EtuV,   alpha_EtuV,   beta_EtuV,   alphap_EtuV,
                    Norm_EtdV,   R_Et_Sea,     R_Htu_xi2,   R_Htd_xi2,    R_Htg_xi2,
                    R_Etu_xi2,   R_Etd_xi2,    R_Etg_xi2,
                    R_Htu_xi4,   R_Htd_xi4,    R_Htg_xi4,
                    R_Etu_xi4,   R_Etd_xi4,    R_Etg_xi4,   bexp_HtSea):

    Paralst = [Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV, 
               Norm_Htubar, alpha_Htubar, beta_Htubar, alphap_Htqbar,
               Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,
               Norm_Htdbar, alpha_Htdbar, beta_Htdbar, 
               Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_Htg,
               Norm_EtuV,   alpha_EtuV,   beta_EtuV,   alphap_EtuV,
               Norm_EtdV,   R_Et_Sea,     R_Htu_xi2,   R_Htd_xi2,    R_Htg_xi2,
               R_Etu_xi2,   R_Etd_xi2,    R_Etg_xi2,
               R_Htu_xi4,   R_Htd_xi4,    R_Htg_xi4,
               R_Etu_xi4,   R_Etd_xi4,    R_Etg_xi4,   bexp_HtSea]
    
    Para_all = ParaManager_Pol(Paralst)
    # PDF_Et_pred = np.array(list(pool.map(partial(PDF_theo, Para = Para_all), np.array(PDF_data_Et))))
    PDF_Et_pred = PDF_theo(PDF_data_Et, Para=Para_all)
    cost_PDF_Et = np.sum(((PDF_Et_pred - PDF_data_Et['f'])/ PDF_data_Et['delta f']) ** 2 )

    # tPDF_Et_pred = np.array(list(pool.map(partial(tPDF_theo, Para = Para_all), np.array(tPDF_data_Et))))
    tPDF_Et_pred = tPDF_theo(tPDF_data_Et, Para=Para_all)
    cost_tPDF_Et = np.sum(((tPDF_Et_pred - tPDF_data_Et['f'])/ tPDF_data_Et['delta f']) ** 2 )

    # GFF_Et_pred = np.array(list(pool.map(partial(GFF_theo, Para = Para_all), np.array(GFF_data_Et))))
    GFF_Et_pred = GFF_theo(GFF_data_Et, Para=Para_all)
    cost_GFF_Et = np.sum(((GFF_Et_pred - GFF_data_Et['f'])/ GFF_data_Et['delta f']) ** 2 )

    return cost_PDF_Et + cost_tPDF_Et + cost_GFF_Et

def forward_Ht_fit(Paralst_Pol):

    [Norm_HtuV_Init,   alpha_HtuV_Init,   beta_HtuV_Init,   alphap_HtuV_Init, 
     Norm_Htubar_Init, alpha_Htubar_Init, beta_Htubar_Init, alphap_Htqbar_Init,
     Norm_HtdV_Init,   alpha_HtdV_Init,   beta_HtdV_Init,   alphap_HtdV_Init,
     Norm_Htdbar_Init, alpha_Htdbar_Init, beta_Htdbar_Init, 
     Norm_Htg_Init,    alpha_Htg_Init,    beta_Htg_Init,    alphap_Htg_Init,
     Norm_EtuV_Init,   alpha_EtuV_Init,   beta_EtuV_Init,   alphap_EtuV_Init,
     Norm_EtdV_Init,   R_Et_Sea_Init,     R_Htu_xi2_Init,   R_Htd_xi2_Init,    R_Htg_xi2_Init,
     R_Etu_xi2_Init,   R_Etd_xi2_Init,    R_Etg_xi2_Init,
     R_Htu_xi4_Init,   R_Htd_xi4_Init,    R_Htg_xi4_Init,
     R_Etu_xi4_Init,   R_Etd_xi4_Init,    R_Etg_xi4_Init,   bexp_HtSea_Init] = Paralst_Pol

    fit_forw_Ht = Minuit(cost_forward_Ht, Norm_HtuV = Norm_HtuV_Init,     alpha_HtuV = alpha_HtuV_Init,      beta_HtuV = beta_HtuV_Init,     alphap_HtuV = alphap_HtuV_Init, 
                                          Norm_Htubar = Norm_Htubar_Init, alpha_Htubar = alpha_Htubar_Init,  beta_Htubar = beta_Htubar_Init, alphap_Htqbar = alphap_Htqbar_Init,
                                          Norm_HtdV = Norm_HtdV_Init,     alpha_HtdV = alpha_HtdV_Init,      beta_HtdV = beta_HtdV_Init,     alphap_HtdV = alphap_HtdV_Init,
                                          Norm_Htdbar = Norm_Htdbar_Init, alpha_Htdbar = alpha_Htdbar_Init,  beta_Htdbar = beta_Htdbar_Init, 
                                          Norm_Htg = Norm_Htg_Init,       alpha_Htg = alpha_Htg_Init,        beta_Htg = beta_Htg_Init,       alphap_Htg = alphap_Htg_Init,
                                          Norm_EtuV = Norm_EtuV_Init,     alpha_EtuV = alpha_EtuV_Init,      beta_EtuV = beta_EtuV_Init,     alphap_EtuV = alphap_EtuV_Init,
                                          Norm_EtdV = Norm_EtdV_Init,     R_Et_Sea = R_Et_Sea_Init,          R_Htu_xi2 = R_Htu_xi2_Init,     R_Htd_xi2 = R_Htd_xi2_Init,        R_Htg_xi2 = R_Htg_xi2_Init,
                                          R_Etu_xi2 = R_Etu_xi2_Init,     R_Etd_xi2 = R_Etd_xi2_Init,        R_Etg_xi2 = R_Etg_xi2_Init,
                                          R_Htu_xi4 = R_Htu_xi4_Init,     R_Htd_xi4 = R_Htd_xi4_Init,        R_Htg_xi4 = R_Htg_xi4_Init,
                                          R_Etu_xi4 = R_Etu_xi4_Init,     R_Etd_xi4 = R_Etd_xi4_Init,        R_Etg_xi4 = R_Etg_xi4_Init,     bexp_HtSea = bexp_HtSea_Init)
    fit_forw_Ht.errordef = 1

    fit_forw_Ht.limits['alpha_HtuV'] = (-2, 1.2)
    fit_forw_Ht.limits['alpha_Htubar'] = (-2, 1.2)
    fit_forw_Ht.limits['alpha_HtdV'] = (-2, 1.2)
    fit_forw_Ht.limits['alpha_Htdbar'] = (-2, 1.2)
    fit_forw_Ht.limits['alpha_Htg'] = (-2, 1.2)
    fit_forw_Ht.limits['alpha_EtuV'] = (-2, 1.2)

    fit_forw_Ht.limits['beta_HtuV'] = (0, 15)
    fit_forw_Ht.limits['beta_Htubar'] = (0, 15)
    fit_forw_Ht.limits['beta_HtdV'] = (0, 15)
    fit_forw_Ht.limits['beta_Htdbar'] = (0, 15)
    fit_forw_Ht.limits['beta_Htg'] = (0, 15)
    fit_forw_Ht.limits['beta_EtuV'] = (0, 15)

    fit_forw_Ht.fixed['alphap_Htqbar'] = True
    fit_forw_Ht.fixed['alphap_Htg'] = True
    
    fit_forw_Ht.fixed['Norm_EtuV'] = True
    fit_forw_Ht.fixed['alpha_EtuV'] = True
    fit_forw_Ht.fixed['beta_EtuV'] = True
    fit_forw_Ht.fixed['alphap_EtuV'] = True

    fit_forw_Ht.fixed['Norm_EtdV'] = True

    fit_forw_Ht.fixed['R_Et_Sea'] = True
    fit_forw_Ht.fixed['R_Htu_xi2'] = True
    fit_forw_Ht.fixed['R_Htd_xi2'] = True 
    fit_forw_Ht.fixed['R_Htg_xi2'] = True 
    fit_forw_Ht.fixed['R_Etu_xi2'] = True
    fit_forw_Ht.fixed['R_Etd_xi2'] = True
    fit_forw_Ht.fixed['R_Etg_xi2'] = True

    fit_forw_Ht.fixed['R_Htu_xi4'] = True
    fit_forw_Ht.fixed['R_Htd_xi4'] = True 
    fit_forw_Ht.fixed['R_Htg_xi4'] = True 
    fit_forw_Ht.fixed['R_Etu_xi4'] = True
    fit_forw_Ht.fixed['R_Etd_xi4'] = True
    fit_forw_Ht.fixed['R_Etg_xi4'] = True

    fit_forw_Ht.fixed['bexp_HtSea'] = True

    global time_start 
    time_start = time.time()
    
    fit_forw_Ht.migrad()
    fit_forw_Ht.hesse()

    ndof_Ht = len(PDF_data_Ht.index) + len(tPDF_data_Ht.index) + len(GFF_data_Ht.index)  - fit_forw_Ht.nfit

    time_end = time.time() -time_start    
    with open('GUMP_Output/Ht_forward_fit.txt', 'w') as f:
        print('Total running time: %3.1f minutes. Total call of cost function: %3d.\n' % ( time_end/60, fit_forw_Ht.nfcn), file=f)
        print('The chi squared/d.o.f. is: %3.1f / %3d ( = %3.1f ).\n' % (fit_forw_Ht.fval, ndof_Ht, fit_forw_Ht.fval/ndof_Ht), file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*fit_forw_Ht.values, sep=", ", file = f)
        print(*fit_forw_Ht.errors, sep=", ", file = f)
        print(fit_forw_Ht.params, file = f)
        print(fit_forw_Ht.covariance, file = f)
    print("Ht fit finished...")
    return fit_forw_Ht

def forward_Et_fit(Paralst_Pol):
    
    [Norm_HtuV_Init,   alpha_HtuV_Init,   beta_HtuV_Init,   alphap_HtuV_Init, 
     Norm_Htubar_Init, alpha_Htubar_Init, beta_Htubar_Init, alphap_Htqbar_Init,
     Norm_HtdV_Init,   alpha_HtdV_Init,   beta_HtdV_Init,   alphap_HtdV_Init,
     Norm_Htdbar_Init, alpha_Htdbar_Init, beta_Htdbar_Init, 
     Norm_Htg_Init,    alpha_Htg_Init,    beta_Htg_Init,    alphap_Htg_Init,
     Norm_EtuV_Init,   alpha_EtuV_Init,   beta_EtuV_Init,   alphap_EtuV_Init,
     Norm_EtdV_Init,   R_Et_Sea_Init,     R_Htu_xi2_Init,   R_Htd_xi2_Init,    R_Htg_xi2_Init,
     R_Etu_xi2_Init,   R_Etd_xi2_Init,    R_Etg_xi2_Init,
     R_Htu_xi4_Init,   R_Htd_xi4_Init,    R_Htg_xi4_Init,
     R_Etu_xi4_Init,   R_Etd_xi4_Init,    R_Etg_xi4_Init,   bexp_HtSea_Init] = Paralst_Pol

    fit_forw_Et = Minuit(cost_forward_Et, Norm_HtuV = Norm_HtuV_Init,     alpha_HtuV = alpha_HtuV_Init,      beta_HtuV = beta_HtuV_Init,     alphap_HtuV = alphap_HtuV_Init, 
                                          Norm_Htubar = Norm_Htubar_Init, alpha_Htubar = alpha_Htubar_Init,  beta_Htubar = beta_Htubar_Init, alphap_Htqbar = alphap_Htqbar_Init,
                                          Norm_HtdV = Norm_HtdV_Init,     alpha_HtdV = alpha_HtdV_Init,      beta_HtdV = beta_HtdV_Init,     alphap_HtdV = alphap_HtdV_Init,
                                          Norm_Htdbar = Norm_Htdbar_Init, alpha_Htdbar = alpha_Htdbar_Init,  beta_Htdbar = beta_Htdbar_Init, 
                                          Norm_Htg = Norm_Htg_Init,       alpha_Htg = alpha_Htg_Init,        beta_Htg = beta_Htg_Init,       alphap_Htg = alphap_Htg_Init,
                                          Norm_EtuV = Norm_EtuV_Init,     alpha_EtuV = alpha_EtuV_Init,      beta_EtuV = beta_EtuV_Init,     alphap_EtuV = alphap_EtuV_Init,
                                          Norm_EtdV = Norm_EtdV_Init,     R_Et_Sea = R_Et_Sea_Init,          R_Htu_xi2 = R_Htu_xi2_Init,     R_Htd_xi2 = R_Htd_xi2_Init,     R_Htg_xi2 = R_Htg_xi2_Init,
                                          R_Etu_xi2 = R_Etu_xi2_Init,     R_Etd_xi2 = R_Etd_xi2_Init,        R_Etg_xi2 = R_Etg_xi2_Init,
                                          R_Htu_xi4 = R_Htu_xi4_Init,     R_Htd_xi4 = R_Htd_xi4_Init,        R_Htg_xi4 = R_Htg_xi4_Init,
                                          R_Etu_xi4 = R_Etu_xi4_Init,     R_Etd_xi4 = R_Etd_xi4_Init,        R_Etg_xi4 = R_Etg_xi4_Init,     bexp_HtSea = bexp_HtSea_Init)
    fit_forw_Et.errordef = 1

    fit_forw_Et.limits['alpha_HtuV'] = (-2, 1.2)
    fit_forw_Et.limits['alpha_Htubar'] = (-2, 1.2)
    fit_forw_Et.limits['alpha_HtdV'] = (-2, 1.2)
    fit_forw_Et.limits['alpha_Htdbar'] = (-2, 1.2)
    fit_forw_Et.limits['alpha_Htg'] = (-2, 1.2)
    fit_forw_Et.limits['alpha_EtuV'] = (-2, 0.8)

    fit_forw_Et.limits['beta_HtuV'] = (0, 15)
    fit_forw_Et.limits['beta_Htubar'] = (0, 15)
    fit_forw_Et.limits['beta_HtdV'] = (0, 15)
    fit_forw_Et.limits['beta_Htdbar'] = (0, 15)
    fit_forw_Et.limits['beta_Htg'] = (0, 15)
    fit_forw_Et.limits['beta_EtuV'] = (0, 15)

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

    fit_forw_Et.fixed['R_Htu_xi2'] = True
    fit_forw_Et.fixed['R_Htd_xi2'] = True 
    fit_forw_Et.fixed['R_Htg_xi2'] = True 
    fit_forw_Et.fixed['R_Etu_xi2'] = True
    fit_forw_Et.fixed['R_Etd_xi2'] = True
    fit_forw_Et.fixed['R_Etg_xi2'] = True

    fit_forw_Et.fixed['R_Htu_xi4'] = True
    fit_forw_Et.fixed['R_Htd_xi4'] = True 
    fit_forw_Et.fixed['R_Htg_xi4'] = True 
    fit_forw_Et.fixed['R_Etu_xi4'] = True
    fit_forw_Et.fixed['R_Etd_xi4'] = True
    fit_forw_Et.fixed['R_Etg_xi4'] = True

    fit_forw_Et.fixed['bexp_HtSea'] = True

    global time_start
    time_start = time.time()
    
    fit_forw_Et.migrad()
    fit_forw_Et.hesse()

    ndof_Et = len(PDF_data_Et.index) + len(tPDF_data_Et.index) + len(GFF_data_Et.index)  - fit_forw_Et.nfit

    time_end = time.time() -time_start    
    with open('GUMP_Output/Et_forward_fit.txt', 'w') as f:
        print('Total running time: %3.1f minutes. Total call of cost function: %3d.\n' % ( time_end/60, fit_forw_Et.nfcn), file=f)
        print('The chi squared/d.o.f. is: %3.1f / %3d ( = %3.1f ).\n' % (fit_forw_Et.fval, ndof_Et, fit_forw_Et.fval/ndof_Et), file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*fit_forw_Et.values, sep=", ", file = f)
        print(*fit_forw_Et.errors, sep=", ", file = f)
        print(fit_forw_Et.params, file = f)
        print(fit_forw_Et.covariance, file = f)
    print("Et fit finished...")
    return fit_forw_Et

def cost_off_forward(Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
                     Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
                     Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
                     Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar, 
                     Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg,
                     Norm_EuV,    alpha_EuV,    beta_EuV,    alphap_EuV,
                     Norm_EdV,    R_E_Sea,      R_Hu_xi2,    R_Hd_xi2,    R_Hg_xi2,
                     R_Eu_xi2,    R_Ed_xi2,     R_Eg_xi2,
                     R_Hu_xi4,    R_Hd_xi4,     R_Hg_xi4,
                     R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea,
                     Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV, 
                     Norm_Htubar, alpha_Htubar, beta_Htubar, alphap_Htqbar,
                     Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,
                     Norm_Htdbar, alpha_Htdbar, beta_Htdbar, 
                     Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_Htg,
                     Norm_EtuV,   alpha_EtuV,   beta_EtuV,   alphap_EtuV,
                     Norm_EtdV,   R_Et_Sea,     R_Htu_xi2,   R_Htd_xi2,    R_Htg_xi2,
                     R_Etu_xi2,   R_Etd_xi2,    R_Etg_xi2,
                     R_Htu_xi4,   R_Htd_xi4,    R_Htg_xi4,
                     R_Etu_xi4,   R_Etd_xi4,    R_Etg_xi4,   bexp_HtSea):

    global Minuit_Counter, Time_Counter

    time_now = time.time() - time_start
    
    if(time_now > Time_Counter * 600):
        print('Runing Time:',round(time_now/60),'minutes. Cost function called total', Minuit_Counter, 'times.')
        Time_Counter = Time_Counter + 1
    
    Minuit_Counter = Minuit_Counter + 1
    Para_Unp_lst = [Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
                    Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
                    Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
                    Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar, 
                    Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg,
                    Norm_EuV,    alpha_EuV,    beta_EuV,    alphap_EuV,
                    Norm_EdV,    R_E_Sea,      R_Hu_xi2,    R_Hd_xi2,    R_Hg_xi2,
                    R_Eu_xi2,    R_Ed_xi2,     R_Eg_xi2,
                    R_Hu_xi4,    R_Hd_xi4,     R_Hg_xi4,
                    R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea]

    Para_Pol_lst = [Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV, 
                    Norm_Htubar, alpha_Htubar, beta_Htubar, alphap_Htqbar,
                    Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,
                    Norm_Htdbar, alpha_Htdbar, beta_Htdbar, 
                    Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_Htg,
                    Norm_EtuV,   alpha_EtuV,   beta_EtuV,   alphap_EtuV,
                    Norm_EtdV,   R_Et_Sea,     R_Htu_xi2,   R_Htd_xi2,    R_Htg_xi2,
                    R_Etu_xi2,   R_Etd_xi2,    R_Etg_xi2,
                    R_Htu_xi4,   R_Htd_xi4,    R_Htg_xi4,
                    R_Etu_xi4,   R_Etd_xi4,    R_Etg_xi4,   bexp_HtSea]
        
    Para_Unp_all = ParaManager_Unp(Para_Unp_lst)
    Para_Pol_all = ParaManager_Pol(Para_Pol_lst)

    cost_DVCS_xBtQ = np.array(list(pool.map(partial(DVCSxsec_cost_xBtQ, Para_Unp = Para_Unp_all, Para_Pol = Para_Pol_all), DVCSxsec_group_data)))
    cost_DVCSxsec = np.sum(cost_DVCS_xBtQ)

    # DVCS_HERA_pred = np.array(list(pool.map(partial(DVCSxsec_HERA_theo, Para_Unp = Para_Unp_all, Para_Pol = Para_Pol_all), np.array(DVCS_HERA_data))))
    DVCS_HERA_pred = DVCSxsec_HERA_theo(DVCS_HERA_data, Para_Unp=Para_Unp_all, Para_Pol=Para_Pol_all)
    cost_DVCS_HERA = np.sum(((DVCS_HERA_pred - DVCS_HERA_data['f'])/ DVCS_HERA_data['delta f']) ** 2 )

    return  cost_DVCSxsec + cost_DVCS_HERA

def cost_off_forward_test(Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
                     Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
                     Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
                     Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar, 
                     Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg,
                     Norm_EuV,    alpha_EuV,    beta_EuV,    alphap_EuV,
                     Norm_EdV,    R_E_Sea,      R_Hu_xi2,    R_Hd_xi2,    R_Hg_xi2,
                     R_Eu_xi2,    R_Ed_xi2,     R_Eg_xi2,
                     R_Hu_xi4,    R_Hd_xi4,     R_Hg_xi4,
                     R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea,
                     Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV, 
                     Norm_Htubar, alpha_Htubar, beta_Htubar, alphap_Htqbar,
                     Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,
                     Norm_Htdbar, alpha_Htdbar, beta_Htdbar, 
                     Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_Htg,
                     Norm_EtuV,   alpha_EtuV,   beta_EtuV,   alphap_EtuV,
                     Norm_EtdV,   R_Et_Sea,     R_Htu_xi2,   R_Htd_xi2,    R_Htg_xi2,
                     R_Etu_xi2,   R_Etd_xi2,    R_Etg_xi2,
                     R_Htu_xi4,   R_Htd_xi4,    R_Htg_xi4,
                     R_Etu_xi4,   R_Etd_xi4,    R_Etg_xi4,   bexp_HtSea):

    global Minuit_Counter, Time_Counter

    time_now = time.time() - time_start
    
    if(time_now > Time_Counter * 600):
        print('Runing Time:',round(time_now/60),'minutes. Cost function called total', Minuit_Counter, 'times.')
        Time_Counter = Time_Counter + 1
    
    Minuit_Counter = Minuit_Counter + 1
    Para_Unp_lst = [Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
                    Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
                    Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
                    Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar, 
                    Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg,
                    Norm_EuV,    alpha_EuV,    beta_EuV,    alphap_EuV,
                    Norm_EdV,    R_E_Sea,     R_Hu_xi2,     R_Hd_xi2,    R_Hg_xi2,
                    R_Eu_xi2,    R_Ed_xi2,     R_Eg_xi2,
                    R_Hu_xi4,    R_Hd_xi4,     R_Hg_xi4,
                    R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea]

    Para_Pol_lst = [Norm_HtuV,   alpha_HtuV,   beta_HtuV,   alphap_HtuV, 
                    Norm_Htubar, alpha_Htubar, beta_Htubar, alphap_Htqbar,
                    Norm_HtdV,   alpha_HtdV,   beta_HtdV,   alphap_HtdV,
                    Norm_Htdbar, alpha_Htdbar, beta_Htdbar, 
                    Norm_Htg,    alpha_Htg,    beta_Htg,    alphap_Htg,
                    Norm_EtuV,   alpha_EtuV,   beta_EtuV,   alphap_EtuV,
                    Norm_EtdV,   R_Et_Sea,     R_Htu_xi2,   R_Htd_xi2,    R_Htg_xi2,
                    R_Etu_xi2,   R_Etd_xi2,    R_Etg_xi2,
                    R_Htu_xi4,   R_Htd_xi4,    R_Htg_xi4,
                    R_Etu_xi4,   R_Etd_xi4,    R_Etg_xi4,   bexp_HtSea]
        
    Para_Unp_all = ParaManager_Unp(Para_Unp_lst)
    Para_Pol_all = ParaManager_Pol(Para_Pol_lst)

    cost_DVCS_xBtQ = np.array(list(pool.map(partial(DVCSxsec_cost_xBtQ, Para_Unp = Para_Unp_all, Para_Pol = Para_Pol_all), DVCSxsec_group_data)))
    cost_DVCSxsec = np.sum(cost_DVCS_xBtQ)

    # DVCS_HERA_pred = np.array(list(pool.map(partial(DVCSxsec_HERA_theo, Para_Unp = Para_Unp_all, Para_Pol = Para_Pol_all), np.array(DVCS_HERA_data))))
    DVCS_HERA_pred = DVCSxsec_HERA_theo(DVCS_HERA_data, Para_Unp=Para_Unp_all, Para_Pol=Para_Pol_all)
    cost_DVCS_HERA = np.sum(((DVCS_HERA_pred - DVCS_HERA_data['f'])/ DVCS_HERA_data['delta f']) ** 2 )

    return  cost_DVCSxsec, cost_DVCS_HERA

def off_forward_fit(Paralst_Unp, Paralst_Pol):

    [Norm_HuV_Init,    alpha_HuV_Init,    beta_HuV_Init,    alphap_HuV_Init, 
     Norm_Hubar_Init,  alpha_Hubar_Init,  beta_Hubar_Init,  alphap_Hqbar_Init,
     Norm_HdV_Init,    alpha_HdV_Init,    beta_HdV_Init,    alphap_HdV_Init,
     Norm_Hdbar_Init,  alpha_Hdbar_Init,  beta_Hdbar_Init, 
     Norm_Hg_Init,     alpha_Hg_Init,     beta_Hg_Init,     alphap_Hg_Init,
     Norm_EuV_Init,    alpha_EuV_Init,    beta_EuV_Init,    alphap_EuV_Init,
     Norm_EdV_Init,    R_E_Sea_Init,      R_Hu_xi2_Init,    R_Hd_xi2_Init,    R_Hg_xi2_Init,
     R_Eu_xi2_Init,    R_Ed_xi2_Init,     R_Eg_xi2_Init,
     R_Hu_xi4_Init,    R_Hd_xi4_Init,     R_Hg_xi4_Init,
     R_Eu_xi4_Init,    R_Ed_xi4_Init,     R_Eg_xi4_Init,    bexp_HSea_Init] = Paralst_Unp

    [Norm_HtuV_Init,   alpha_HtuV_Init,   beta_HtuV_Init,   alphap_HtuV_Init, 
     Norm_Htubar_Init, alpha_Htubar_Init, beta_Htubar_Init, alphap_Htqbar_Init,
     Norm_HtdV_Init,   alpha_HtdV_Init,   beta_HtdV_Init,   alphap_HtdV_Init,
     Norm_Htdbar_Init, alpha_Htdbar_Init, beta_Htdbar_Init, 
     Norm_Htg_Init,    alpha_Htg_Init,    beta_Htg_Init,    alphap_Htg_Init,
     Norm_EtuV_Init,   alpha_EtuV_Init,   beta_EtuV_Init,   alphap_EtuV_Init,
     Norm_EtdV_Init,   R_Et_Sea_Init,     R_Htu_xi2_Init,   R_Htd_xi2_Init,    R_Htg_xi2_Init,
     R_Etu_xi2_Init,   R_Etd_xi2_Init,    R_Etg_xi2_Init,
     R_Htu_xi4_Init,   R_Htd_xi4_Init,    R_Htg_xi4_Init,
     R_Etu_xi4_Init,   R_Etd_xi4_Init,    R_Etg_xi4_Init,   bexp_HtSea_Init] = Paralst_Pol

    fit_off_forward = Minuit(cost_off_forward, Norm_HuV = Norm_HuV_Init,     alpha_HuV = alpha_HuV_Init,      beta_HuV = beta_HuV_Init,     alphap_HuV = alphap_HuV_Init, 
                                               Norm_Hubar = Norm_Hubar_Init, alpha_Hubar = alpha_Hubar_Init,  beta_Hubar = beta_Hubar_Init, alphap_Hqbar = alphap_Hqbar_Init,
                                               Norm_HdV = Norm_HdV_Init,     alpha_HdV = alpha_HdV_Init,      beta_HdV = beta_HdV_Init,     alphap_HdV = alphap_HdV_Init,
                                               Norm_Hdbar = Norm_Hdbar_Init, alpha_Hdbar = alpha_Hdbar_Init,  beta_Hdbar = beta_Hdbar_Init, 
                                               Norm_Hg = Norm_Hg_Init,       alpha_Hg = alpha_Hg_Init,        beta_Hg = beta_Hg_Init,       alphap_Hg = alphap_Hg_Init,
                                               Norm_EuV = Norm_EuV_Init,     alpha_EuV = alpha_EuV_Init,      beta_EuV = beta_EuV_Init,     alphap_EuV = alphap_EuV_Init, 
                                               Norm_EdV = Norm_EdV_Init,     R_E_Sea = R_E_Sea_Init,          R_Hu_xi2 = R_Hu_xi2_Init,     R_Hd_xi2 = R_Hd_xi2_Init,     R_Hg_xi2 = R_Hg_xi2_Init,
                                               R_Eu_xi2 = R_Eu_xi2_Init,     R_Ed_xi2 = R_Ed_xi2_Init,        R_Eg_xi2 = R_Eg_xi2_Init,
                                               R_Hu_xi4 = R_Hu_xi4_Init,     R_Hd_xi4 = R_Hd_xi4_Init,        R_Hg_xi4 = R_Hg_xi4_Init,
                                               R_Eu_xi4 = R_Eu_xi4_Init,     R_Ed_xi4 = R_Ed_xi4_Init,        R_Eg_xi4 = R_Eg_xi4_Init,     bexp_HSea = bexp_HSea_Init,
                                               Norm_HtuV = Norm_HtuV_Init,     alpha_HtuV = alpha_HtuV_Init,      beta_HtuV = beta_HtuV_Init,     alphap_HtuV = alphap_HtuV_Init, 
                                               Norm_Htubar = Norm_Htubar_Init, alpha_Htubar = alpha_Htubar_Init,  beta_Htubar = beta_Htubar_Init, alphap_Htqbar = alphap_Htqbar_Init,
                                               Norm_HtdV = Norm_HtdV_Init,     alpha_HtdV = alpha_HtdV_Init,      beta_HtdV = beta_HtdV_Init,     alphap_HtdV = alphap_HtdV_Init,
                                               Norm_Htdbar = Norm_Htdbar_Init, alpha_Htdbar = alpha_Htdbar_Init,  beta_Htdbar = beta_Htdbar_Init, 
                                               Norm_Htg = Norm_Htg_Init,       alpha_Htg = alpha_Htg_Init,        beta_Htg = beta_Htg_Init,       alphap_Htg = alphap_Htg_Init,
                                               Norm_EtuV = Norm_EtuV_Init,     alpha_EtuV = alpha_EtuV_Init,      beta_EtuV = beta_EtuV_Init,     alphap_EtuV = alphap_EtuV_Init,
                                               Norm_EtdV = Norm_EtdV_Init,     R_Et_Sea = R_Et_Sea_Init,          R_Htu_xi2 = R_Htu_xi2_Init,     R_Htd_xi2 = R_Htd_xi2_Init,     R_Htg_xi2 = R_Htg_xi2_Init,
                                               R_Etu_xi2 = R_Etu_xi2_Init,     R_Etd_xi2 = R_Etd_xi2_Init,        R_Etg_xi2 = R_Etg_xi2_Init,
                                               R_Htu_xi4 = R_Htu_xi4_Init,     R_Htd_xi4 = R_Htd_xi4_Init,        R_Htg_xi4 = R_Htg_xi4_Init,
                                               R_Etu_xi4 = R_Etu_xi4_Init,     R_Etd_xi4 = R_Etd_xi4_Init,        R_Etg_xi4 = R_Etg_xi4_Init,     bexp_HtSea = bexp_HtSea_Init)
    fit_off_forward.errordef = 1

    fit_off_forward.limits['bexp_HSea']  = (0, 10)
    fit_off_forward.limits['bexp_HtSea'] = (0, 10)

    fit_off_forward.fixed['Norm_HuV'] = True
    fit_off_forward.fixed['alpha_HuV'] = True
    fit_off_forward.fixed['beta_HuV'] = True
    fit_off_forward.fixed['alphap_HuV'] = True

    fit_off_forward.fixed['Norm_Hubar'] = True
    fit_off_forward.fixed['alpha_Hubar'] = True
    fit_off_forward.fixed['beta_Hubar'] = True

    fit_off_forward.fixed['alphap_Hqbar'] = True

    fit_off_forward.fixed['Norm_HdV'] = True
    fit_off_forward.fixed['alpha_HdV'] = True
    fit_off_forward.fixed['beta_HdV'] = True
    fit_off_forward.fixed['alphap_HdV'] = True

    fit_off_forward.fixed['Norm_Hdbar'] = True
    fit_off_forward.fixed['alpha_Hdbar'] = True
    fit_off_forward.fixed['beta_Hdbar'] = True

    fit_off_forward.fixed['Norm_Hg'] = True
    fit_off_forward.fixed['alpha_Hg'] = True
    fit_off_forward.fixed['beta_Hg'] = True

    fit_off_forward.fixed['Norm_EuV'] = True
    fit_off_forward.fixed['alpha_EuV'] = True
    fit_off_forward.fixed['beta_EuV'] = True
    fit_off_forward.fixed['alphap_EuV'] = True

    fit_off_forward.fixed['Norm_EdV'] = True

    fit_off_forward.fixed['Norm_HtuV'] = True
    fit_off_forward.fixed['alpha_HtuV'] = True
    fit_off_forward.fixed['beta_HtuV'] = True
    fit_off_forward.fixed['alphap_HtuV'] = True

    fit_off_forward.fixed['Norm_Htubar'] = True
    fit_off_forward.fixed['alpha_Htubar'] = True
    fit_off_forward.fixed['beta_Htubar'] = True

    fit_off_forward.fixed['alphap_Htqbar'] = True

    fit_off_forward.fixed['Norm_HtdV'] = True
    fit_off_forward.fixed['alpha_HtdV'] = True
    fit_off_forward.fixed['beta_HtdV'] = True
    fit_off_forward.fixed['alphap_HtdV'] = True

    fit_off_forward.fixed['Norm_Htdbar'] = True
    fit_off_forward.fixed['alpha_Htdbar'] = True
    fit_off_forward.fixed['beta_Htdbar'] = True

    fit_off_forward.fixed['Norm_Htg'] = True
    fit_off_forward.fixed['alpha_Htg'] = True
    fit_off_forward.fixed['beta_Htg'] = True

    fit_off_forward.fixed['Norm_EtuV'] = True
    fit_off_forward.fixed['alpha_EtuV'] = True
    fit_off_forward.fixed['beta_EtuV'] = True
    fit_off_forward.fixed['alphap_EtuV'] = True

    fit_off_forward.fixed['Norm_EtdV'] = True

    fit_off_forward.fixed['alphap_Hg'] = True
    fit_off_forward.fixed['alphap_Htg'] = True

    fit_off_forward.fixed['R_Hg_xi2'] = True
    fit_off_forward.fixed['R_Eg_xi2'] = True
    fit_off_forward.fixed['R_Htg_xi2'] = True
    fit_off_forward.fixed['R_Etg_xi2'] = True

    fit_off_forward.fixed['R_Hg_xi4'] = True
    fit_off_forward.fixed['R_Eg_xi4'] = True
    fit_off_forward.fixed['R_Htg_xi4'] = True
    fit_off_forward.fixed['R_Etg_xi4'] = True

    """
    fit_off_forward.fixed['R_Hu_xi4'] = True
    fit_off_forward.fixed['R_Hd_xi4'] = True 
    fit_off_forward.fixed['R_Eu_xi4'] = True
    fit_off_forward.fixed['R_Ed_xi4'] = True

    fit_off_forward.fixed['R_Htu_xi4'] = True
    fit_off_forward.fixed['R_Htd_xi4'] = True 
    fit_off_forward.fixed['R_Etu_xi4'] = True
    fit_off_forward.fixed['R_Etd_xi4'] = True
    """

    global Minuit_Counter, Time_Counter, time_start
    Minuit_Counter = 0
    Time_Counter = 1
    time_start = time.time()
    
    fit_off_forward.migrad()
    fit_off_forward.hesse()

    ndof_off_forward = len(DVCSxsec_data.index) + len(DVCS_HERA_data.index)  - fit_off_forward.nfit 

    time_end = time.time() -time_start

    with open('GUMP_Output/off_forward_fit.txt', 'w') as f:
        print('Total running time: %3.1f minutes. Total call of cost function: %3d.\n' % ( time_end/60, fit_off_forward.nfcn), file=f)
        print('The chi squared/d.o.f. is: %3.1f / %3d ( = %3.1f ).\n' % (fit_off_forward.fval, ndof_off_forward, fit_off_forward.fval/ndof_off_forward), file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*fit_off_forward.values, sep=", ", file = f)
        print(*fit_off_forward.errors, sep=", ", file = f)
        print(fit_off_forward.params, file = f)
        print(fit_off_forward.covariance, file = f)
    print("off forward fit finished...")
    return fit_off_forward

if __name__ == '__main__':
    pool = Pool()
    time_start = time.time()

    Paralst_Unp     = [4.923245780014504, 0.21630849765192428, 3.22890722499297, 2.3489354011979526, 0.16346585165260322, 1.135711231720756, 6.897907293004202, 0.15, 3.3570431064205124, 0.18430950949075475, 4.4164330846199, 3.474060988244228, 0.24911513048067477, 1.051977198682268, 6.546615550866641, 2.8639557573930126, 1.052330303667227, 7.412339020128528, 0.15, 0.16051495538648822, 0.9163110652883968, 1.0193962830435552, 0.41274932784170376, -0.1977991308710107, -1.2564920365188517, 1.067457804938632, -19.770730039892825, 1.0, 10.069595476994552, -3.2467026195137683, 1.0, -0.32089849276114984, 5.693312161571689, 0.0, -2.926022924238427, 0.6575758917146275, 0.0, 3.7811389938079354]
    Paralst_Pol     = [4.523625044880523, -0.24625833924504925, 3.0349443428078073, 2.6130504668028984, 0.07655463228254533, 0.5156744483888329, 4.371557716864458, 0.15, -0.7111561056489031, 0.2110536573461057, 3.2394039683470357, 4.37109681578387, -0.05648412944234933, 0.6133679635068163, 2.0897772013914437, 0.24322592280875213, 0.6308477000598409, 2.7182293377476165, 0.15, 8.93993973044334, 0.7999950847932062, 7.329257777405136, 2.043840761349927, -3.531387135862607, -1.267179457495525, 5.110641851061494, 42.0289770774691, 1.0, 2.092997157826637, -2.1779671813414163, 1.0, -1.5931689979366248, -12.338040498859877, 0.0, 2.6588214837192625, 36.25529526389492, 0.0, 2.805380264758739e-06]

    fit_forward_H   = forward_H_fit(Paralst_Unp)
    Paralst_Unp     = np.array(fit_forward_H.values)

    fit_forward_Ht  = forward_Ht_fit(Paralst_Pol)
    Paralst_Pol     = np.array(fit_forward_Ht.values)

    fit_forward_E   = forward_E_fit(Paralst_Unp)
    Paralst_Unp     = np.array(fit_forward_E.values)

    fit_forward_Et  = forward_Et_fit(Paralst_Pol)
    Paralst_Pol     = np.array(fit_forward_Et.values)

    fit_off_forward = off_forward_fit(Paralst_Unp, Paralst_Pol)

    Para_Unp_All    = ParaManager_Unp(Paralst_Unp)
    Para_Pol_All    = ParaManager_Pol(Paralst_Pol)

    #print(cost_off_forward_test(4.923245780014504, 0.21630849765192428, 3.22890722499297, 2.3489354011979526, 0.16346585165260322, 1.135711231720756, 6.897907293004202, 0.15, 3.3570431064205124, 0.18430950949075475, 4.4164330846199, 3.474060988244228, 0.24911513048067477, 1.051977198682268, 6.546615550866641, 2.8639557573930126, 1.052330303667227, 7.412339020128528, 0.15, 0.16051495538648822, 0.9163110652883968, 1.0193962830435552, 0.41274932784170376, -0.1977991308710107, -1.2564920365188517, 1.067457804938632, -19.770730039892825, 1.0, 10.069595476994552, -3.2467026195137683, 1.0, -0.32089849276114984, 5.693312161571689, 0.0, -2.926022924238427, 0.6575758917146275, 0.0, 3.7811389938079354, 4.523625044880523, -0.24625833924504925, 3.0349443428078073, 2.6130504668028984, 0.07655463228254533, 0.5156744483888329, 4.371557716864458, 0.15, -0.7111561056489031, 0.2110536573461057, 3.2394039683470357, 4.37109681578387, -0.05648412944234933, 0.6133679635068163, 2.0897772013914437, 0.24322592280875213, 0.6308477000598409, 2.7182293377476165, 0.15, 8.93993973044334, 0.7999950847932062, 7.329257777405136, 2.043840761349927, -3.531387135862607, -1.267179457495525, 5.110641851061494, 42.0289770774691, 1.0, 2.092997157826637, -2.1779671813414163, 1.0, -1.5931689979366248, -12.338040498859877, 0.0, 2.6588214837192625, 36.25529526389492, 0.0, 2.805380264758739e-06))