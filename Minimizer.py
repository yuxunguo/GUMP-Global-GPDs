from Parameters import ParaManager_Unp, ParaManager_Pol
from Observables import GPDobserv
from DVCS_xsec import dsigma_TOT, dsigma_DVCS_HERA, M
from DVMP_xsec import dsigma_dt, M_jpsi
from multiprocessing import Pool
from functools import partial
from iminuit import Minuit
import numpy as np
import pandas as pd
import time
import csv
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

Minuit_Counter = 0

Time_Counter = 1

Q_threshold = 1.9

xB_Cut = 0.5

xB_small_Cut = 0.0001

PDF_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/PDFdata.csv'),       header = None, names = ['x', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'x': float, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})
PDF_data_H  = PDF_data[PDF_data['spe'] == 0]
PDF_data_H_g = PDF_data_H[PDF_data_H['flv'] == 'g']
PDF_data_H_g = PDF_data_H_g[PDF_data_H_g['x'] < xB_Cut]
PDF_data_E  = PDF_data[PDF_data['spe'] == 1]
PDF_data_Ht = PDF_data[PDF_data['spe'] == 2]
PDF_data_Et = PDF_data[PDF_data['spe'] == 3]

PDF_smallx_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/PDFg_smallx.csv'),       header = None, names = ['x', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'x': float, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})
PDF_H_smallx_data = PDF_smallx_data[PDF_smallx_data['spe']==0]
PDFg_smallx_data = PDF_H_smallx_data[PDF_H_smallx_data['flv']=='g']
PDFg_smallx_data = PDF_H_smallx_data[PDF_H_smallx_data['x'] > xB_small_Cut]

tPDF_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/tPDFdata.csv'),     header = None, names = ['x', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'x': float, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})
tPDF_data_H  = tPDF_data[tPDF_data['spe'] == 0]
tPDF_data_H_g = tPDF_data_H[tPDF_data_H['flv'] == 'g']
tPDF_data_H_g = tPDF_data_H[tPDF_data_H['x'] < xB_small_Cut]
tPDF_data_E  = tPDF_data[tPDF_data['spe'] == 1]
tPDF_data_Ht = tPDF_data[tPDF_data['spe'] == 2]
tPDF_data_Et = tPDF_data[tPDF_data['spe'] == 3]

GFF_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/GFFdata_Quark.csv'),       header = None, names = ['j', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'j': int, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})
GFF_data_H  = GFF_data[GFF_data['spe'] == 0]
GFF_data_E  = GFF_data[GFF_data['spe'] == 1]
GFF_data_Ht = GFF_data[GFF_data['spe'] == 2]
GFF_data_Et = GFF_data[GFF_data['spe'] == 3]

GFF_Gluon_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/GFFdata_Gluon.csv'),       header = None, names = ['j', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'j': int, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})
GFF_Gluon_data_H  = GFF_Gluon_data[GFF_Gluon_data['spe'] == 0]
GFF_Gluon_data_E  = GFF_Gluon_data[GFF_Gluon_data['spe'] == 1]
GFF_Gluon_data_Ht = GFF_Gluon_data[GFF_Gluon_data['spe'] == 2]
GFF_Gluon_data_Et = GFF_Gluon_data[GFF_Gluon_data['spe'] == 3]

DVCSxsec_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVCSxsec.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'phi', 'f', 'delta f', 'pol'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'phi': float, 'f': float, 'delta f': float, 'pol': str})
DVCSxsec_data_invalid = DVCSxsec_data[DVCSxsec_data['t']*(DVCSxsec_data['xB']-1) - M ** 2 * DVCSxsec_data['xB'] ** 2 < 0]
DVCSxsec_data = DVCSxsec_data[(DVCSxsec_data['Q'] > Q_threshold) & (DVCSxsec_data['xB'] < xB_Cut) & (DVCSxsec_data['t']*(DVCSxsec_data['xB']-1) - M ** 2 * DVCSxsec_data['xB'] ** 2 > 0)]
xBtQlst = DVCSxsec_data.drop_duplicates(subset = ['xB', 't', 'Q'], keep = 'first')[['xB','t','Q']].values.tolist()
DVCSxsec_group_data = list(map(lambda set: DVCSxsec_data[(DVCSxsec_data['xB'] == set[0]) & (DVCSxsec_data['t'] == set[1]) & ((DVCSxsec_data['Q'] == set[2]))], xBtQlst))

DVCSxsec_HERA_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVCSxsec_HERA.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f', 'pol'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float, 'pol': str})
DVCSxsec_HERA_data_invalid = DVCSxsec_HERA_data[DVCSxsec_HERA_data['t']*(DVCSxsec_HERA_data['xB']-1) - M ** 2 * DVCSxsec_HERA_data['xB'] ** 2 < 0]
DVCSxsec_HERA_data = DVCSxsec_HERA_data[(DVCSxsec_HERA_data['Q'] > Q_threshold) & (DVCSxsec_HERA_data['xB'] < xB_Cut) & (DVCSxsec_HERA_data['t']*(DVCSxsec_HERA_data['xB']-1) - M ** 2 * DVCSxsec_HERA_data['xB'] ** 2 > 0)]
xBtQlst_HERA = DVCSxsec_HERA_data.drop_duplicates(subset = ['xB', 't', 'Q'], keep = 'first')[['xB','t','Q']].values.tolist()
DVCSxsec_HERA_group_data = list(map(lambda set: DVCSxsec_HERA_data[(DVCSxsec_HERA_data['xB'] == set[0]) & (DVCSxsec_HERA_data['t'] == set[1]) & ((DVCSxsec_HERA_data['Q'] == set[2]))], xBtQlst_HERA))



# rho and phi data from HERA, R = sigma_L / sigma_T is currently handled on the theory side for these

DVrhoPZEUSxsec_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/DVrhoPZEUSdt.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float})
DVrhoPZEUSxsec_data['Q'] = np.sqrt(DVrhoPZEUSxsec_data['Q'])
DVrhoPZEUSxsec_data['t'] = -1 * DVrhoPZEUSxsec_data['t']
DVrhoPZEUSxsec_data = DVrhoPZEUSxsec_data[(DVrhoPZEUSxsec_data['Q']>Q_threshold)]
xBtQlst_rhoZ = DVrhoPZEUSxsec_data.drop_duplicates(subset = ['xB', 't', 'Q'], keep = 'first')[['xB','t','Q']].values.tolist()
DVrhoPZEUSxsec_group_data = list(map(lambda set: DVrhoPZEUSxsec_data[(DVrhoPZEUSxsec_data['xB'] == set[0]) & (DVrhoPZEUSxsec_data['t'] == set[1]) & ((DVrhoPZEUSxsec_data['Q'] == set[2]))], xBtQlst_rhoZ))

DVrhoPH1xsec_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/DVrhoPH1dt.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float})
DVrhoPH1xsec_data['Q'] = np.sqrt(DVrhoPH1xsec_data['Q'])
DVrhoPH1xsec_data['t'] = -1 * DVrhoPH1xsec_data['t']
DVrhoPH1xsec_data = DVrhoPH1xsec_data[(DVrhoPH1xsec_data['Q']>Q_threshold)]
xBtQlst_rhoH = DVrhoPH1xsec_data.drop_duplicates(subset = ['xB', 't', 'Q'], keep = 'first')[['xB','t','Q']].values.tolist()
DVrhoPH1xsec_group_data = list(map(lambda set: DVrhoPH1xsec_data[(DVrhoPH1xsec_data['xB'] == set[0]) & (DVrhoPH1xsec_data['t'] == set[1]) & ((DVrhoPH1xsec_data['Q'] == set[2]))], xBtQlst_rhoH))

DVphiPZEUSxsec_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/DVphiPZEUSdt.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float})
DVphiPZEUSxsec_data['Q'] = np.sqrt(DVphiPZEUSxsec_data['Q'])
DVphiPZEUSxsec_data['t'] = -1 * DVphiPZEUSxsec_data['t']
DVphiPZEUSxsec_data = DVphiPZEUSxsec_data[(DVphiPZEUSxsec_data['Q']>Q_threshold)]
xBtQlst_phiZ = DVphiPZEUSxsec_data.drop_duplicates(subset = ['xB', 't', 'Q'], keep = 'first')[['xB','t','Q']].values.tolist()
DVphiPZEUSxsec_group_data = list(map(lambda set: DVphiPZEUSxsec_data[(DVphiPZEUSxsec_data['xB'] == set[0]) & (DVphiPZEUSxsec_data['t'] == set[1]) & ((DVphiPZEUSxsec_data['Q'] == set[2]))], xBtQlst_phiZ))

DVphiPH1xsec_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/DVphiPH1dt.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float})
DVphiPH1xsec_data['Q'] = np.sqrt(DVphiPH1xsec_data['Q'])
DVphiPH1xsec_data['t'] = -1 * DVphiPH1xsec_data['t']
DVphiPH1xsec_data = DVphiPH1xsec_data[(DVphiPH1xsec_data['Q']>Q_threshold)]
xBtQlst_phiH = DVphiPH1xsec_data.drop_duplicates(subset = ['xB', 't', 'Q'], keep = 'first')[['xB','t','Q']].values.tolist()
DVphiPH1xsec_group_data = list(map(lambda set: DVphiPH1xsec_data[(DVphiPH1xsec_data['xB'] == set[0]) & (DVphiPH1xsec_data['t'] == set[1]) & ((DVphiPH1xsec_data['Q'] == set[2]))], xBtQlst_phiH))


# Jpsi data from HERA as well as R = sigma_L / sigma_T values hardcoded and used to convert data xsec simga_tot to sigma_L

DVJpsiPH1xsec_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/DVJpsiPH1dt_w_mass.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float})
DVJpsiPH1xsec_data['Q'] = np.sqrt(DVJpsiPH1xsec_data['Q'])
DVJpsiPH1xsec_data['t'] = -1 * DVJpsiPH1xsec_data['t']
DVJpsiPH1xsec_data = DVJpsiPH1xsec_data[(DVJpsiPH1xsec_data['Q']>Q_threshold)]
xBtQlst_JpsiH = DVJpsiPH1xsec_data.drop_duplicates(subset = ['xB', 't', 'Q'], keep = 'first')[['xB','t','Q']].values.tolist()
DVJpsiPH1xsec_L_data = DVJpsiPH1xsec_data.copy()
R_H1 = DVJpsiPH1xsec_L_data['f'].copy()
R_H1[(DVJpsiPH1xsec_L_data['Q']**2 > 1) & (DVJpsiPH1xsec_L_data['Q']**2 < 5)] = 0.052
R_H1[(DVJpsiPH1xsec_L_data['Q']**2 > 5) & (DVJpsiPH1xsec_L_data['Q']**2 < 10)] = 0.23
R_H1[(DVJpsiPH1xsec_L_data['Q']**2 > 10)] = 0.62
R_H1_err = DVJpsiPH1xsec_L_data['f'].copy()
R_H1_err[(DVJpsiPH1xsec_L_data['Q']**2 > 2) & (DVJpsiPH1xsec_L_data['Q']**2 < 5)] = 0.113
R_H1_err[(DVJpsiPH1xsec_L_data['Q']**2 > 5) & (DVJpsiPH1xsec_L_data['Q']**2 < 10)] = 0.27
R_H1_err[(DVJpsiPH1xsec_L_data['Q']**2 > 10)] = 0.61
DVJpsiPH1xsec_L_data['f'] = DVJpsiPH1xsec_L_data['f'] / ((1 - DVJpsiPH1xsec_L_data['y']) / (1 - DVJpsiPH1xsec_L_data['y'] - DVJpsiPH1xsec_L_data['y']**2 / 2) + (1 / R_H1))
DVJpsiPH1xsec_L_data['delta f'] = np.sqrt((DVJpsiPH1xsec_L_data['delta f'] / ((1 - DVJpsiPH1xsec_L_data['y']) / (1 - DVJpsiPH1xsec_L_data['y'] - DVJpsiPH1xsec_L_data['y']**2 / 2) + (1 / R_H1)))**2 + (R_H1_err * DVJpsiPH1xsec_L_data['f'] / (1 + (1 - DVJpsiPH1xsec_L_data['y']) / (1 - DVJpsiPH1xsec_L_data['y'] - DVJpsiPH1xsec_L_data['y']**2 / 2) * R_H1)**2)**2)
DVJpsiPH1xsec_group_data = list(map(lambda set: DVJpsiPH1xsec_data[(DVJpsiPH1xsec_data['xB'] == set[0]) & (DVJpsiPH1xsec_data['t'] == set[1]) & ((DVJpsiPH1xsec_data['Q'] == set[2]))], xBtQlst_JpsiH))
DVJpsiPH1xsec_L_group_data = list(map(lambda set: DVJpsiPH1xsec_L_data[(DVJpsiPH1xsec_data['xB'] == set[0]) & (DVJpsiPH1xsec_L_data['t'] == set[1]) & ((DVJpsiPH1xsec_L_data['Q'] == set[2]))], xBtQlst_JpsiH))

DVJpsiPZEUSxsec_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/DVJpsiPZEUSdt_w_mass.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float})
DVJpsiPZEUSxsec_data['Q'] = np.sqrt(DVJpsiPZEUSxsec_data['Q'])
DVJpsiPZEUSxsec_data['t'] = -1 * DVJpsiPZEUSxsec_data['t']
DVJpsiPZEUSxsec_data = DVJpsiPZEUSxsec_data[(DVJpsiPZEUSxsec_data['Q']>Q_threshold)]
xBtQlst_JpsiZ = DVJpsiPZEUSxsec_data.drop_duplicates(subset = ['xB', 't', 'Q'], keep = 'first')[['xB','t','Q']].values.tolist()
DVJpsiPZEUSxsec_L_data = DVJpsiPZEUSxsec_data.copy()
R_ZEUS = DVJpsiPZEUSxsec_L_data['f'].copy()
R_ZEUS[(DVJpsiPZEUSxsec_L_data['Q']**2 > 2) & (DVJpsiPZEUSxsec_L_data['Q']**2 < 5)] = 0.13
R_ZEUS[(DVJpsiPZEUSxsec_L_data['Q']**2 > 5) & (DVJpsiPZEUSxsec_L_data['Q']**2 < 10)] = 0.33
R_ZEUS[(DVJpsiPZEUSxsec_L_data['Q']**2 > 10)] = 1.19
R_ZEUS_err = DVJpsiPZEUSxsec_L_data['f'].copy()
R_ZEUS_err[(DVJpsiPZEUSxsec_L_data['Q']**2 > 2) & (DVJpsiPZEUSxsec_L_data['Q']**2 < 5)] = 0.19
R_ZEUS_err[(DVJpsiPZEUSxsec_L_data['Q']**2 > 5) & (DVJpsiPZEUSxsec_L_data['Q']**2 < 10)] = 0.25
R_ZEUS_err[(DVJpsiPZEUSxsec_L_data['Q']**2 > 10)] = 0.58
DVJpsiPZEUSxsec_L_data['f'] = DVJpsiPZEUSxsec_L_data['f'] / ((1 - DVJpsiPZEUSxsec_L_data['y']) / (1 - DVJpsiPZEUSxsec_L_data['y'] - DVJpsiPZEUSxsec_L_data['y']**2 / 2) + (1 / R_ZEUS))
DVJpsiPZEUSxsec_L_data['delta f'] = np.sqrt((DVJpsiPZEUSxsec_L_data['delta f'] / ((1 - DVJpsiPZEUSxsec_L_data['y']) / (1 - DVJpsiPZEUSxsec_L_data['y'] - DVJpsiPZEUSxsec_L_data['y']**2 / 2) + (1 / R_ZEUS)))**2 + (R_ZEUS_err * DVJpsiPZEUSxsec_L_data['f'] / (1 + (1 - DVJpsiPZEUSxsec_L_data['y']) / (1 - DVJpsiPZEUSxsec_L_data['y'] - DVJpsiPZEUSxsec_L_data['y']**2 / 2) * R_ZEUS)**2)**2)
DVJpsiPZEUSxsec_group_data = list(map(lambda set: DVJpsiPZEUSxsec_data[(DVJpsiPZEUSxsec_data['xB'] == set[0]) & (DVJpsiPZEUSxsec_data['t'] == set[1]) & ((DVJpsiPZEUSxsec_data['Q'] == set[2]))], xBtQlst_JpsiZ))
DVJpsiPZEUSxsec_L_group_data = list(map(lambda set: DVJpsiPZEUSxsec_L_data[(DVJpsiPZEUSxsec_data['xB'] == set[0]) & (DVJpsiPZEUSxsec_L_data['t'] == set[1]) & ((DVJpsiPZEUSxsec_L_data['Q'] == set[2]))], xBtQlst_JpsiZ))

'''
JpsiphotoH1xsec_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/DVJpsiPZEUSdt_w_mass.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float})
JpsiphotoH1xsec_data['Q'] = np.sqrt(JpsiphotoH1xsec_data['Q'])
JpsiphotoH1xsec_data['t'] = -1 * JpsiphotoH1xsec_data['t']
JpsiphotoH1xsec_data = JpsiphotoH1xsec_data[(JpsiphotoH1xsec_data['Q']>Q_threshold)]
xBtQlst_JpsiphotoH1 = JpsiphotoH1xsec_data.drop_duplicates(subset = ['xB', 't', 'Q'], keep = 'first')[['xB','t','Q']].values.tolist()
'''

# Helper function for scalar computation
def PDF_theo_scalar_helper(args):
    x_i, xi_i, t_i, Q_i, p_i, flv_i, Para_i, p_order = args
    _PDF_theo = GPDobserv(x_i, xi_i, t_i, Q_i, p_i)  
    return _PDF_theo.tPDF(flv_i, Para_i, p_order)

# Helper function for scalar computation
def GFF_theo_scalar_helper(args):
    j_i, x, xi, t_i, Q_i, p_i, flv_i, Para_i, p_order = args
    _GFF_theo = GPDobserv(x, xi, t_i, Q_i, p_i)  # Assume GPDobserv is defined elsewhere
    return _GFF_theo.GFFj0(j_i, flv_i, Para_i, p_order)

def PDF_theo(PDF_input: pd.DataFrame, Para: np.array, p_order = 1):
    xs = PDF_input['x'].to_numpy()
    ts = PDF_input['t'].to_numpy()
    Qs = PDF_input['Q'].to_numpy()
    flvs = PDF_input['flv'].to_numpy()
    spes = PDF_input['spe'].to_numpy()
    
    xis = np.zeros_like(xs)
    ps = np.where(spes <= 1, 1, -1)
    spes = np.where(spes <= 1, spes, spes - 2)
    Para_spe = Para[spes]

    # Prepare input arguments for parallel computation
    args = [(x_i, xi_i, t_i, Q_i, p_i, flv_i, Para_i, p_order) 
            for x_i, xi_i, t_i, Q_i, p_i, flv_i, Para_i 
            in zip(xs, xis, ts, Qs, ps, flvs, Para_spe)]
    
    # Use multiprocessing Pool to parallelize the computation
    result = pool.map(PDF_theo_scalar_helper, args)
    
    return np.array(result)

tPDF_theo = PDF_theo

def GFF_theo(GFF_input: pd.DataFrame, Para: np.array, p_order = 1):
    
    js = GFF_input['j'].to_numpy()
    ts = GFF_input['t'].to_numpy()
    Qs = GFF_input['Q'].to_numpy()
    flvs = GFF_input['flv'].to_numpy()
    spes = GFF_input['spe'].to_numpy()
    
    # Constants
    x = 0
    xi = 0
    
    ps = np.where(spes <= 1, 1, -1)
    spes = np.where(spes <= 1, spes, spes - 2)
    Para_spe = Para[spes]

    # Prepare input arguments for parallel computation
    args = [(j_i, x, xi, t_i, Q_i, p_i, flv_i, Para_i, p_order) 
            for j_i, t_i, Q_i, p_i, flv_i, Para_i 
            in zip(js, ts, Qs, ps, flvs, Para_spe)]

    result = pool.map(GFF_theo_scalar_helper, args)
    
    return np.array(result)

def CFF_theo(xB, t, Q, Para_Unp, Para_Pol):
    x = 0
    xi = (1/(2 - xB) - (2*t*(-1 + xB))/(Q**2*(-2 + xB)**2))*xB
    H_E = GPDobserv(x, xi, t, Q, 1)
    Ht_Et = GPDobserv(x, xi, t, Q, -1)
    HCFF = H_E.CFF(Para_Unp[..., 0, :, :, :, :], Q)
    ECFF = H_E.CFF(Para_Unp[..., 1, :, :, :, :], Q)
    HtCFF = Ht_Et.CFF(Para_Pol[..., 0, :, :, :, :], Q)
    EtCFF = Ht_Et.CFF(Para_Pol[..., 1, :, :, :, :], Q)

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

def DVCSxsec_HERA_theo(DVCSxsec_HERA_input: pd.DataFrame, CFF_input: np.array):
    #[y, xB, t, Q, f, delta_f, pol]  = DVCSxsec_data_HERA
    y = DVCSxsec_HERA_input['y'].to_numpy()
    xB = DVCSxsec_HERA_input['xB'].to_numpy()
    t = DVCSxsec_HERA_input['t'].to_numpy()
    Q = DVCSxsec_HERA_input['Q'].to_numpy()
    #f = DVCSxsec_data_HERA['f'].to_numpy()
    #delta_f = DVCSxsec_data_HERA['delta f'].to_numpy()
    pol = DVCSxsec_HERA_input['pol'].to_numpy()

    [HCFF, ECFF, HtCFF, EtCFF] = CFF_input
    return dsigma_DVCS_HERA(y, xB, t, Q, pol, HCFF, ECFF, HtCFF, EtCFF)

def DVCSxsec_HERA_cost_xBtQ(DVCSxsec_HERA_data_xBtQ: pd.DataFrame, Para_Unp, Para_Pol):
    [xB, t, Q] = [DVCSxsec_HERA_data_xBtQ['xB'].iat[0], DVCSxsec_HERA_data_xBtQ['t'].iat[0], DVCSxsec_HERA_data_xBtQ['Q'].iat[0]] 
    [HCFF, ECFF, HtCFF, EtCFF] = CFF_theo(xB, t, Q, Para_Unp, Para_Pol) # scalar for each of them
    # DVCS_pred_xBtQ = np.array(list(map(partial(DVCSxsec_theo, CFF_input = [HCFF, ECFF, HtCFF, EtCFF]), np.array(DVCSxsec_data_xBtQ))))
    DVCS_HERA_pred_xBtQ = DVCSxsec_HERA_theo(DVCSxsec_HERA_data_xBtQ, CFF_input = [HCFF, ECFF, HtCFF, EtCFF])
    return np.sum(((DVCS_HERA_pred_xBtQ - DVCSxsec_HERA_data_xBtQ['f'])/ DVCSxsec_HERA_data_xBtQ['delta f']) ** 2 )

def TFF_theo(xB, t, Q, Para_Unp, meson:int, p_order = 1, muset = 1, flv = 'All'):
    x = 0
    xi = (1/(2 - xB) - (2*t*(-1 + xB))/((Q**2)*(-2 + xB)**2))*xB
    if (meson==3):
       xi = (1/(2 - xB) - (2*t*(-1 + xB))/((Q**2 + M_jpsi**2)*(-2 + xB)**2))*xB
    H_E = GPDobserv(x, xi, t, Q, 1)
    HTFF = H_E.TFF(Para_Unp[..., 0, :, :, :, :], muset * Q, meson, p_order, flv)
    ETFF = H_E.TFF(Para_Unp[..., 1, :, :, :, :], muset * Q, meson, p_order, flv)

    return  [ HTFF, ETFF]

def DVMPxsec_theo(DVMPxsec_input: pd.DataFrame, TFF_input: np.array, meson:int):
    y = DVMPxsec_input['y'].to_numpy()
    xB = DVMPxsec_input['xB'].to_numpy()
    t = DVMPxsec_input['t'].to_numpy()
    Q = DVMPxsec_input['Q'].to_numpy()    
    [HTFF, ETFF] = TFF_input

    return dsigma_dt(y, xB, t, Q, meson, HTFF, ETFF)

def DVMPxsec_cost_xBtQ(DVMPxsec_data_xBtQ: pd.DataFrame, Para_Unp, xsec_norm, meson:int, p_order=2):

    [xB, t, Q] = [DVMPxsec_data_xBtQ['xB'].iat[0], DVMPxsec_data_xBtQ['t'].iat[0], DVMPxsec_data_xBtQ['Q'].iat[0]] 
    [HTFF, ETFF] = TFF_theo(xB, t, Q, Para_Unp, meson, p_order, muset = 1)
    DVMP_pred_xBtQ = DVMPxsec_theo(DVMPxsec_data_xBtQ, TFF_input = [HTFF, ETFF], meson = meson) * xsec_norm**2
    return np.sum(((DVMP_pred_xBtQ - DVMPxsec_data_xBtQ['f'])/ DVMPxsec_data_xBtQ['delta f']) ** 2 )

def cost_forward_H(Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
                   Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
                   Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
                   Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar, 
                   Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg,
                   Norm_EuV,    alpha_EuV,    beta_EuV,    alphap_EuV,
                   Norm_EdV,    R_E_Sea,      R_Hu_xi2,    R_Hd_xi2,    R_Hg_xi2,
                   R_Eu_xi2,    R_Ed_xi2,     R_Eg_xi2,
                   R_Hu_xi4,    R_Hd_xi4,     R_Hg_xi4,
                   R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea, bexp_Hg, Invm2_Hg):

    # parameters not used in dvcs fit
    bexp_Hg = bexp_HSea
    Invm2_Hg = 0
    
    global Minuit_Counter, Time_Counter

    time_now = time.time() - time_start
    
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
               Norm_EdV,    R_E_Sea,      R_Hu_xi2,    R_Hd_xi2,    R_Hg_xi2,
               R_Eu_xi2,    R_Ed_xi2,     R_Eg_xi2,
               R_Hu_xi4,    R_Hd_xi4,     R_Hg_xi4,
               R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea,   bexp_Hg, Invm2_Hg]
    
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
                   R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea, bexp_Hg, Invm2_Hg):

    # parameters not used in dvcs fit
    bexp_Hg = bexp_HSea
    Invm2_Hg = 0
    
    Paralst = [Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
               Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
               Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
               Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar, 
               Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg,
               Norm_EuV,    alpha_EuV,    beta_EuV,    alphap_EuV,
               Norm_EdV,    R_E_Sea,      R_Hu_xi2,    R_Hd_xi2,    R_Hg_xi2,
               R_Eu_xi2,    R_Ed_xi2,     R_Eg_xi2,
               R_Hu_xi4,    R_Hd_xi4,     R_Hg_xi4,
               R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea, bexp_Hg, Invm2_Hg]
    
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
     R_Eu_xi4_Init,    R_Ed_xi4_Init,     R_Eg_xi4_Init,    bexp_HSea_Init, bexp_Hg_Init, Invm2_Hg_Init] = Paralst_Unp

    fit_forw_H = Minuit(cost_forward_H, Norm_HuV = Norm_HuV_Init,     alpha_HuV = alpha_HuV_Init,      beta_HuV = beta_HuV_Init,     alphap_HuV = alphap_HuV_Init, 
                                        Norm_Hubar = Norm_Hubar_Init, alpha_Hubar = alpha_Hubar_Init,  beta_Hubar = beta_Hubar_Init, alphap_Hqbar = alphap_Hqbar_Init,
                                        Norm_HdV = Norm_HdV_Init,     alpha_HdV = alpha_HdV_Init,      beta_HdV = beta_HdV_Init,     alphap_HdV = alphap_HdV_Init,
                                        Norm_Hdbar = Norm_Hdbar_Init, alpha_Hdbar = alpha_Hdbar_Init,  beta_Hdbar = beta_Hdbar_Init, 
                                        Norm_Hg = Norm_Hg_Init,       alpha_Hg = alpha_Hg_Init,        beta_Hg = beta_Hg_Init,       alphap_Hg = alphap_Hg_Init,
                                        Norm_EuV = Norm_EuV_Init,     alpha_EuV = alpha_EuV_Init,      beta_EuV = beta_EuV_Init,     alphap_EuV = alphap_EuV_Init, 
                                        Norm_EdV = Norm_EdV_Init,     R_E_Sea = R_E_Sea_Init,          R_Hu_xi2 = R_Hu_xi2_Init,     R_Hd_xi2 = R_Hd_xi2_Init,     R_Hg_xi2 = R_Hg_xi2_Init,
                                        R_Eu_xi2 = R_Eu_xi2_Init,     R_Ed_xi2 = R_Ed_xi2_Init,        R_Eg_xi2 = R_Eg_xi2_Init,
                                        R_Hu_xi4 = R_Hu_xi4_Init,     R_Hd_xi4 = R_Hd_xi4_Init,        R_Hg_xi4 = R_Hg_xi4_Init,
                                        R_Eu_xi4 = R_Eu_xi4_Init,     R_Ed_xi4 = R_Ed_xi4_Init,        R_Eg_xi4 = R_Eg_xi4_Init,     bexp_HSea = bexp_HSea_Init,   bexp_Hg = bexp_Hg_Init, Invm2_Hg = Invm2_Hg_Init)
    fit_forw_H.errordef = 1
    
    # Parameters not used in DVCS fit
    fit_forw_H.fixed['bexp_Hg'] = True
    fit_forw_H.fixed['Invm2_Hg'] = True
    
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
    
    global Minuit_Counter, Time_Counter, time_start
    Minuit_Counter = 0
    Time_Counter = 1
    time_start = time.time()

    fit_forw_H.migrad()
    fit_forw_H.hesse()

    ndof_H = len(PDF_data_H.index) + len(tPDF_data_H.index) + len(GFF_data_H.index)  - fit_forw_H.nfit 

    time_end = time.time() -time_start

    with open(os.path.join(dir_path,'GUMP_Output/H_forward_fit.txt'), 'w', encoding='utf-8') as f:
        print('Total running time: %.1f minutes. Total call of cost function: %3d.\n' % ( time_end/60, fit_forw_H.nfcn), file=f)
        print('The chi squared/d.o.f. is: %.2f / %3d ( = %.2f ).\n' % (fit_forw_H.fval, ndof_H, fit_forw_H.fval/ndof_H), file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*fit_forw_H.values, sep=", ", file = f)
        print(*fit_forw_H.errors, sep=", ", file = f)
        print(fit_forw_H.params, file = f)

    with open(os.path.join(dir_path,"GUMP_Output/H_forward_cov.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows([*fit_forw_H.covariance])

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
     R_Eu_xi4_Init,    R_Ed_xi4_Init,     R_Eg_xi4_Init,    bexp_HSea_Init, bexp_Hg_Init, Invm2_Hg_Init] = Paralst_Unp

    fit_forw_E = Minuit(cost_forward_E, Norm_HuV = Norm_HuV_Init,     alpha_HuV = alpha_HuV_Init,      beta_HuV = beta_HuV_Init,     alphap_HuV = alphap_HuV_Init, 
                                        Norm_Hubar = Norm_Hubar_Init, alpha_Hubar = alpha_Hubar_Init,  beta_Hubar = beta_Hubar_Init, alphap_Hqbar = alphap_Hqbar_Init,
                                        Norm_HdV = Norm_HdV_Init,     alpha_HdV = alpha_HdV_Init,      beta_HdV = beta_HdV_Init,     alphap_HdV = alphap_HdV_Init,
                                        Norm_Hdbar = Norm_Hdbar_Init, alpha_Hdbar = alpha_Hdbar_Init,  beta_Hdbar = beta_Hdbar_Init, 
                                        Norm_Hg = Norm_Hg_Init,       alpha_Hg = alpha_Hg_Init,        beta_Hg = beta_Hg_Init,       alphap_Hg = alphap_Hg_Init,
                                        Norm_EuV = Norm_EuV_Init,     alpha_EuV = alpha_EuV_Init,      beta_EuV = beta_EuV_Init,     alphap_EuV = alphap_EuV_Init, 
                                        Norm_EdV = Norm_EdV_Init,     R_E_Sea = R_E_Sea_Init,          R_Hu_xi2 = R_Hu_xi2_Init,     R_Hd_xi2 = R_Hd_xi2_Init,     R_Hg_xi2 = R_Hg_xi2_Init,
                                        R_Eu_xi2 = R_Eu_xi2_Init,     R_Ed_xi2 = R_Ed_xi2_Init,        R_Eg_xi2 = R_Eg_xi2_Init,
                                        R_Hu_xi4 = R_Hu_xi4_Init,     R_Hd_xi4 = R_Hd_xi4_Init,        R_Hg_xi4 = R_Hg_xi4_Init,
                                        R_Eu_xi4 = R_Eu_xi4_Init,     R_Ed_xi4 = R_Ed_xi4_Init,        R_Eg_xi4 = R_Eg_xi4_Init,     bexp_HSea = bexp_HSea_Init,   bexp_Hg = bexp_Hg_Init, Invm2_Hg = Invm2_Hg_Init)
    fit_forw_E.errordef = 1

    # Parameters not used in DVCS fit
    fit_forw_E.fixed['bexp_Hg'] = True
    fit_forw_E.fixed['Invm2_Hg'] = True
    
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

    with open(os.path.join(dir_path,'GUMP_Output/E_forward_fit.txt'), 'w', encoding='utf-8') as f:
        print('Total running time: %.1f minutes. Total call of cost function: %3d.\n' % ( time_end/60, fit_forw_E.nfcn), file=f)
        print('The chi squared/d.o.f. is: %.2f / %3d ( = %.2f ).\n' % (fit_forw_E.fval, ndof_E, fit_forw_E.fval/ndof_E), file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*fit_forw_E.values, sep=", ", file = f)
        print(*fit_forw_E.errors, sep=", ", file = f)
        print(fit_forw_E.params, file = f)

    with open(os.path.join(dir_path,"GUMP_Output/E_forward_cov.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows([*fit_forw_E.covariance])

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
    with open(os.path.join(dir_path,'GUMP_Output/Ht_forward_fit.txt'), 'w', encoding='utf-8') as f:
        print('Total running time: %.1f minutes. Total call of cost function: %3d.\n' % ( time_end/60, fit_forw_Ht.nfcn), file=f)
        print('The chi squared/d.o.f. is: %.2f / %3d ( = %.2f ).\n' % (fit_forw_Ht.fval, ndof_Ht, fit_forw_Ht.fval/ndof_Ht), file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*fit_forw_Ht.values, sep=", ", file = f)
        print(*fit_forw_Ht.errors, sep=", ", file = f)
        print(fit_forw_Ht.params, file = f)

    with open(os.path.join(dir_path,"GUMP_Output/Ht_forward_cov.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows([*fit_forw_Ht.covariance])

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
    with open(os.path.join(dir_path,'GUMP_Output/Et_forward_fit.txt'), 'w', encoding='utf-8') as f:
        print('Total running time: %.1f minutes. Total call of cost function: %3d.\n' % ( time_end/60, fit_forw_Et.nfcn), file=f)
        print('The chi squared/d.o.f. is: %.2f / %3d ( = %.2f ).\n' % (fit_forw_Et.fval, ndof_Et, fit_forw_Et.fval/ndof_Et), file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*fit_forw_Et.values, sep=", ", file = f)
        print(*fit_forw_Et.errors, sep=", ", file = f)
        print(fit_forw_Et.params, file = f)

    with open(os.path.join(dir_path,"GUMP_Output/Et_forward_cov.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows([*fit_forw_Et.covariance])

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
                     R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea,  bexp_Hg, Invm2_Hg,
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

    # Parameters not used in DVCS fit
    bexp_Hg = bexp_HSea
    Invm2_Hg = 0
    
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
                    R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea, bexp_Hg, Invm2_Hg]

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
    
    cost_DVCS_HERA_xBtQ = np.array(list(pool.map(partial(DVCSxsec_HERA_cost_xBtQ, Para_Unp = Para_Unp_all, Para_Pol = Para_Pol_all), DVCSxsec_HERA_group_data)))
    cost_DVCSxsec_HERA = np.sum(cost_DVCS_HERA_xBtQ)

    # DVCS_HERA_pred = np.array(list(pool.map(partial(DVCSxsec_HERA_theo, Para_Unp = Para_Unp_all, Para_Pol = Para_Pol_all), np.array(DVCS_HERA_data))))
    #DVCS_HERA_pred = DVCSxsec_HERA_theo(DVCS_HERA_data, Para_Unp=Para_Unp_all, Para_Pol=Para_Pol_all)
    #cost_DVCS_HERA = np.sum(((DVCS_HERA_pred - DVCS_HERA_data['f'])/ DVCS_HERA_data['delta f']) ** 2 )

    return  cost_DVCSxsec + cost_DVCSxsec_HERA



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
     R_Eu_xi4_Init,    R_Ed_xi4_Init,     R_Eg_xi4_Init,    bexp_HSea_Init, bexp_Hg_Init, Invm2_Hg_Init] = Paralst_Unp

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
                                               R_Eu_xi4 = R_Eu_xi4_Init,     R_Ed_xi4 = R_Ed_xi4_Init,        R_Eg_xi4 = R_Eg_xi4_Init,     bexp_HSea = bexp_HSea_Init, bexp_Hg = bexp_Hg_Init, Invm2_Hg = Invm2_Hg_Init,
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
    
    #Parameters not used in dvcs fit
    fit_off_forward.fixed['bexp_Hg'] = True
    fit_off_forward.fixed['Invm2_Hg'] = True

    fit_off_forward.limits['bexp_HSea']  = (0, 10)
    fit_off_forward.limits['bexp_HtSea'] = (0, 10)

    fit_off_forward.limits['R_Et_Sea']   = (-50, 50)

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

    fit_off_forward.fixed['R_Hu_xi4']  = True 
    fit_off_forward.fixed['R_Eu_xi4']  = True
    fit_off_forward.fixed['R_Htu_xi4'] = True 
    fit_off_forward.fixed['R_Etu_xi4'] = True

    fit_off_forward.fixed['R_Hd_xi4']  = True 
    fit_off_forward.fixed['R_Ed_xi4']  = True
    fit_off_forward.fixed['R_Htd_xi4'] = True 
    fit_off_forward.fixed['R_Etd_xi4'] = True

    global Minuit_Counter, Time_Counter, time_start
    Minuit_Counter = 0
    Time_Counter = 1
    time_start = time.time()
    
    fit_off_forward.migrad()
    fit_off_forward.hesse()

    ndof_off_forward = len(DVCSxsec_data.index) + len(DVCSxsec_HERA_data.index) - fit_off_forward.nfit 
    
    #len(DVCSxsec_data.index) + 

    time_end = time.time() -time_start

    with open(os.path.join(dir_path,'GUMP_Output/off_forward_fit.txt'), 'w', encoding='utf-8') as f:
        print('Total running time: %.1f minutes. Total call of cost function: %3d.\n' % ( time_end/60, fit_off_forward.nfcn), file=f)
        print('The chi squared/d.o.f. is: %.2f / %3d ( = %.2f ).\n' % (fit_off_forward.fval, ndof_off_forward, fit_off_forward.fval/ndof_off_forward), file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*fit_off_forward.values, sep=", ", file = f)
        print(*fit_off_forward.errors, sep=", ", file = f)
        print(fit_off_forward.params, file = f)

    with open(os.path.join(dir_path,"GUMP_Output/Off_forward_cov.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows([*fit_off_forward.covariance])
          
    print("off forward fit finished...")
    return fit_off_forward

def cost_dvmp(Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV, 
                     Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
                     Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,
                     Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar, 
                     Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg,
                     Norm_EuV,    alpha_EuV,    beta_EuV,    alphap_EuV,
                     Norm_EdV,    R_E_Sea,      R_Hu_xi2,    R_Hd_xi2,    R_Hg_xi2,
                     R_Eu_xi2,    R_Ed_xi2,     R_Eg_xi2,
                     R_Hu_xi4,    R_Hd_xi4,     R_Hg_xi4,
                     R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea, bexp_Hg, Invm2_Hg, norm, norm2):

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
                    R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea, bexp_Hg, Invm2_Hg, norm, norm2]

    jpsinorm = Para_Unp_lst[-2]
    jpsinormzeus = Para_Unp_lst[-1] 
    Para_Unp_all = ParaManager_Unp(Para_Unp_lst[:-2])
    
    PDF_H_g_smallx_pred = PDF_theo(PDFg_smallx_data, Para=Para_Unp_all, p_order = 2)
    cost_PDF_H_g_smallx = np.sum(((PDF_H_g_smallx_pred - PDFg_smallx_data['f'])/ PDFg_smallx_data['delta f']) ** 2 )

    cost_DVjpsiPH1_xBtQ = np.array(list(pool.map(partial(DVMPxsec_cost_xBtQ, Para_Unp = Para_Unp_all, xsec_norm = jpsinorm, meson = 3, p_order = 2), DVJpsiPH1xsec_group_data)))
    cost_DVjpsiPH1xsec = np.sum(cost_DVjpsiPH1_xBtQ)

    return cost_PDF_H_g_smallx + cost_DVjpsiPH1xsec #+ cost_DVjpsiPZEUSxsec

def dvmp_fit(Paralst_Unp):

    [Norm_HuV_Init,    alpha_HuV_Init,    beta_HuV_Init,    alphap_HuV_Init, 
     Norm_Hubar_Init,  alpha_Hubar_Init,  beta_Hubar_Init,  alphap_Hqbar_Init,
     Norm_HdV_Init,    alpha_HdV_Init,    beta_HdV_Init,    alphap_HdV_Init,
     Norm_Hdbar_Init,  alpha_Hdbar_Init,  beta_Hdbar_Init, 
     Norm_Hg_Init,     alpha_Hg_Init,     beta_Hg_Init,     alphap_Hg_Init,
     Norm_EuV_Init,    alpha_EuV_Init,    beta_EuV_Init,    alphap_EuV_Init,
     Norm_EdV_Init,    R_E_Sea_Init,      R_Hu_xi2_Init,    R_Hd_xi2_Init,    R_Hg_xi2_Init,
     R_Eu_xi2_Init,    R_Ed_xi2_Init,     R_Eg_xi2_Init,
     R_Hu_xi4_Init,    R_Hd_xi4_Init,     R_Hg_xi4_Init,
     R_Eu_xi4_Init,    R_Ed_xi4_Init,     R_Eg_xi4_Init,    bexp_HSea_Init, bexp_Hg_Init, Invm2_Hg_Init, norm_Init, norm2_Init] = Paralst_Unp

    Invm2_Hg_Init = 0
    #bexp_Hg_Init = 0
    fit_dvmp = Minuit(cost_dvmp, Norm_HuV = Norm_HuV_Init,     alpha_HuV = alpha_HuV_Init,      beta_HuV = beta_HuV_Init,     alphap_HuV = alphap_HuV_Init, 
                                            Norm_Hubar = Norm_Hubar_Init, alpha_Hubar = alpha_Hubar_Init,  beta_Hubar = beta_Hubar_Init, alphap_Hqbar = alphap_Hqbar_Init,
                                            Norm_HdV = Norm_HdV_Init,     alpha_HdV = alpha_HdV_Init,      beta_HdV = beta_HdV_Init,     alphap_HdV = alphap_HdV_Init,
                                            Norm_Hdbar = Norm_Hdbar_Init, alpha_Hdbar = alpha_Hdbar_Init,  beta_Hdbar = beta_Hdbar_Init, 
                                            Norm_Hg = Norm_Hg_Init,       alpha_Hg = alpha_Hg_Init,        beta_Hg = beta_Hg_Init,       alphap_Hg = alphap_Hg_Init,
                                            Norm_EuV = Norm_EuV_Init,     alpha_EuV = alpha_EuV_Init,      beta_EuV = beta_EuV_Init,     alphap_EuV = alphap_EuV_Init, 
                                            Norm_EdV = Norm_EdV_Init,     R_E_Sea = R_E_Sea_Init,          R_Hu_xi2 = R_Hu_xi2_Init,     R_Hd_xi2 = R_Hd_xi2_Init,     R_Hg_xi2 = R_Hg_xi2_Init,
                                            R_Eu_xi2 = R_Eu_xi2_Init,     R_Ed_xi2 = R_Ed_xi2_Init,        R_Eg_xi2 = R_Eg_xi2_Init,
                                            R_Hu_xi4 = R_Hu_xi4_Init,     R_Hd_xi4 = R_Hd_xi4_Init,        R_Hg_xi4 = R_Hg_xi4_Init,
                                            R_Eu_xi4 = R_Eu_xi4_Init,     R_Ed_xi4 = R_Ed_xi4_Init,        R_Eg_xi4 = R_Eg_xi4_Init,     bexp_HSea = bexp_HSea_Init, bexp_Hg = bexp_Hg_Init, Invm2_Hg = Invm2_Hg_Init, norm = norm_Init, norm2 = norm2_Init)
    fit_dvmp.errordef = 1

    fit_dvmp.fixed['bexp_HSea'] = True
    
    #fit_dvmp.fixed['bexp_Hg'] = True
    #fit_dvmp.limits['Invm2_Hg'] = (0.1,10)
    
    fit_dvmp.fixed['Invm2_Hg'] = True
    fit_dvmp.limits['bexp_Hg']  = (0.1,4)
    
    fit_dvmp.limits['norm'] = (0.1,10)
    #fit_dvmp.fixed['norm'] = True
    
    #fit_dvmp.limits['norm2'] = (0.1,10)
    fit_dvmp.fixed['norm2'] = True

    fit_dvmp.fixed['Norm_HuV'] = True
    fit_dvmp.fixed['alpha_HuV'] = True
    fit_dvmp.fixed['beta_HuV'] = True
    fit_dvmp.fixed['alphap_HuV'] = True
    
    #fit_dvmp.limits['Norm_Hubar']  = (0.2, 3)
    #fit_dvmp.limits['alpha_Hubar']  = (1.02, 1.5)

    fit_dvmp.fixed['Norm_Hubar'] = True
    fit_dvmp.fixed['alpha_Hubar'] = True
    fit_dvmp.fixed['beta_Hubar'] = True

    fit_dvmp.fixed['alphap_Hqbar'] = True

    fit_dvmp.fixed['Norm_HdV'] = True
    fit_dvmp.fixed['alpha_HdV'] = True
    fit_dvmp.fixed['beta_HdV'] = True
    fit_dvmp.fixed['alphap_HdV'] = True
    
    #fit_dvmp.limits['Norm_Hdbar']  = (0, 10)

    fit_dvmp.fixed['Norm_Hdbar'] = True
    fit_dvmp.fixed['alpha_Hdbar'] = True
    fit_dvmp.fixed['beta_Hdbar'] = True

    

    fit_dvmp.limits['Norm_Hg']=(0,10)
    fit_dvmp.limits['alpha_Hg']=(0,2)
    fit_dvmp.fixed['beta_Hg'] = True
    #fit_dvmp.limits['beta_Hg']=(1,10)

   # fit_dvmp.fixed['Norm_Hg'] = True
   # fit_dvmp.fixed['alpha_Hg'] = True
   # fit_dvmp.fixed['beta_Hg'] = True

    fit_dvmp.fixed['Norm_EuV'] = True
    fit_dvmp.fixed['alpha_EuV'] = True
    fit_dvmp.fixed['beta_EuV'] = True
    fit_dvmp.fixed['alphap_EuV'] = True

    fit_dvmp.fixed['Norm_EdV'] = True

    #fit_dvmp.fixed['alphap_Hg'] = True
    fit_dvmp.limits['alphap_Hg'] = (0,0.5)
    
    #fit_dvmp.limits['R_E_Sea']=(0,10)    

    #fit_dvmp.limits['R_Hg_xi2'] = (-20,20)

    fit_dvmp.fixed['R_E_Sea'] = True    
    fit_dvmp.fixed['R_Hu_xi2'] = True
    fit_dvmp.fixed['R_Hd_xi2'] = True     
   # fit_dvmp.fixed['R_Hg_xi2'] = True
    fit_dvmp.fixed['R_Eu_xi2'] = True
    fit_dvmp.fixed['R_Ed_xi2'] = True 
    fit_dvmp.fixed['R_Eg_xi2'] = True
    

    #fit_dvmp.fixed['R_Hg_xi4'] = True
    fit_dvmp.fixed['R_Eg_xi4'] = True
    

    fit_dvmp.fixed['R_Hu_xi4']  = True 
    fit_dvmp.fixed['R_Eu_xi4']  = True

    fit_dvmp.fixed['R_Hd_xi4']  = True 
    fit_dvmp.fixed['R_Ed_xi4']  = True
    
    
    global Minuit_Counter, Time_Counter, time_start
    Minuit_Counter = 0
    Time_Counter = 1
    time_start = time.time()
    
    fit_dvmp.migrad()
    fit_dvmp.hesse()

    ndof_dvmp = len(DVJpsiPH1xsec_data) + len(PDFg_smallx_data) - fit_dvmp.nfit 

    time_end = time.time() -time_start

    with open(os.path.join(dir_path,'GUMP_Output/dvmp_fit_NLOPDF.txt'), 'w', encoding="utf-8") as f:
        print('Total running time: %.1f minutes. Total call of cost function: %3d.\n' % ( time_end/60, fit_dvmp.nfcn), file=f)
        print('The chi squared/d.o.f. is: %.2f / %3d ( = %.2f ).\n' % (fit_dvmp.fval, ndof_dvmp, fit_dvmp.fval/ndof_dvmp), file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*fit_dvmp.values, sep=", ", file = f)
        print(*fit_dvmp.errors, sep=", ", file = f)
        print(fit_dvmp.params, file = f)

    with open(os.path.join(dir_path,"GUMP_Output/dvmp_cov.csv"),"w", newline='', encoding="utf-8") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows([*fit_dvmp.covariance])
        
    with open(os.path.join(dir_path,"GUMP_Params/Para_Unp.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerow(list([*fit_dvmp.values]))
        csvWriter.writerow(list([*fit_dvmp.errors]))
  
    print("dvmp fit finished...")
    return fit_dvmp

if __name__ == '__main__':
    pool = Pool()
    time_start = time.time()
    
    '''
    Paralst_Unp=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Unp.csv'), header=None).to_numpy()[0]
    Paralst_Pol=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Pol.csv'), header=None).to_numpy()[0]

    fit_dvmp = dvmp_fit(Paralst_Unp)
    '''

    Paralst_Unp     = [4.922551238,0.21635596,3.228702555,2.349193947,0.163440601,1.135738688,6.896742038,0.15,3.358541913,0.184196049,4.41726899,3.475742056,0.249183402,1.051922382,6.548676693,2.864281106,1.052305853,7.412779844,0.15,0.161159704,0.916012032,1.02239598,0.41423421,-0.198595321,0.0,0.18394307,-2.260952723,0,1.159322377,2.569800357,0,0,0,0,0,0,0,3.296968216,0,0]
    Paralst_Pol     = [4.529773253,-0.246812532,3.037043159,2.607360484,0.076575866,0.516192897,4.369657188,0.15,-0.711694724,0.210181857,3.243538578,4.319727451,-0.057100694,0.612255908,2.099180441,0.243247279,0.630824175,2.71840147,0.15,9.065736349,0.79999977,7.357005187,2.083472023,-3.562901039,0.0,-0.634095327,-7.058667382,0,2.861662204,23.1231347,0,0,0,0,0,0,0,5.379752095]
   
    Para_Unp = ParaManager_Unp(Paralst_Unp)
    Para_Pol = ParaManager_Pol(Paralst_Pol)
    
    fit_forward_H   = forward_H_fit(Paralst_Unp)
    Paralst_Unp     = np.array(fit_forward_H.values)

    fit_forward_Ht  = forward_Ht_fit(Paralst_Pol)
    Paralst_Pol     = np.array(fit_forward_Ht.values)

    fit_forward_E   = forward_E_fit(Paralst_Unp)
    Paralst_Unp     = np.array(fit_forward_E.values)

    fit_forward_Et  = forward_Et_fit(Paralst_Pol)
    Paralst_Pol     = np.array(fit_forward_Et.values)
    
    fit_off_forward = off_forward_fit(Paralst_Unp, Paralst_Pol)
