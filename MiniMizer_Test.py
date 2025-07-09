from Parameters import ParaManager_Unp, ParaManager_Pol
from Observables import GPDobserv
from DVCS_xsec import dsigma_DVCS_TOT, dsigma_DVCS_HERA, M
from DVMP_xsec import dsigma_DVMP_dt,dsigmaL_DVMP_dt, M_jpsi,epsilon, R_fitted
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

"""
************************ PDF and tPDFs data preprocessing ****************************
"""

PDF_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/PDFdata.csv'), header = 0, names = ['x', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'x': float, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})
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

tPDF_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/tPDFdata.csv'),     header = 0, names = ['x', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'x': float, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})
tPDF_data_H  = tPDF_data[tPDF_data['spe'] == 0]
tPDF_data_H_g = tPDF_data_H[tPDF_data_H['flv'] == 'g']
tPDF_data_H_g = tPDF_data_H[tPDF_data_H['x'] < xB_small_Cut]
tPDF_data_E  = tPDF_data[tPDF_data['spe'] == 1]
tPDF_data_Ht = tPDF_data[tPDF_data['spe'] == 2]
tPDF_data_Et = tPDF_data[tPDF_data['spe'] == 3]

"""
************************ GFF data preprocessing ****************************
"""

GFF_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/GFFdata_Quark.csv'),       header = 0, names = ['j', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'j': int, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})
GFF_data_H  = GFF_data[GFF_data['spe'] == 0]
GFF_data_E  = GFF_data[GFF_data['spe'] == 1]
GFF_data_Ht = GFF_data[GFF_data['spe'] == 2]
GFF_data_Et = GFF_data[GFF_data['spe'] == 3]

GFF_Gluon_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/GFFdata_Gluon.csv'),       header = None, names = ['j', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'j': int, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})
GFF_Gluon_data_H  = GFF_Gluon_data[GFF_Gluon_data['spe'] == 0]
GFF_Gluon_data_E  = GFF_Gluon_data[GFF_Gluon_data['spe'] == 1]
GFF_Gluon_data_Ht = GFF_Gluon_data[GFF_Gluon_data['spe'] == 2]
GFF_Gluon_data_Et = GFF_Gluon_data[GFF_Gluon_data['spe'] == 3]

"""
************************ DVCS data preprocessing ****************************
"""

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

"""
************************ DVMP for rho data preprocessing ****************************
"""

def DVMP_L_Error_Prop(DVMP_tot_xsec: pd.DataFrame, meson: int =1): 
    """ Error propagation for dσ_L /dt= (dσ_tot /dt) / (ε(y) + 1/R(Q;a,p,meson)).

    Args:
        DVMP_tot_xsec (DataFrame): total cross-sections including at least four column:
            'y', 'Q', 'f' and 'delta f' for lepton energy loss, photon virtuality, total cross-sections and uncertainties of total cross-sections.
        meson: 1 for rho production

    Returns:
        dσ_L(y,Q,a,p,meson) / dt with standart deviation
    """
    y_vals = DVMP_tot_xsec['y'].to_numpy()
    Q_vals = DVMP_tot_xsec['Q'].to_numpy()
    tot_xsec = DVMP_tot_xsec['f'].to_numpy()
    tot_xsec_err = DVMP_tot_xsec['delta f'].to_numpy()
    
    R_Mean, R_err = R_fitted(Q_vals, meson = meson)
    
    # ∂(dσ_L/dt)/∂(dσ_tot/dt) = 1 / (ε + 1/R)
    partial_derivative_dsigma_dt = 1 / (epsilon(y_vals) + 1 / R_Mean)
    # ∂(dσ_L/dt)/∂R = (dσ_tot/dt) / (R²(ε + 1/R)²)
    partial_derivative_R = tot_xsec /(R_Mean**2 * (epsilon(y_vals) + 1 / R_Mean)**2)  
   
    #Forming each piece of the variance: 
    part_sigma_dt = partial_derivative_dsigma_dt**2  * tot_xsec_err ** 2
    part_R = partial_derivative_R**2 * R_err ** 2
    
    dsigmaL_xsec_dt = tot_xsec / (epsilon(y_vals) + 1 / R_Mean)
    variance_dsigmaL_dt=part_sigma_dt + part_R  # Here we assume σ_tot and R independent, so their corelation=0

    return dsigmaL_xsec_dt, np.sqrt(variance_dsigmaL_dt)

DVrhoPZEUSxsec_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/DVrhoPZEUSdt.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float})
DVrhoPZEUSxsec_data['Q'] = np.sqrt(DVrhoPZEUSxsec_data['Q'])
DVrhoPZEUSxsec_data['t'] = -1 * DVrhoPZEUSxsec_data['t']
DVrhoPZEUSxsec_data = DVrhoPZEUSxsec_data[(DVrhoPZEUSxsec_data['Q']>Q_threshold)]
xBtQlst_rhoZ = DVrhoPZEUSxsec_data.drop_duplicates(subset = ['xB', 't', 'Q'], keep = 'first')[['xB','t','Q']].values.tolist()
DVrhoPZEUSxsec_group_data = list(map(lambda set: DVrhoPZEUSxsec_data[(DVrhoPZEUSxsec_data['xB'] == set[0]) & (DVrhoPZEUSxsec_data['t'] == set[1]) & ((DVrhoPZEUSxsec_data['Q'] == set[2]))], xBtQlst_rhoZ))

# Converting to longitudinal cross-sections
DVrhoPZEUSxsecL_data = DVrhoPZEUSxsec_data.copy()
dsigmaL_dt_ZEUS, dsigmaL_dt_err_ZEUS = DVMP_L_Error_Prop(DVrhoPZEUSxsecL_data, 1)
DVrhoPZEUSxsecL_data['f'] = dsigmaL_dt_ZEUS
DVrhoPZEUSxsecL_data['delta f'] = dsigmaL_dt_err_ZEUS
DVrhoPZEUSxsecL_group_data = list(map(lambda set: DVrhoPZEUSxsecL_data[(DVrhoPZEUSxsecL_data['xB'] == set[0]) & (DVrhoPZEUSxsecL_data['t'] == set[1]) & ((DVrhoPZEUSxsecL_data['Q'] == set[2]))], xBtQlst_rhoZ))

DVrhoPH1xsec_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/DVrhoPH1dt.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float})
DVrhoPH1xsec_data['Q'] = np.sqrt(DVrhoPH1xsec_data['Q'])
DVrhoPH1xsec_data['t'] = -1 * DVrhoPH1xsec_data['t']
DVrhoPH1xsec_data = DVrhoPH1xsec_data[(DVrhoPH1xsec_data['Q']>Q_threshold)]
xBtQlst_rhoH = DVrhoPH1xsec_data.drop_duplicates(subset = ['xB', 't', 'Q'], keep = 'first')[['xB','t','Q']].values.tolist()
DVrhoPH1xsec_group_data = list(map(lambda set: DVrhoPH1xsec_data[(DVrhoPH1xsec_data['xB'] == set[0]) & (DVrhoPH1xsec_data['t'] == set[1]) & ((DVrhoPH1xsec_data['Q'] == set[2]))], xBtQlst_rhoH))

# Converting to longitudinal cross-sections
DVrhoPH1xsecL_data = DVrhoPH1xsec_data.copy()
dsigmaL_dt_H1, dsigmaL_dt_err_H1 = DVMP_L_Error_Prop(DVrhoPH1xsecL_data, 1)
DVrhoPH1xsecL_data['f'] = dsigmaL_dt_H1
DVrhoPH1xsecL_data['delta f'] = dsigmaL_dt_err_H1
DVrhoPH1xsecL_group_data = list(map(lambda set: DVrhoPH1xsecL_data[(DVrhoPH1xsecL_data['xB'] == set[0]) & (DVrhoPH1xsecL_data['t'] == set[1]) & ((DVrhoPH1xsecL_data['Q'] == set[2]))], xBtQlst_rhoH))

"""
************************ DVMP for phi data preprocessing ****************************
"""

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

"""
************************ DVMP for Jpsi data preprocessing ****************************
"""

DVJpsiPH1xsec_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/DVJpsiPH1dt_w_mass.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float})
DVJpsiPH1xsec_data['Q'] = np.sqrt(DVJpsiPH1xsec_data['Q'])
DVJpsiPH1xsec_data['t'] = -1 * DVJpsiPH1xsec_data['t']
DVJpsiPH1xsec_data = DVJpsiPH1xsec_data[(DVJpsiPH1xsec_data['Q']>Q_threshold)]
xBtQlst_JpsiH = DVJpsiPH1xsec_data.drop_duplicates(subset = ['xB', 't', 'Q'], keep = 'first')[['xB','t','Q']].values.tolist()
DVJpsiPH1xsec_L_data = DVJpsiPH1xsec_data.copy()

'''
For Jpsi we used NRQCD framework that predict the R ratio, the following code not in use

R = sigma_L / sigma_T values hardcoded and used to convert data xsec simga_tot to sigma_L

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

"""
************************ Photon productions of Jpsi data preprocessing (Not in use) ****************************
"""

JpsiphotoH1xsec_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/DVJpsiPZEUSdt_w_mass.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float})
JpsiphotoH1xsec_data['Q'] = np.sqrt(JpsiphotoH1xsec_data['Q'])
JpsiphotoH1xsec_data['t'] = -1 * JpsiphotoH1xsec_data['t']
JpsiphotoH1xsec_data = JpsiphotoH1xsec_data[(JpsiphotoH1xsec_data['Q']>Q_threshold)]
xBtQlst_JpsiphotoH1 = JpsiphotoH1xsec_data.drop_duplicates(subset = ['xB', 't', 'Q'], keep = 'first')[['xB','t','Q']].values.tolist()

# Helper function for scalar computation
def PDF_theo_scalar_helper(args):
    x_i, xi_i, t_i, Q_i, p_i, flv_i, Para_i, p_order = args
    _PDF_theo = GPDobserv(x_i, xi_i, t_i, Q_i, p_i)  
    return _PDF_theo.tPDF(flv_i, Para_i, p_order)

def PDF_theo(PDF_input: pd.DataFrame, Para: np.array, p_order = 2):
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

# Helper function for scalar computation
def GFF_theo_scalar_helper(args):

    j_i, x, xi, t_i, Q_i, p_i, flv_i, Para_i, p_order = args
    _GFF_theo = GPDobserv(x, xi, t_i, Q_i, p_i)
    return _GFF_theo.GFFj0(j_i, flv_i, Para_i, p_order)

def GFF_theo(GFF_input: pd.DataFrame, Para: np.array, p_order = 2):
    
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
    return dsigma_DVCS_TOT(y, xB, t, Q, phi, pol, HCFF, ECFF, HtCFF, EtCFF)

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

def TFF_theo(xB, t, Q, Para_Unp, meson:int, p_order = 2, muset = 1, flv = 'All'):
    x = 0
    xi = (1/(2 - xB) - (2*t*(-1 + xB))/((Q**2)*(-2 + xB)**2))*xB
    if (meson==3):
       xi = (1/(2 - xB) - (2*t*(-1 + xB))/((Q**2 + M_jpsi**2)*(-2 + xB)**2))*xB
    H_E = GPDobserv(x, xi, t, Q, 1)
    HTFF = H_E.TFF(Para_Unp[..., 0, :, :, :, :], muset * Q, meson, p_order, flv)
    ETFF = H_E.TFF(Para_Unp[..., 1, :, :, :, :], muset * Q, meson, p_order, flv)

    return  [ HTFF, ETFF]

def DVMPxsec_theo(DVMPxsec_input: pd.DataFrame,  TFF_input: np.array, meson:int):
    y = DVMPxsec_input['y'].to_numpy()
    xB = DVMPxsec_input['xB'].to_numpy()
    t = DVMPxsec_input['t'].to_numpy()
    Q = DVMPxsec_input['Q'].to_numpy()    
    [HTFF, ETFF] = TFF_input
    
    if (meson==3):
        # a and p are the parameters for R which are not need for J/psi, put int 0 for both of them as placeholder.
        return dsigma_DVMP_dt(y, xB, t, Q, meson, HTFF, ETFF,0,0)

    if (meson==1):
        return dsigmaL_DVMP_dt(y, xB, t, Q, meson, HTFF, ETFF)

def DVMPxsec_cost_xBtQ(DVMPxsec_data_xBtQ: pd.DataFrame, Para_Unp, xsec_norm, meson:int, p_order = 2):

    [xB, t, Q] = [DVMPxsec_data_xBtQ['xB'].iat[0], DVMPxsec_data_xBtQ['t'].iat[0], DVMPxsec_data_xBtQ['Q'].iat[0]] 
    [HTFF, ETFF] = TFF_theo(xB, t, Q, Para_Unp, meson, p_order, muset = 1)
    DVMP_pred_xBtQ = DVMPxsec_theo(DVMPxsec_data_xBtQ, [HTFF, ETFF], meson) * xsec_norm**2
    return np.sum(((DVMP_pred_xBtQ - DVMPxsec_data_xBtQ['f'])/ DVMPxsec_data_xBtQ['delta f']) ** 2 )

def cost_forward_H(Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV,   Invm2_HuV,
                   Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
                   Norm_Hubar_2,  alpha_Hubar_2,  beta_Hubar_2,
                   Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV,   Invm2_HdV,
                   Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar, 
                   Norm_Hdbar_2,  alpha_Hdbar_2,  beta_Hdbar_2,
                   Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg,  Invm2_Hg,
                   Norm_Hg_2,     alpha_Hg_2,     beta_Hg_2,
                   Norm_EuV,    alpha_EuV,    beta_EuV,    alphap_EuV,
                   Norm_EdV,    R_E_Sea,      R_Hu_xi2,    R_Hd_xi2,    R_Hg_xi2,
                   R_Eu_xi2,    R_Ed_xi2,     R_Eg_xi2,
                   R_Hu_xi4,    R_Hd_xi4,     R_Hg_xi4,
                   R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea, bexp_Hg):

    # parameters not used in dvcs fit
    bexp_Hg = bexp_HSea
    Invm2_Hg = 0
    
    global Minuit_Counter, Time_Counter

    time_now = time.time() - time_start
    
    if(time_now > Time_Counter * 600):
        print('Runing Time:',round(time_now/60),'minutes. Cost function called total', Minuit_Counter, 'times.')
        Time_Counter = Time_Counter + 1
    
    Minuit_Counter = Minuit_Counter + 1
    
    Paralst = [Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV,  Invm2_HuV,
                Norm_Hubar,  alpha_Hubar,  beta_Hubar,  alphap_Hqbar,
                Norm_Hubar_2,  alpha_Hubar_2,  beta_Hubar_2,
                Norm_HdV,    alpha_HdV,    beta_HdV,    alphap_HdV, Invm2_HdV,
                Norm_Hdbar,  alpha_Hdbar,  beta_Hdbar, 
                Norm_Hdbar_2,  alpha_Hdbar_2,  beta_Hdbar_2,
                Norm_Hg,     alpha_Hg,     beta_Hg,     alphap_Hg, Invm2_Hg,
                Norm_Hg_2,     alpha_Hg_2,     beta_Hg_2,
                Norm_EuV,    alpha_EuV,    beta_EuV,    alphap_EuV,
                Norm_EdV,    R_E_Sea,      R_Hu_xi2,    R_Hd_xi2,    R_Hg_xi2,
                R_Eu_xi2,    R_Ed_xi2,     R_Eg_xi2,
                R_Hu_xi4,    R_Hd_xi4,     R_Hg_xi4,
                R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea, bexp_Hg]
    
    Para_all = ParaManager_Unp(Paralst)
    # PDF_H_pred = np.array(list(pool.map(partial(PDF_theo, Para = Para_all), np.array(PDF_data_H))))
    PDF_H_pred = PDF_theo(PDF_data_H, Para=Para_all)
    cost_PDF_H = np.sum(((PDF_H_pred - PDF_data_H['f'])/ PDF_data_H['delta f']) ** 2 )
    
    PDF_data_Hpred = PDF_data_H.copy()
    PDF_data_Hpred['f pred'] = PDF_H_pred
    PDF_data_Hpred.to_csv(os.path.join(dir_path,'GUMP_Results/PDFcomp3.csv'),index= False)
    
    # tPDF_H_pred = np.array(list(pool.map(partial(tPDF_theo, Para = Para_all), np.array(tPDF_data_H))))
    tPDF_H_pred = tPDF_theo(tPDF_data_H, Para=Para_all)
    cost_tPDF_H = np.sum(((tPDF_H_pred - tPDF_data_H['f'])/ tPDF_data_H['delta f']) ** 2 )

    tPDF_data_Hpred = tPDF_data_H.copy()
    tPDF_data_Hpred['f pred'] = tPDF_H_pred
    tPDF_data_Hpred.to_csv(os.path.join(dir_path,'GUMP_Results/tPDFcomp3.csv'),index= False)
    # GFF_H_pred = np.array(list(pool.map(partial(GFF_theo, Para = Para_all), np.array(GFF_data_H))))
    GFF_H_pred = GFF_theo(GFF_data_H, Para=Para_all)
    cost_GFF_H = np.sum(((GFF_H_pred - GFF_data_H['f'])/ GFF_data_H['delta f']) ** 2 )

    GFF_data_Hpred = GFF_data_H.copy()
    GFF_data_Hpred['fpred'] = GFF_H_pred
    GFF_data_Hpred.to_csv(os.path.join(dir_path,'GUMP_Results/GFFcomp3.csv'),index= False)
    
    return cost_PDF_H/len(PDF_H_pred), cost_GFF_H/len(GFF_H_pred), cost_tPDF_H/len(tPDF_H_pred)

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

    #cost_DVjpsiPH1_xBtQ = np.array(list(pool.map(partial(DVMPxsec_cost_xBtQ, Para_Unp = Para_Unp_all, xsec_norm = jpsinorm, meson = 3, p_order = 2), DVJpsiPH1xsec_group_data)))
    #cost_DVjpsiPH1xsec = np.sum(cost_DVjpsiPH1_xBtQ)

    cost_DVrhoPH1_xBtQ = np.array(list(pool.map(partial(DVMPxsec_cost_xBtQ, Para_Unp = Para_Unp_all, xsec_norm = 1, meson = 1, p_order = 2), DVrhoPH1xsecL_group_data)))
    cost_DVrhoPZEUS_xBtQ = np.array(list(pool.map(partial(DVMPxsec_cost_xBtQ, Para_Unp = Para_Unp_all, xsec_norm = 1, meson = 1, p_order = 2), DVrhoPZEUSxsecL_group_data)))
   
    cost_DVrhoPH1_Lxsec = np.sum(cost_DVrhoPH1_xBtQ)
    cost_DVrhoPZEUS_Lxsec = np.sum(cost_DVrhoPZEUS_xBtQ)


    return cost_PDF_H_g_smallx + cost_DVrhoPH1_Lxsec + cost_DVrhoPZEUS_Lxsec

if __name__ == '__main__':
    pool = Pool()
    time_start = time.time()

    Paralst_Unp=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Unp.csv'), header=None).to_numpy()[0]
    Paralst_Pol=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Pol.csv'), header=None).to_numpy()[0]
    
    [Norm_HuV_Init,    alpha_HuV_Init,    beta_HuV_Init,    alphap_HuV_Init, Invm2_HuV_Init,
     Norm_Hubar_Init,  alpha_Hubar_Init,  beta_Hubar_Init,  alphap_Hqbar_Init,
     Norm_Hubar_2_Init,  alpha_Hubar_2_Init,  beta_Hubar_2_Init,
     Norm_HdV_Init,    alpha_HdV_Init,    beta_HdV_Init,    alphap_HdV_Init, Invm2_HdV_Init,
     Norm_Hdbar_Init,  alpha_Hdbar_Init,  beta_Hdbar_Init, 
     Norm_Hdbar_2_Init,  alpha_Hdbar_2_Init,  beta_Hdbar_2_Init, 
     Norm_Hg_Init,     alpha_Hg_Init,     beta_Hg_Init,     alphap_Hg_Init, Invm2_Hg_Init,
     Norm_Hg_2_Init,     alpha_Hg_2_Init,     beta_Hg_2_Init,
     Norm_EuV_Init,    alpha_EuV_Init,    beta_EuV_Init,    alphap_EuV_Init,
     Norm_EdV_Init,    R_E_Sea_Init,      R_Hu_xi2_Init,    R_Hd_xi2_Init,    R_Hg_xi2_Init,
     R_Eu_xi2_Init,    R_Ed_xi2_Init,     R_Eg_xi2_Init,
     R_Hu_xi4_Init,    R_Hd_xi4_Init,     R_Hg_xi4_Init,
     R_Eu_xi4_Init,    R_Ed_xi4_Init,     R_Eg_xi4_Init,    bexp_HSea_Init, bexp_Hg_Init] = Paralst_Unp

    print(cost_forward_H(Norm_HuV = Norm_HuV_Init,     alpha_HuV = alpha_HuV_Init,      beta_HuV = beta_HuV_Init,     alphap_HuV = alphap_HuV_Init, Invm2_HuV = Invm2_HuV_Init,
                    Norm_Hubar = Norm_Hubar_Init, alpha_Hubar = alpha_Hubar_Init,  beta_Hubar = beta_Hubar_Init, alphap_Hqbar = alphap_Hqbar_Init,
                    Norm_Hubar_2 = Norm_Hubar_2_Init, alpha_Hubar_2 = alpha_Hubar_2_Init,  beta_Hubar_2 = beta_Hubar_2_Init,
                    Norm_HdV = Norm_HdV_Init,     alpha_HdV = alpha_HdV_Init,      beta_HdV = beta_HdV_Init,     alphap_HdV = alphap_HdV_Init, Invm2_HdV = Invm2_HdV_Init,
                    Norm_Hdbar = Norm_Hdbar_Init, alpha_Hdbar = alpha_Hdbar_Init,  beta_Hdbar = beta_Hdbar_Init, 
                    Norm_Hdbar_2 = Norm_Hdbar_2_Init, alpha_Hdbar_2 = alpha_Hdbar_2_Init,  beta_Hdbar_2 = beta_Hdbar_2_Init, 
                    Norm_Hg = Norm_Hg_Init,       alpha_Hg = alpha_Hg_Init,        beta_Hg = beta_Hg_Init,       alphap_Hg = alphap_Hg_Init, Invm2_Hg = Invm2_Hg_Init,
                    Norm_Hg_2 = Norm_Hg_2_Init,       alpha_Hg_2 = alpha_Hg_2_Init,        beta_Hg_2 = beta_Hg_2_Init,
                    Norm_EuV = Norm_EuV_Init,     alpha_EuV = alpha_EuV_Init,      beta_EuV = beta_EuV_Init,     alphap_EuV = alphap_EuV_Init, 
                    Norm_EdV = Norm_EdV_Init,     R_E_Sea = R_E_Sea_Init,          R_Hu_xi2 = R_Hu_xi2_Init,     R_Hd_xi2 = R_Hd_xi2_Init,     R_Hg_xi2 = R_Hg_xi2_Init,
                    R_Eu_xi2 = R_Eu_xi2_Init,     R_Ed_xi2 = R_Ed_xi2_Init,        R_Eg_xi2 = R_Eg_xi2_Init,
                    R_Hu_xi4 = R_Hu_xi4_Init,     R_Hd_xi4 = R_Hd_xi4_Init,        R_Hg_xi4 = R_Hg_xi4_Init,
                    R_Eu_xi4 = R_Eu_xi4_Init,     R_Ed_xi4 = R_Ed_xi4_Init,        R_Eg_xi4 = R_Eg_xi4_Init,     bexp_HSea = bexp_HSea_Init,   bexp_Hg = bexp_Hg_Init))

    '''
    fit_forward_Ht  = forward_Ht_fit(Paralst_Pol)
    Paralst_Pol     = np.array(fit_forward_Ht.values)

    fit_forward_E   = forward_E_fit(Paralst_Unp)
    Paralst_Unp     = np.array(fit_forward_E.values)

    fit_forward_Et  = forward_Et_fit(Paralst_Pol)
    Paralst_Pol     = np.array(fit_forward_Et.values)
    
    fit_off_forward = off_forward_fit(Paralst_Unp, Paralst_Pol)
    '''
    '''
    Paralst_Unp_Ext2 = np.concatenate((Paralst_Unp, np.array([norm1,norm2])))
    fit_dvmp = dvmp_fit(Paralst_Unp_Ext2)
    '''

