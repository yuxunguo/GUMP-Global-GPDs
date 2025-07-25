from Parameters import ParaManager_Unp, ParaManager_Pol
from Observables import GPDobserv
from DVCS_xsec import dsigma_DVCS_TOT, Asymmetry_DVCS_TOT, dsigma_DVCS_HERA, M
from DVMP_xsec import dsigma_DVMP_dt,dsigmaL_DVMP_dt, M_jpsi,epsilon, R_fitted
from multiprocessing import Pool
from functools import partial
from iminuit import Minuit
import numpy as np
import pandas as pd
import time
import csv
import os
from config import Export_Mode

dir_path = os.path.dirname(os.path.realpath(__file__))
Minuit_Counter = 0
Time_Counter = 1
Q_threshold = 1.9
xB_Cut = 0.5
xB_small_Cut = 0.0001

Paralst_Unp_Names = [
    "Norm_HuV", "alpha_HuV", "beta_HuV", "alphap_HuV", "Invm2_HuV",
    "Norm_Hubar", "alpha_Hubar", "beta_Hubar", "alphap_Hqbar",
    "Norm_Hubar_2", "alpha_Hubar_2", "beta_Hubar_2",
    "Norm_HdV", "alpha_HdV", "beta_HdV", "alphap_HdV", "Invm2_HdV",
    "Norm_Hdbar", "alpha_Hdbar", "beta_Hdbar",
    "Norm_Hdbar_2", "alpha_Hdbar_2", "beta_Hdbar_2",
    "Norm_Hg", "alpha_Hg", "beta_Hg", "alphap_Hg", "Invm2_Hg",
    "Norm_Hg_2", "alpha_Hg_2", "beta_Hg_2",
    "Norm_EuV", "alpha_EuV", "beta_EuV", "alphap_EuV",
    "Norm_EdV", "R_E_Sea", "R_Hu_xi2", "R_Hd_xi2", "R_Hg_xi2",
    "R_Eu_xi2", "R_Ed_xi2", "R_Eg_xi2",
    "R_Hu_xi4", "R_Hd_xi4", "R_Hg_xi4",
    "R_Eu_xi4", "R_Ed_xi4", "R_Eg_xi4", "bexp_HSea", "bexp_Hg"
]

Paralst_Pol_Names = [
    "Norm_HtuV", "alpha_HtuV", "beta_HtuV", "alphap_HtuV",
    "Norm_Htubar", "alpha_Htubar", "beta_Htubar", "alphap_Htqbar",
    "Norm_HtdV", "alpha_HtdV", "beta_HtdV", "alphap_HtdV",
    "Norm_Htdbar", "alpha_Htdbar", "beta_Htdbar",
    "Norm_Htg", "alpha_Htg", "beta_Htg", "alphap_Htg",
    "Norm_EtuV", "alpha_EtuV", "beta_EtuV", "alphap_EtuV",
    "Norm_EtdV", "R_Et_Sea", "R_Htu_xi2", "R_Htd_xi2", "R_Htg_xi2",
    "R_Etu_xi2", "R_Etd_xi2", "R_Etg_xi2",
    "R_Htu_xi4", "R_Htd_xi4", "R_Htg_xi4",
    "R_Etu_xi4", "R_Etd_xi4", "R_Etg_xi4", "bexp_HtSea"
]

def validate_params(params: dict, required_names: set):
    param_keys = set(params.keys())

    missing = required_names - param_keys
    extra = param_keys - required_names
    none_values = [k for k in required_names if params.get(k) is None]

    errors = []
    if missing:
        errors.append(f"Missing parameters: {sorted(missing)}")
    if extra:
        errors.append(f"Extra parameters: {sorted(extra)}")
    if none_values:
        errors.append(f"Parameters with None values: {sorted(none_values)}")

    if errors:
        raise ValueError("; ".join(errors))

First_Write_Flag = {}

def Export_Frame_Append(df, filename):

    os.makedirs(os.path.join(dir_path, 'GUMP_Results'), exist_ok=True)
    
    global First_Write_Flag

    first_write = First_Write_Flag.get(filename, True)

    mode = 'w' if first_write else 'a'
    header = first_write

    df.to_csv(os.path.join(dir_path,'GUMP_Results',filename), mode=mode, index=False, header=header)

    # Mark file as written
    First_Write_Flag[filename] = False
    
"""
************************ PDF and tPDFs data preprocessing ****************************
"""

PDF_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/PDFdata.csv'), header = 0, names = ['x', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'x': float, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})
PDF_data_H  = PDF_data[PDF_data['spe'] == 0]
PDF_data_E  = PDF_data[PDF_data['spe'] == 1]
PDF_data_Ht = PDF_data[PDF_data['spe'] == 2]
PDF_data_Et = PDF_data[PDF_data['spe'] == 3]

tPDF_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/tPDFdata.csv'),     header = 0, names = ['x', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'x': float, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})

"""
************************ GFF data preprocessing ****************************
"""

GFF_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/GFFdata_Quark.csv'),       header = 0, names = ['j', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'j': int, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})

GFF_Gluon_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/GFFdata_Gluon.csv'),       header = None, names = ['j', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'j': int, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})
GFF_Gluon_data_H  = GFF_Gluon_data[GFF_Gluon_data['spe'] == 0]
GFF_Gluon_data_E  = GFF_Gluon_data[GFF_Gluon_data['spe'] == 1]
GFF_Gluon_data_Ht = GFF_Gluon_data[GFF_Gluon_data['spe'] == 2]
GFF_Gluon_data_Et = GFF_Gluon_data[GFF_Gluon_data['spe'] == 3]

"""
************************ DVCS data preprocessing ****************************
"""

DVCSxsec_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVCSxsec_Old.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'phi', 'f', 'delta f', 'pol'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'phi': float, 'f': float, 'delta f': float, 'pol': str})
DVCSxsec_data_invalid = DVCSxsec_data[DVCSxsec_data['t']*(DVCSxsec_data['xB']-1) - M ** 2 * DVCSxsec_data['xB'] ** 2 < 0]
DVCSxsec_data = DVCSxsec_data[(DVCSxsec_data['Q'] > Q_threshold) & (DVCSxsec_data['xB'] < xB_Cut) & (DVCSxsec_data['t']*(DVCSxsec_data['xB']-1) - M ** 2 * DVCSxsec_data['xB'] ** 2 > 0) & (DVCSxsec_data['delta f']>0) & ((DVCSxsec_data['f']>0) | (DVCSxsec_data['pol']!='UU'))]
xBtQlst = DVCSxsec_data.drop_duplicates(subset = ['xB', 't', 'Q'], keep = 'first')[['xB','t','Q']].values.tolist()
DVCSxsec_group_data = list(map(lambda set: DVCSxsec_data[(DVCSxsec_data['xB'] == set[0]) & (DVCSxsec_data['t'] == set[1]) & ((DVCSxsec_data['Q'] == set[2]))], xBtQlst))

DVCSxsec_HERA_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVCSxsec_HERA.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f', 'pol'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float, 'pol': str})
DVCSxsec_HERA_data_invalid = DVCSxsec_HERA_data[DVCSxsec_HERA_data['t']*(DVCSxsec_HERA_data['xB']-1) - M ** 2 * DVCSxsec_HERA_data['xB'] ** 2 < 0]
DVCSxsec_HERA_data = DVCSxsec_HERA_data[(DVCSxsec_HERA_data['Q'] > Q_threshold) & (DVCSxsec_HERA_data['xB'] < xB_Cut) & (DVCSxsec_HERA_data['t']*(DVCSxsec_HERA_data['xB']-1) - M ** 2 * DVCSxsec_HERA_data['xB'] ** 2 > 0)]
xBtQlst_HERA = DVCSxsec_HERA_data.drop_duplicates(subset = ['xB', 't', 'Q'], keep = 'first')[['xB','t','Q']].values.tolist()
DVCSxsec_HERA_group_data = list(map(lambda set: DVCSxsec_HERA_data[(DVCSxsec_HERA_data['xB'] == set[0]) & (DVCSxsec_HERA_data['t'] == set[1]) & ((DVCSxsec_HERA_data['Q'] == set[2]))], xBtQlst_HERA))

DVCSAsym_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVCSAsym.csv'), header = 0, names = ['y', 'xB', 't', 'Q', 'phi', 'f', 'delta f', 'pol'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'phi': float, 'f': float, 'delta f': float, 'pol': str})
DVCSAsym_data_invalid = DVCSAsym_data[DVCSAsym_data['t']*(DVCSAsym_data['xB']-1) - M ** 2 * DVCSAsym_data['xB'] ** 2 < 0]
DVCSAsym_data = DVCSAsym_data[(DVCSAsym_data['Q'] > Q_threshold) & (DVCSAsym_data['xB'] < xB_Cut) & (DVCSAsym_data['t']*(DVCSAsym_data['xB']-1) - M ** 2 * DVCSAsym_data['xB'] ** 2 > 0) & DVCSAsym_data['delta f']>0]
AsymxBtQlst = DVCSAsym_data.drop_duplicates(subset = ['xB', 't', 'Q'], keep = 'first')[['xB','t','Q']].values.tolist()
DVCSAsym_group_data = list(map(lambda set: DVCSAsym_data[(DVCSAsym_data['xB'] == set[0]) & (DVCSAsym_data['t'] == set[1]) & ((DVCSAsym_data['Q'] == set[2]))], AsymxBtQlst))

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
    
    PDF_input = PDF_input.copy()
    
    xs = PDF_input['x'].to_numpy()
    ts = PDF_input['t'].to_numpy()
    Qs = PDF_input['Q'].to_numpy()
    flvs = PDF_input['flv'].to_numpy()
    spes = PDF_input['spe'].to_numpy()
    ps = np.where(spes <= 1, 1, -1)
    Para_spe = Para[spes]
    
    xis = np.zeros_like(xs)
    
    # Prepare input arguments for parallel computation
    args = [(x_i, xi_i, t_i, Q_i, p_i, flv_i, Para_i, p_order) 
            for x_i, xi_i, t_i, Q_i, p_i, flv_i, Para_i 
            in zip(xs, xis, ts, Qs, ps, flvs, Para_spe)]
    
    # Use multiprocessing Pool to parallelize the computation
    PDF_input['pred f'] = list(pool.map(PDF_theo_scalar_helper, args))
    PDF_input['cost'] = ((PDF_input["pred f"]-PDF_input["f"])/PDF_input["delta f"])**2
    
    return PDF_input

tPDF_theo = PDF_theo

# Helper function for scalar computation
def GFF_theo_scalar_helper(args):

    j_i, x, xi, t_i, Q_i, p_i, flv_i, Para_i, p_order = args
    _GFF_theo = GPDobserv(x, xi, t_i, Q_i, p_i)
    return _GFF_theo.GFFj0(j_i, flv_i, Para_i, p_order)

def GFF_theo(GFF_input: pd.DataFrame, Para: np.array, p_order = 2):
    
    GFF_input = GFF_input.copy()
    
    js = GFF_input['j'].to_numpy()
    ts = GFF_input['t'].to_numpy()
    Qs = GFF_input['Q'].to_numpy()
    flvs = GFF_input['flv'].to_numpy()
    spes = GFF_input['spe'].to_numpy()
    ps = np.where(spes <= 1, 1, -1)
    Para_spe = Para[spes]
    
    # Constants
    x = 0
    xi = 0
    
    # Prepare input arguments for parallel computation
    args = [(j_i, x, xi, t_i, Q_i, p_i, flv_i, Para_i, p_order) 
            for j_i, t_i, Q_i, p_i, flv_i, Para_i 
            in zip(js, ts, Qs, ps, flvs, Para_spe)]

    GFF_input['pred f'] = list(pool.map(GFF_theo_scalar_helper, args))
    GFF_input['cost'] = ((GFF_input["pred f"]-GFF_input["f"])/GFF_input["delta f"])**2
    
    return GFF_input

def CFF_theo(xB, t, Q, Para_Unp, Para_Pol, porder = 2):
    x = 0
    xi = (1/(2 - xB) - (2*t*(-1 + xB))/(Q**2*(-2 + xB)**2))*xB
    H_E = GPDobserv(x, xi, t, Q, 1)
    Ht_Et = GPDobserv(x, xi, t, Q, -1)
    HCFF = H_E.CFF(Para_Unp[..., 0, :, :, :, :], Q, p_order = porder)
    ECFF = H_E.CFF(Para_Unp[..., 1, :, :, :, :], Q, p_order = porder)
    HtCFF = Ht_Et.CFF(Para_Pol[..., 0, :, :, :, :], Q, p_order = porder)
    EtCFF = Ht_Et.CFF(Para_Pol[..., 1, :, :, :, :], Q, p_order = porder)

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

def DVCSxsec_cost_xBtQ(DVCSxsec_data_xBtQ: pd.DataFrame, Para_Unp, Para_Pol, P_order = 2):
    [xB, t, Q] = [DVCSxsec_data_xBtQ['xB'].iat[0], DVCSxsec_data_xBtQ['t'].iat[0], DVCSxsec_data_xBtQ['Q'].iat[0]] 
    [HCFF, ECFF, HtCFF, EtCFF] = CFF_theo(xB, t, Q, Para_Unp, Para_Pol, porder= P_order) # scalar for each of them
    DVCS_pred_xBtQ = DVCSxsec_theo(DVCSxsec_data_xBtQ, CFF_input = [HCFF, ECFF, HtCFF, EtCFF])
    DVCS_cost_xBtQ = ((DVCS_pred_xBtQ - DVCSxsec_data_xBtQ['f'])/ DVCSxsec_data_xBtQ['delta f']) ** 2
    
    DVCSxsec_data_xBtQ['pred f'] = DVCS_pred_xBtQ
    DVCSxsec_data_xBtQ['cost'] = DVCS_cost_xBtQ
        
    return DVCSxsec_data_xBtQ

def DVCSAsym_theo(DVCSAsym_input: pd.DataFrame, CFF_input: np.array):
    # CFF_input is a list of np.arrays
    # [y, xB, t, Q, phi, f, delta_f, pol] = DVCSxsec_input    

    y = DVCSAsym_input['y'].to_numpy()
    xB = DVCSAsym_input['xB'].to_numpy()
    t = DVCSAsym_input['t'].to_numpy()
    Q = DVCSAsym_input['Q'].to_numpy()
    phi = DVCSAsym_input['phi'].to_numpy()
    pol = DVCSAsym_input['pol'].to_numpy()

    [HCFF, ECFF, HtCFF, EtCFF] = CFF_input # each of them have shape (N); scalar is also OK if we use 
    return Asymmetry_DVCS_TOT(y, xB, t, Q, phi, pol, HCFF, ECFF, HtCFF, EtCFF)

def DVCSAsym_cost_xBtQ(DVCSAsym_data_xBtQ: pd.DataFrame, Para_Unp, Para_Pol, P_order = 2):
    [xB, t, Q] = [DVCSAsym_data_xBtQ['xB'].iat[0], DVCSAsym_data_xBtQ['t'].iat[0], DVCSAsym_data_xBtQ['Q'].iat[0]] 
    [HCFF, ECFF, HtCFF, EtCFF] = CFF_theo(xB, t, Q, Para_Unp, Para_Pol, porder= P_order) # scalar for each of them
    DVCS_Asym_pred_xBtQ = DVCSAsym_theo(DVCSAsym_data_xBtQ, CFF_input = [HCFF, ECFF, HtCFF, EtCFF])
    DVCS_Asym_cost_xBtQ = ((DVCS_Asym_pred_xBtQ - DVCSAsym_data_xBtQ['f'])/ DVCSAsym_data_xBtQ['delta f']) ** 2

    DVCSAsym_data_xBtQ['pred f'] = DVCS_Asym_pred_xBtQ
    DVCSAsym_data_xBtQ['cost'] = DVCS_Asym_cost_xBtQ
    
    return DVCSAsym_data_xBtQ

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

def DVCSxsec_HERA_cost_xBtQ(DVCSxsec_HERA_data_xBtQ: pd.DataFrame, Para_Unp, Para_Pol , P_order = 2):
    
    [xB, t, Q] = [DVCSxsec_HERA_data_xBtQ['xB'].iat[0], DVCSxsec_HERA_data_xBtQ['t'].iat[0], DVCSxsec_HERA_data_xBtQ['Q'].iat[0]] 
    [HCFF, ECFF, HtCFF, EtCFF] = CFF_theo(xB, t, Q, Para_Unp, Para_Pol, porder = P_order) # scalar for each of them
    DVCS_HERA_pred_xBtQ = DVCSxsec_HERA_theo(DVCSxsec_HERA_data_xBtQ, CFF_input = [HCFF, ECFF, HtCFF, EtCFF])
    DVCS_HERA_cost_xBtQ = ((DVCS_HERA_pred_xBtQ - DVCSxsec_HERA_data_xBtQ['f'])/ DVCSxsec_HERA_data_xBtQ['delta f']) ** 2
    
    DVCSxsec_HERA_data_xBtQ['pred f'] = DVCS_HERA_pred_xBtQ
    DVCSxsec_HERA_data_xBtQ['cost'] = DVCS_HERA_cost_xBtQ

    return DVCSxsec_HERA_data_xBtQ

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
    DVMP_cost_xBtQ = ((DVMP_pred_xBtQ - DVMPxsec_data_xBtQ['f'])/ DVMPxsec_data_xBtQ['delta f']) ** 2

    DVMPxsec_data_xBtQ['pred f'] = DVMP_pred_xBtQ
    DVMPxsec_data_xBtQ['cost'] = DVMP_cost_xBtQ
        
    return DVMPxsec_data_xBtQ

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

    params = locals()
    validate_params(params, set(Paralst_Unp_Names))
    Para_Unp_lst = [params[name] for name in Paralst_Unp_Names]
    Para_Unp_all = ParaManager_Unp(Para_Unp_lst)
    # This is just to match the shape, the 2nd Para_Unp_all corresponds to species = 2 and 3 should never be called for H fit
    Para_Comb = np.concatenate([Para_Unp_all, Para_Unp_all], axis=0)
    
    global Minuit_Counter, Time_Counter

    time_now = time.time() - time_start
    
    if(time_now > Time_Counter * 600):
        print('Runing Time:',round(time_now/60),'minutes. Cost function called total', Minuit_Counter, 'times.')
        Time_Counter = Time_Counter + 1
    
    Minuit_Counter = Minuit_Counter + 1
    
    PDF_H_pred = PDF_theo(PDF_data_H, Para=Para_Comb)

    if (Export_Mode == True):
        Export_Frame_Append(PDF_H_pred,"PDFcomp.csv")
        
        return PDF_H_pred['cost'].sum()/len(PDF_H_pred.index)
        
    return PDF_H_pred['cost'].sum() 

def forward_H_fit(Paralst_Unp):
    
    assert Export_Mode == False, "Make sure the Export_Mode is set to False in config.py before fitting"
    
    params_unp = dict(zip(Paralst_Unp_Names, Paralst_Unp))
    fit_forw_H = Minuit(cost_forward_H, **params_unp)
    
    fit_forw_H.errordef = 1

    fit_forw_H.limits['alpha_HuV'] = (-2, 1.2)
    fit_forw_H.limits['alpha_Hubar'] = (0, 1.2)
    fit_forw_H.limits['alpha_HdV'] = (-2, 1.2)
    fit_forw_H.limits['alpha_Hdbar'] = (0, 1.2)
    fit_forw_H.limits['alpha_Hg'] = (0, 1.2)
    fit_forw_H.limits['alpha_EuV'] = (-2, 1.2)

    # make the second set of parameters 'valence-like' bounded by small x <~ x **(-0.8)
    fit_forw_H.limits['alpha_Hubar_2'] = (0, 0.8)
    fit_forw_H.limits['alpha_Hdbar_2'] = (0, 0.8)
    fit_forw_H.limits['alpha_Hg_2'] = (0, 0.8)

    fit_forw_H.limits['beta_HuV'] = (0, 20)
    fit_forw_H.limits['beta_Hubar'] = (0, 20)
    fit_forw_H.limits['beta_HdV'] = (0, 20)
    fit_forw_H.limits['beta_Hdbar'] = (0, 20)
    fit_forw_H.limits['beta_Hg'] = (0, 20)    
    fit_forw_H.limits['beta_EuV'] = (0, 20)
    
    fit_forw_H.limits['beta_Hubar_2'] = (0, 20)
    fit_forw_H.limits['beta_Hdbar_2'] = (0, 20)
    fit_forw_H.limits['beta_Hg_2'] = (0, 20)  
    
    # All E GPD and off-forward parameters are fixed
    fixed_params_H = [ 'alphap_HuV', 'Invm2_HuV', 'alphap_HdV', 'Invm2_HdV',
                        'alphap_Hqbar', 'alphap_Hg', 'Invm2_Hg',
                        'Norm_EuV', 'alpha_EuV', 'beta_EuV', 'alphap_EuV',
                        'Norm_EdV', 'R_E_Sea', 'R_Hu_xi2', 'R_Hd_xi2', 'R_Hg_xi2',
                        'R_Eu_xi2', 'R_Ed_xi2', 'R_Eg_xi2',
                        'R_Hu_xi4', 'R_Hd_xi4', 'R_Hg_xi4',
                        'R_Eu_xi4', 'R_Ed_xi4', 'R_Eg_xi4', 'bexp_HSea', 'bexp_Hg',
                    ]
    for param in fixed_params_H:
        fit_forw_H.fixed[param] = True
        
    global Minuit_Counter, Time_Counter, time_start
    Minuit_Counter = 0
    Time_Counter = 1
    time_start = time.time()

    fit_forw_H.migrad()
    fit_forw_H.hesse()
    
    print("H fit finished...")
    
    ndof_H = len(PDF_data_H.index)  - fit_forw_H.nfit 

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

    with open(os.path.join(dir_path,"GUMP_Params/Para_Unp.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerow(list([*fit_forw_H.values]))
        csvWriter.writerow(list([*fit_forw_H.errors]))
        print("H fit parameters saved to Para_Unp.csv")
        
    return fit_forw_H

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

    params = locals()
    validate_params(params, set(Paralst_Pol_Names))
    Para_Pol_lst = [params[name] for name in Paralst_Pol_Names]
    Para_Pol_all = ParaManager_Pol(Para_Pol_lst)
    # This is just to match the shape, the 1st Para_Unp_all corresponds to species = 0 and 1 should never be called for Ht fit
    Para_Comb = np.concatenate([Para_Pol_all, Para_Pol_all], axis=0)
    
    PDF_Ht_pred = PDF_theo(PDF_data_Ht, Para=Para_Comb)

    if (Export_Mode == True):
        Export_Frame_Append(PDF_Ht_pred,"PDFcomp.csv")
        
        return PDF_Ht_pred['cost'].sum()/len(PDF_Ht_pred.index)

    return PDF_Ht_pred['cost'].sum()

def forward_Ht_fit(Paralst_Pol):
    
    assert Export_Mode == False, "Make sure the Export_Mode is set to False in config.py before fitting"
    
    params_pol = dict(zip(Paralst_Pol_Names, Paralst_Pol))
    fit_forw_Ht = Minuit(cost_forward_Ht, **params_pol)
    
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

    fixed_params_Ht = ['alphap_HtuV', 'alphap_HtdV','alphap_Htqbar', 'alphap_Htg',
                        'Norm_EtuV', 'alpha_EtuV', 'beta_EtuV', 'alphap_EtuV',
                        'Norm_EtdV', 'R_Et_Sea', 'R_Htu_xi2', 'R_Htd_xi2', 'R_Htg_xi2',
                        'R_Etu_xi2', 'R_Etd_xi2', 'R_Etg_xi2',
                        'R_Htu_xi4', 'R_Htd_xi4', 'R_Htg_xi4', 
                        'R_Etu_xi4', 'R_Etd_xi4', 'R_Etg_xi4', 'bexp_HtSea',
                        ]
    for param in fixed_params_Ht:
        fit_forw_Ht.fixed[param] = True
    
    global time_start 
    time_start = time.time()
    
    fit_forw_Ht.migrad()
    fit_forw_Ht.hesse()

    print("Ht fit finished...")
    
    ndof_Ht = len(PDF_data_Ht.index)  - fit_forw_Ht.nfit

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

    with open(os.path.join(dir_path,"GUMP_Params/Para_Pol.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerow(list([*fit_forw_Ht.values]))
        csvWriter.writerow(list([*fit_forw_Ht.errors]))
        print("Ht fit parameters saved to Para_Pol.csv")
        
    return fit_forw_Ht

def cost_off_forward(Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV,   Invm2_HuV,
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
                    R_Eu_xi4,    R_Ed_xi4,     R_Eg_xi4,    bexp_HSea, bexp_Hg,
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
    
    global Export_Mode
    
    params = locals()
    validate_params(params, set(Paralst_Unp_Names + Paralst_Pol_Names))
    Para_Unp_lst = [params[name] for name in Paralst_Unp_Names]
    Para_Pol_lst = [params[name] for name in Paralst_Pol_Names]
    
    global Minuit_Counter, Time_Counter

    time_now = time.time() - time_start
    
    if(time_now > Time_Counter * 600):
        print('Runing Time:',round(time_now/60),'minutes. Cost function called total', Minuit_Counter, 'times.')
        Time_Counter = Time_Counter + 1
    
    Minuit_Counter = Minuit_Counter + 1
    
    Para_Unp_all = ParaManager_Unp(Para_Unp_lst)
    Para_Pol_all = ParaManager_Pol(Para_Pol_lst)
    Para_Comb = np.concatenate([Para_Unp_all, Para_Pol_all], axis=0)
    
    DVCS_pred_xBtQ = pd.concat(list(pool.map(partial(DVCSxsec_cost_xBtQ, Para_Unp = Para_Unp_all, Para_Pol = Para_Pol_all, P_order = 2), DVCSxsec_group_data)), ignore_index=True)
    DVCS_HERA_pred_xBtQ = pd.concat(list(pool.map(partial(DVCSxsec_HERA_cost_xBtQ, Para_Unp = Para_Unp_all, Para_Pol = Para_Pol_all, P_order = 2), DVCSxsec_HERA_group_data)), ignore_index=True)
    DVrhoPH1_pred_xBtQ = pd.concat(list(pool.map(partial(DVMPxsec_cost_xBtQ, Para_Unp = Para_Unp_all, xsec_norm = 1, meson = 1, p_order = 2), DVrhoPH1xsecL_group_data)), ignore_index=True)
    DVrhoPZEUS_pred_xBtQ = pd.concat(list(pool.map(partial(DVMPxsec_cost_xBtQ, Para_Unp = Para_Unp_all, xsec_norm = 1, meson = 1, p_order = 2), DVrhoPZEUSxsecL_group_data)), ignore_index=True)
    
    tPDF_pred = tPDF_theo(tPDF_data, Para=Para_Comb)
    GFF_pred = GFF_theo(GFF_data, Para=Para_Comb)
    
    #DVCS_Asym_pred_xBtQ = pd.concat(list(pool.map(partial(DVCSAsym_cost_xBtQ, Para_Unp = Para_Unp_all, Para_Pol = Para_Pol_all, P_order = 2), DVCSAsym_group_data)), ignore_index=True)

    if (Export_Mode == True):
        Export_Frame_Append(DVCS_pred_xBtQ,"DVCSxsec.csv")
        Export_Frame_Append(DVCS_HERA_pred_xBtQ,"DVCSxsec_HERA.csv")
        Export_Frame_Append(DVrhoPH1_pred_xBtQ,"DVMPxsec.csv")
        Export_Frame_Append(DVrhoPZEUS_pred_xBtQ,"DVMPxsec.csv")
        
        Export_Frame_Append(tPDF_pred,"tPDFcomp.csv")
        Export_Frame_Append(GFF_pred,"GFFcomp.csv")
        
        return [DVCS_pred_xBtQ['cost'].sum()/len(DVCS_pred_xBtQ.index), DVCS_HERA_pred_xBtQ['cost'].sum()/len(DVCS_HERA_pred_xBtQ.index),
                DVrhoPH1_pred_xBtQ['cost'].sum()/len(DVrhoPH1_pred_xBtQ.index), DVrhoPZEUS_pred_xBtQ['cost'].sum()/len(DVrhoPZEUS_pred_xBtQ.index), # + cost_DVCSAsym
                tPDF_pred['cost'].sum()/len(tPDF_pred.index), GFF_pred['cost'].sum()/len(GFF_pred.index)]
        
    return  (DVCS_pred_xBtQ['cost'].sum() + DVCS_HERA_pred_xBtQ['cost'].sum() 
             + DVrhoPH1_pred_xBtQ['cost'].sum() + DVrhoPZEUS_pred_xBtQ['cost'].sum() # + cost_DVCSAsym
             + tPDF_pred['cost'].sum() + GFF_pred['cost'].sum())

def off_forward_fit(Paralst_Unp, Paralst_Pol):

    assert Export_Mode == False, "Make sure the Export_Mode is set to False in config.py before fitting"
    
    # Create dictionaries by zipping names and values
    params_unp = dict(zip(Paralst_Unp_Names, Paralst_Unp))
    params_pol = dict(zip(Paralst_Pol_Names, Paralst_Pol))
    params = {**params_unp, **params_pol}

    fit_off_forward = Minuit(cost_off_forward, **params)
    
    fit_off_forward.errordef = 1

    fit_off_forward.limits['alphap_HuV'] = (0,3)
    fit_off_forward.limits['Invm2_HuV'] = (0,3)
    fit_off_forward.limits['alphap_HdV'] = (0,3)
    fit_off_forward.limits['Invm2_HdV'] = (0,3)
    
    fit_off_forward.limits['alpha_EuV'] = (-2, 1.2)
    fit_off_forward.limits['alphap_EuV'] = (0,3)
    
    fit_off_forward.limits['bexp_Hg']  = (0, 10)
    fit_off_forward.limits['bexp_HSea']  = (0, 10)
    
    fit_off_forward.limits['alphap_HtuV'] = (0,3)
    fit_off_forward.limits['alphap_HtdV'] = (0,3)
    
    fit_off_forward.limits['alpha_EtuV'] = (-2, 0.8)
    fit_off_forward.limits['beta_EtuV'] = (0, 15)
    
    fit_off_forward.limits['bexp_HtSea'] = (0, 10)
    
    fixed_params = [
        "Norm_HuV", "alpha_HuV", "beta_HuV", 
        "Norm_Hubar", "alpha_Hubar", "beta_Hubar", "alphap_Hqbar",
        "Norm_Hubar_2", "alpha_Hubar_2", "beta_Hubar_2",
        "Norm_HdV", "alpha_HdV", "beta_HdV", 
        "Norm_Hdbar", "alpha_Hdbar", "beta_Hdbar",
        "Norm_Hdbar_2", "alpha_Hdbar_2", "beta_Hdbar_2",
        "Norm_Hg", "alpha_Hg", "beta_Hg", "alphap_Hg", "Invm2_Hg",
        "Norm_Hg_2", "alpha_Hg_2", "beta_Hg_2",
        
        "Norm_HtuV", "alpha_HtuV", "beta_HtuV",
        "Norm_Htubar", "alpha_Htubar", "beta_Htubar", "alphap_Htqbar",
        "Norm_HtdV", "alpha_HtdV", "beta_HtdV",
        "Norm_Htdbar", "alpha_Htdbar", "beta_Htdbar",
        "Norm_Htg", "alpha_Htg", "beta_Htg", "alphap_Htg",
        "R_Htg_xi4", "R_Etg_xi4", "R_Htu_xi4", "R_Etu_xi4", "R_Htd_xi4", "R_Etd_xi4",
        ]
    
    for param in fixed_params:
        fit_off_forward.fixed[param] = True
    
    global Minuit_Counter, Time_Counter, time_start
    Minuit_Counter = 0
    Time_Counter = 1
    time_start = time.time()
    
    fit_off_forward.migrad()
    fit_off_forward.hesse()
    
    print("off forward fit finished...")
    time_end = time.time() -time_start
    
    ndof_off_forward = (len(DVCSxsec_data.index) + len(DVCSxsec_HERA_data.index) 
                        + len(DVrhoPH1xsec_data.index) + len(DVrhoPZEUSxsec_data.index)
                         + len(tPDF_data.index) + len(GFF_data.index) - fit_off_forward.nfit)

    with open(os.path.join(dir_path,'GUMP_Output/off_forward_fit.txt'), 'w', encoding='utf-8') as f:
        print('Total running time: %.1f minutes. Total call of cost function: %3d.\n' % ( time_end/60, fit_off_forward.nfcn), file=f)
        print('The chi squared/d.o.f. is: %.2f / %3d ( = %.2f ).\n' % (fit_off_forward.fval, ndof_off_forward, fit_off_forward.fval/ndof_off_forward), file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*fit_off_forward.values, sep=", ", file = f)
        print(*fit_off_forward.errors, sep=", ", file = f)
        print(fit_off_forward.params, file = f)

    FitVals = list([*fit_off_forward.values])
    FitErrs = list([*fit_off_forward.errors])
    UnpLength = len(Paralst_Unp)
    
    with open(os.path.join(dir_path,"GUMP_Params/Para_Unp_Off_forward.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerow(FitVals[:UnpLength])
        csvWriter.writerow(FitErrs[:UnpLength])
        print("off-forward fit unpolarized parameters saved to Para_Unp_Off_forward.csv")
    
    with open(os.path.join(dir_path,"GUMP_Params/Para_Pol_Off_forward.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerow(FitVals[UnpLength:])
        csvWriter.writerow(FitErrs[UnpLength:])
        print("off-forward fit polarized parameters saved to Para_Pol_Off_forward.csv")

    with open(os.path.join(dir_path,"GUMP_Output/Off_forward_cov.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows([*fit_off_forward.covariance])
        
    return fit_off_forward

if __name__ == '__main__':
    pool = Pool()
    time_start = time.time()
    
    #'''
    Paralst_Unp=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Unp.csv'), header=None).to_numpy()[0]
    Paralst_Pol=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Pol.csv'), header=None).to_numpy()[0]
    
    Para_Unp = ParaManager_Unp(Paralst_Unp)
    Para_Pol = ParaManager_Pol(Paralst_Pol)
    
    fit_forward_H   = forward_H_fit(Paralst_Unp)
    Paralst_Unp     = np.array(fit_forward_H.values)

    fit_forward_Ht  = forward_Ht_fit(Paralst_Pol)
    Paralst_Pol     = np.array(fit_forward_Ht.values)

    Paralst_Unp=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Unp.csv'), header=None).to_numpy()[0]
    Paralst_Pol=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Pol.csv'), header=None).to_numpy()[0]
    fit_off_forward = off_forward_fit(Paralst_Unp, Paralst_Pol)
    #'''
    #
    # Below is for testing, set Export_Mode to True in config.py and run through to generate the outputs
    #
    
    '''
    Paralst_Unp=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Unp_Off_forward.csv'), header=None).to_numpy()[0]
    Paralst_Pol=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Pol_Off_forward.csv'), header=None).to_numpy()[0]
    #Paralst_Unp=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Unp.csv'), header=None).to_numpy()[0]
    #Paralst_Pol=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Pol.csv'), header=None).to_numpy()[0]
    
    params_unp = dict(zip(Paralst_Unp_Names, Paralst_Unp))
    params_pol = dict(zip(Paralst_Pol_Names, Paralst_Pol))
    params = {**params_unp, **params_pol}

    print(cost_forward_H(**params_unp))
    print(cost_forward_Ht(**params_pol))
    print(cost_off_forward(**params))
    '''