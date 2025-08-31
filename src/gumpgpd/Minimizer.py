from .Parameters import ParaManager_Unp, ParaManager_Pol
from .Observables import GPDobserv
from .DVCS_xsec import dsigma_DVCS_TOT, Asymmetry_DVCS_TOT, dsigma_DVCS_HERA, M
from .DVMP_xsec import dsigma_DVMP_dt,dsigmaL_DVMP_dt, M_jpsi,epsilon, R_fitted
from . import config

from multiprocessing import Pool
from functools import partial
from iminuit import Minuit
import numpy as np
import pandas as pd
import time, csv, os, atexit

dir_path = os.path.dirname(os.path.realpath(__file__))
Minuit_Counter = 0
Time_Counter = 1
Q_threshold = 1.9
t_threshold_Jpsi = 0.6
xB_Cut = 0.5
xB_small_Cut = 0.0001
time_start = time.time()
"""
************************ Some auxilary functions and variables for convience ****************************
"""

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

Paralst_Aux_Names = ["jpsinorm"] 

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
SAVE_TO_FILE = True
SAVE_TO_FILE_PATH = '.'
def Export_Frame_Append(df, filename, export_path = None):

    if export_path is None:
        export_path = SAVE_TO_FILE_PATH

    os.makedirs(os.path.join(export_path, 'GUMP_Results'), exist_ok=True)
    
    global First_Write_Flag

    first_write = First_Write_Flag.get(filename, True)

    mode = 'w' if first_write else 'a'
    header = first_write

    df.to_csv(os.path.join(export_path,'GUMP_Results',filename), mode=mode, index=False, header=header)

    # Mark file as written
    First_Write_Flag[filename] = False

_pool = None

def _cleanup_pool():
    global _pool
    if _pool is not None:
        _pool.close()
        _pool.join()
        _pool = None

def get_pool(processes=None):
    global _pool
    if _pool is None:
        _pool = Pool(processes)
        atexit.register(_cleanup_pool)
    return _pool

def close_pool():
    _cleanup_pool()

def group_by_unique(data, subset=['xB', 't', 'Q']):
    """
    Group a DataFrame by unique combinations of specified columns.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        subset (List[str], optional): Columns to group by. Default is ['xB', 't', 'Q'].

    Returns:
        List[pd.DataFrame]: A list of DataFrames, each corresponding to a unique group.
    """
    unique_combinations = data.drop_duplicates(subset=subset, keep='first')[subset].values.tolist()
    grouped_data = [
        data[np.logical_and.reduce([data[col] == val for col, val in zip(subset, values)])]
        for values in unique_combinations
    ]
    return grouped_data

"""
************************ PDF and tPDFs data preprocessing ****************************
"""

PDF_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/PDFdata.csv'), header = 0, names = ['x', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'x': float, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})

tPDF_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/tPDFdata.csv'),     header = 0, names = ['x', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'x': float, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})

GPD_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/GPDdata.csv'), header = 0, names = ['x', 'xi', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'x': float, 'xi': float, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})

"""
************************ GFF data preprocessing ****************************
"""

GFF_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/GFFdata.csv'),       header = 0, names = ['j', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'j': int, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})

if not config.INC_gGFF:
    GFF_data= GFF_data[GFF_data['flv']!='g']

"""
************************ DVCS data preprocessing ****************************
"""

DVCSxsec_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVCSxsec_Old.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'phi', 'f', 'delta f', 'pol'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'phi': float, 'f': float, 'delta f': float, 'pol': str})
DVCSxsec_data_invalid = DVCSxsec_data[DVCSxsec_data['t']*(DVCSxsec_data['xB']-1) - M ** 2 * DVCSxsec_data['xB'] ** 2 < 0]
DVCSxsec_data = DVCSxsec_data[(DVCSxsec_data['Q'] > Q_threshold) & (DVCSxsec_data['xB'] < xB_Cut) & (DVCSxsec_data['t']*(DVCSxsec_data['xB']-1) - M ** 2 * DVCSxsec_data['xB'] ** 2 > 0) & (DVCSxsec_data['delta f']>0) & ((DVCSxsec_data['f']>0) | (DVCSxsec_data['pol']!='UU'))]
DVCSxsec_group_data = group_by_unique(DVCSxsec_data)

DVCSxsecNew_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVCSxsec_New.csv'), header = 0, names = ['y', 'xB', 't', 'Q', 'phi', 'f', 'delta f', 'pol'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'phi': float, 'f': float, 'delta f': float, 'pol': str})
DVCSxsecNew_data_invalid = DVCSxsecNew_data[DVCSxsecNew_data['t']*(DVCSxsecNew_data['xB']-1) - M ** 2 * DVCSxsecNew_data['xB'] ** 2 < 0]
DVCSxsecNew_data = DVCSxsecNew_data[(DVCSxsecNew_data['Q'] > Q_threshold) & (DVCSxsecNew_data['xB'] < xB_Cut) & (DVCSxsecNew_data['t']*(DVCSxsecNew_data['xB']-1) - M ** 2 * DVCSxsecNew_data['xB'] ** 2 > 0) & (DVCSxsecNew_data['delta f']>0) & ((DVCSxsecNew_data['f']>0) | (DVCSxsecNew_data['pol']!='UU'))]
DVCSxsecNew_group_data = group_by_unique(DVCSxsecNew_data)

DVCSxsec_HERA_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVCSxsec_HERA.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f', 'pol'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float, 'pol': str})
DVCSxsec_HERA_data_invalid = DVCSxsec_HERA_data[DVCSxsec_HERA_data['t']*(DVCSxsec_HERA_data['xB']-1) - M ** 2 * DVCSxsec_HERA_data['xB'] ** 2 < 0]
DVCSxsec_HERA_data = DVCSxsec_HERA_data[(DVCSxsec_HERA_data['Q'] > Q_threshold) & (DVCSxsec_HERA_data['xB'] < xB_Cut) & (DVCSxsec_HERA_data['t']*(DVCSxsec_HERA_data['xB']-1) - M ** 2 * DVCSxsec_HERA_data['xB'] ** 2 > 0)]
DVCSxsec_HERA_group_data = group_by_unique(DVCSxsec_HERA_data)

DVCSAsym_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVCSAsym.csv'), header = 0, names = ['y', 'xB', 't', 'Q', 'phi', 'f', 'delta f', 'pol'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'phi': float, 'f': float, 'delta f': float, 'pol': str})
DVCSAsym_data_invalid = DVCSAsym_data[DVCSAsym_data['t']*(DVCSAsym_data['xB']-1) - M ** 2 * DVCSAsym_data['xB'] ** 2 < 0]
DVCSAsym_data = DVCSAsym_data[(DVCSAsym_data['Q'] > Q_threshold) & (DVCSAsym_data['xB'] < xB_Cut) & (DVCSAsym_data['t']*(DVCSAsym_data['xB']-1) - M ** 2 * DVCSAsym_data['xB'] ** 2 > 0) & DVCSAsym_data['delta f']>0]
DVCSAsym_group_data = group_by_unique(DVCSAsym_data)

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
DVrhoPZEUSxsec_group_data = group_by_unique(DVrhoPZEUSxsec_data)

# Converting to longitudinal cross-sections
DVrhoPZEUSxsecL_data = DVrhoPZEUSxsec_data.copy()
dsigmaL_dt_ZEUS, dsigmaL_dt_err_ZEUS = DVMP_L_Error_Prop(DVrhoPZEUSxsecL_data, 1)
DVrhoPZEUSxsecL_data['f'] = dsigmaL_dt_ZEUS
DVrhoPZEUSxsecL_data['delta f'] = dsigmaL_dt_err_ZEUS
DVrhoPZEUSxsecL_group_data = group_by_unique(DVrhoPZEUSxsecL_data)

DVrhoPH1xsec_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/DVrhoPH1dt.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float})
DVrhoPH1xsec_data['Q'] = np.sqrt(DVrhoPH1xsec_data['Q'])
DVrhoPH1xsec_data['t'] = -1 * DVrhoPH1xsec_data['t']
DVrhoPH1xsec_data = DVrhoPH1xsec_data[(DVrhoPH1xsec_data['Q']>Q_threshold)]
DVrhoPH1xsec_group_data = group_by_unique(DVrhoPH1xsec_data)

# Converting to longitudinal cross-sections
DVrhoPH1xsecL_data = DVrhoPH1xsec_data.copy()
dsigmaL_dt_H1, dsigmaL_dt_err_H1 = DVMP_L_Error_Prop(DVrhoPH1xsecL_data, 1)
DVrhoPH1xsecL_data['f'] = dsigmaL_dt_H1
DVrhoPH1xsecL_data['delta f'] = dsigmaL_dt_err_H1
DVrhoPH1xsecL_group_data = group_by_unique(DVrhoPH1xsecL_data)

"""
************************ DVMP for phi data preprocessing ****************************
"""

DVphiPZEUSxsec_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/DVphiPZEUSdt.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float})
DVphiPZEUSxsec_data['Q'] = np.sqrt(DVphiPZEUSxsec_data['Q'])
DVphiPZEUSxsec_data['t'] = -1 * DVphiPZEUSxsec_data['t']
DVphiPZEUSxsec_data = DVphiPZEUSxsec_data[(DVphiPZEUSxsec_data['Q']>Q_threshold)]
DVphiPZEUSxsec_group_data = group_by_unique(DVphiPZEUSxsec_data)

DVphiPH1xsec_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/DVphiPH1dt.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float})
DVphiPH1xsec_data['Q'] = np.sqrt(DVphiPH1xsec_data['Q'])
DVphiPH1xsec_data['t'] = -1 * DVphiPH1xsec_data['t']
DVphiPH1xsec_data = DVphiPH1xsec_data[(DVphiPH1xsec_data['Q']>Q_threshold)]
DVphiPH1xsec_group_data = group_by_unique(DVphiPH1xsec_data)

"""
************************ DVMP for Jpsi data preprocessing ****************************
"""

DVJpsiPH1xsec_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/DVJpsiPH1dt_w_mass.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float})
DVJpsiPH1xsec_data['Q'] = np.sqrt(DVJpsiPH1xsec_data['Q'])
DVJpsiPH1xsec_data['t'] = -1 * DVJpsiPH1xsec_data['t']
DVJpsiPH1xsec_data = DVJpsiPH1xsec_data[(DVJpsiPH1xsec_data['Q']>Q_threshold)]
DVJpsiPH1xsec_data = DVJpsiPH1xsec_data[(-DVJpsiPH1xsec_data['t']<t_threshold_Jpsi)]
DVJpsiPH1xsec_group_data = group_by_unique(DVJpsiPH1xsec_data)

DVJpsiPZEUSxsec_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/DVJpsiPZEUSdt_w_mass.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float})
DVJpsiPZEUSxsec_data['Q'] = np.sqrt(DVJpsiPZEUSxsec_data['Q'])
DVJpsiPZEUSxsec_data['t'] = -1 * DVJpsiPZEUSxsec_data['t']
DVJpsiPZEUSxsec_data = DVJpsiPZEUSxsec_data[(DVJpsiPZEUSxsec_data['Q']>Q_threshold)]
DVJpsiPZEUSxsec_data = DVJpsiPZEUSxsec_data[(-DVJpsiPZEUSxsec_data['t']<t_threshold_Jpsi)]
DVJpsiPZEUSxsec_group_data = group_by_unique(DVJpsiPZEUSxsec_data)

'''
For Jpsi we used NRQCD framework that predict the R ratio, the following code not in use

R = sigma_L / sigma_T values hardcoded and used to convert data xsec simga_tot to sigma_L
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

"""
************************ Photon productions of Jpsi data preprocessing (Not in use) ****************************
"""

JpsiphotoH1xsec_data = pd.read_csv(os.path.join(dir_path,'GUMPDATA/DVMP_HERA/DVJpsiPZEUSdt_w_mass.csv'), header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float})
JpsiphotoH1xsec_data['Q'] = np.sqrt(JpsiphotoH1xsec_data['Q'])
JpsiphotoH1xsec_data['t'] = -1 * JpsiphotoH1xsec_data['t']
JpsiphotoH1xsec_data = JpsiphotoH1xsec_data[(JpsiphotoH1xsec_data['Q']>Q_threshold)]
xBtQlst_JpsiphotoH1 = JpsiphotoH1xsec_data.drop_duplicates(subset = ['xB', 't', 'Q'], keep = 'first')[['xB','t','Q']].values.tolist()

# Helper functions now take multiple arguments directly
def PDF_theo_scalar_helper(x_i, xi_i, t_i, Q_i, p_i, flv_i, Para_i, p_order):
    _PDF_theo = GPDobserv(x_i, xi_i, t_i, Q_i, p_i)  
    return _PDF_theo.tPDF(flv_i, Para_i, p_order)

# Helper function for scalar computation
def GFF_theo_scalar_helper(j_i, x, xi, t_i, Q_i, p_i, flv_i, Para_i, p_order):
    _GFF_theo = GPDobserv(x, xi, t_i, Q_i, p_i)
    return _GFF_theo.GFFj0(j_i, flv_i, Para_i, p_order)

def GPD_theo_scalar_helper(x_i, xi_i, t_i, Q_i, p_i, flv_i, Para_i, p_order):
    _GPD_theo = GPDobserv(x_i, xi_i, t_i, Q_i, p_i)  
    return _GPD_theo.GPD(flv_i, Para_i, p_order)

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
    args = zip(xs, xis, ts, Qs, ps, flvs, Para_spe, [p_order]*len(xs))

    # Use multiprocessing Pool to parallelize the computation'
    pool = get_pool()
    PDF_input['pred f'] = list(pool.starmap(PDF_theo_scalar_helper, args))
    if "f" in PDF_input and "delta f" in PDF_input:
        PDF_input['cost'] = ((PDF_input["pred f"]-PDF_input["f"])/PDF_input["delta f"])**2
    
    return PDF_input

tPDF_theo = PDF_theo

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
    
    args = zip(js, [x]*len(js), [xi]*len(js), ts, Qs, ps, flvs, Para_spe, [p_order]*len(js))

    pool = get_pool()
    GFF_input['pred f'] = list(pool.starmap(GFF_theo_scalar_helper, args))
    if "f" in GFF_input and "delta f" in GFF_input:
        GFF_input['cost'] = ((GFF_input["pred f"]-GFF_input["f"])/GFF_input["delta f"])**2
    
    return GFF_input

def GPD_theo(GPD_input: pd.DataFrame, Para: np.array, p_order = 2):
    
    GPD_input = GPD_input.copy()
    
    xs = GPD_input['x'].to_numpy()
    xis = GPD_input['xi'].to_numpy()
    ts = GPD_input['t'].to_numpy()
    Qs = GPD_input['Q'].to_numpy()
    flvs = GPD_input['flv'].to_numpy()
    spes = GPD_input['spe'].to_numpy()
    ps = np.where(spes <= 1, 1, -1)
    Para_spe = Para[spes]
    
    # Prepare input arguments for parallel computation
    args = zip(xs, xis, ts, Qs, ps, flvs, Para_spe, [p_order]*len(xs))

    # Use multiprocessing Pool to parallelize the computation
    pool = get_pool()
    GPD_input['pred f'] = list(pool.starmap(GPD_theo_scalar_helper, args))
    if "f" in GPD_input and "delta f" in GPD_input:
        GPD_input['cost'] = ((GPD_input["pred f"]-GPD_input["f"])/GPD_input["delta f"])**2
    
    return GPD_input

def DVCSxsec_theo_helper(DVCSxsec_input: pd.DataFrame, CFF_input: np.array):
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

def DVCSAsym_theo_helper(DVCSAsym_input: pd.DataFrame, CFF_input: np.array):
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

def DVCSxsec_HERA_theo_helper(DVCSxsec_HERA_input: pd.DataFrame, CFF_input: np.array):
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

def DVMPxsec_theo_helper(DVMPxsec_input: pd.DataFrame,  TFF_input: np.array, meson:int):
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

def TFF_theo(xB, t, Q, Para_Unp, meson:int, p_order = 2, muset = 1, flv = 'All'):
    x = 0
    xi = (1/(2 - xB) - (2*t*(-1 + xB))/((Q**2)*(-2 + xB)**2))*xB
    if (meson==3):
       xi = (1/(2 - xB) - (2*t*(-1 + xB))/((Q**2 + M_jpsi**2)*(-2 + xB)**2))*xB
    H_E = GPDobserv(x, xi, t, Q, 1)
    HTFF = H_E.TFF(Para_Unp[..., 0, :, :, :, :], muset * Q, meson, p_order, flv)
    ETFF = H_E.TFF(Para_Unp[..., 1, :, :, :, :], muset * Q, meson, p_order, flv)

    return  [ HTFF, ETFF]

def DVCSxsec_theo_xBtQ(DVCSxsec_data_xBtQ: pd.DataFrame, Para_Unp, Para_Pol, P_order = 2):
    [xB, t, Q] = [DVCSxsec_data_xBtQ['xB'].iat[0], DVCSxsec_data_xBtQ['t'].iat[0], DVCSxsec_data_xBtQ['Q'].iat[0]] 
    [HCFF, ECFF, HtCFF, EtCFF] = CFF_theo(xB, t, Q, Para_Unp, Para_Pol, porder= P_order) # scalar for each of them

    DVCSxsec_data_xBtQ['pred f'] = DVCSxsec_theo_helper(DVCSxsec_data_xBtQ, CFF_input = [HCFF, ECFF, HtCFF, EtCFF])
    if "f" in DVCSxsec_data_xBtQ and "delta f" in DVCSxsec_data_xBtQ:
        DVCSxsec_data_xBtQ['cost'] = ((DVCSxsec_data_xBtQ['pred f'] - DVCSxsec_data_xBtQ['f'])/ DVCSxsec_data_xBtQ['delta f']) ** 2

    return DVCSxsec_data_xBtQ

def DVCSAsym_theo_xBtQ(DVCSAsym_data_xBtQ: pd.DataFrame, Para_Unp, Para_Pol, P_order = 2):
    [xB, t, Q] = [DVCSAsym_data_xBtQ['xB'].iat[0], DVCSAsym_data_xBtQ['t'].iat[0], DVCSAsym_data_xBtQ['Q'].iat[0]] 
    [HCFF, ECFF, HtCFF, EtCFF] = CFF_theo(xB, t, Q, Para_Unp, Para_Pol, porder= P_order) # scalar for each of them

    DVCSAsym_data_xBtQ['pred f'] = DVCSAsym_theo_helper(DVCSAsym_data_xBtQ, CFF_input = [HCFF, ECFF, HtCFF, EtCFF])
    if "f" in DVCSAsym_data_xBtQ and "delta f" in DVCSAsym_data_xBtQ:
        DVCSAsym_data_xBtQ['cost'] = ((DVCSAsym_data_xBtQ['pred f'] - DVCSAsym_data_xBtQ['f'])/ DVCSAsym_data_xBtQ['delta f']) ** 2

    return DVCSAsym_data_xBtQ

def DVCSxsecHERA_theo_xBtQ(DVCSxsec_HERA_data_xBtQ: pd.DataFrame, Para_Unp, Para_Pol , P_order = 2):

    [xB, t, Q] = [DVCSxsec_HERA_data_xBtQ['xB'].iat[0], DVCSxsec_HERA_data_xBtQ['t'].iat[0], DVCSxsec_HERA_data_xBtQ['Q'].iat[0]] 
    [HCFF, ECFF, HtCFF, EtCFF] = CFF_theo(xB, t, Q, Para_Unp, Para_Pol, porder = P_order) # scalar for each of them

    DVCSxsec_HERA_data_xBtQ['pred f'] = DVCSxsec_HERA_theo_helper(DVCSxsec_HERA_data_xBtQ, CFF_input = [HCFF, ECFF, HtCFF, EtCFF])
    if "f" in DVCSxsec_HERA_data_xBtQ and "delta f" in DVCSxsec_HERA_data_xBtQ:
        DVCSxsec_HERA_data_xBtQ['cost'] = ((DVCSxsec_HERA_data_xBtQ['pred f'] - DVCSxsec_HERA_data_xBtQ['f'])/ DVCSxsec_HERA_data_xBtQ['delta f']) ** 2
        
    return DVCSxsec_HERA_data_xBtQ

def DVMPxsec_theo_xBtQ(DVMPxsec_data_xBtQ: pd.DataFrame, Para_Unp, xsec_norm, meson:int, p_order = 2):

    [xB, t, Q] = [DVMPxsec_data_xBtQ['xB'].iat[0], DVMPxsec_data_xBtQ['t'].iat[0], DVMPxsec_data_xBtQ['Q'].iat[0]] 
    [HTFF, ETFF] = TFF_theo(xB, t, Q, Para_Unp, meson, p_order, muset = 1)
    
    DVMPxsec_data_xBtQ['pred f'] = DVMPxsec_theo_helper(DVMPxsec_data_xBtQ, [HTFF, ETFF], meson) * xsec_norm**2
    if "f" in DVMPxsec_data_xBtQ and "delta f" in DVMPxsec_data_xBtQ:
        DVMPxsec_data_xBtQ['cost'] = ((DVMPxsec_data_xBtQ['pred f'] - DVMPxsec_data_xBtQ['f'])/ DVMPxsec_data_xBtQ['delta f']) ** 2

    return DVMPxsec_data_xBtQ

def DVCSxsec_theo(DVCSxsec_data: pd.DataFrame, Para_Unp, Para_Pol, P_order = 2):
    
    DVCSxsec_data_xBtQ = group_by_unique(DVCSxsec_data)
    pool = get_pool()
    DVCSxsec_data_xBtQ = pd.concat(list(pool.map(partial(DVCSxsec_theo_xBtQ, Para_Unp = Para_Unp, Para_Pol = Para_Pol, P_order = P_order), DVCSxsec_data_xBtQ)), ignore_index=True)
    return DVCSxsec_data_xBtQ

def DVCSAsym_theo(DVCSAsym_data: pd.DataFrame, Para_Unp, Para_Pol, P_order = 2):

    DVCSAsym_data_xBtQ = group_by_unique(DVCSAsym_data)
    pool = get_pool()
    DVCSAsym_data_xBtQ = pd.concat(list(pool.map(partial(DVCSAsym_theo_xBtQ, Para_Unp = Para_Unp, Para_Pol = Para_Pol, P_order = P_order), DVCSAsym_data_xBtQ)), ignore_index=True)
    return DVCSAsym_data_xBtQ

def DVMPxsec_theo(DVMPxsec_data: pd.DataFrame, Para_Unp, xsec_norm, meson:int, p_order = 2):
    
    DVMPxsec_data_xBtQ = group_by_unique(DVMPxsec_data)
    pool = get_pool()
    DVMPxsec_data_xBtQ = pd.concat(list(pool.map(partial(DVMPxsec_theo_xBtQ, Para_Unp = Para_Unp, xsec_norm = xsec_norm, meson = meson, p_order = p_order), DVMPxsec_data_xBtQ)), ignore_index=True)
    return DVMPxsec_data_xBtQ

def DVCSxsecHERA_theo(DVCSxsec_HERA_data: pd.DataFrame, Para_Unp, Para_Pol, P_order = 2):

    DVCSxsec_HERA_data_xBtQ = group_by_unique(DVCSxsec_HERA_data)
    pool = get_pool()
    DVCSxsec_HERA_data_xBtQ = pd.concat(list(pool.map(partial(DVCSxsecHERA_theo_xBtQ, Para_Unp = Para_Unp, Para_Pol = Para_Pol, P_order = P_order), DVCSxsec_HERA_data_xBtQ)), ignore_index=True)
    return DVCSxsec_HERA_data_xBtQ

def simple_dispatch(task):
    func, arg = task
    return func(arg)

def cost_off_forward_withH_withHt(Norm_HuV,    alpha_HuV,    beta_HuV,    alphap_HuV,   Invm2_HuV,
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
                    R_Etu_xi4,   R_Etd_xi4,    R_Etg_xi4,   bexp_HtSea,
                    jpsinorm):
    
    params = locals()
    validate_params(params, set(Paralst_Unp_Names + Paralst_Pol_Names + Paralst_Aux_Names))
    Para_Unp_lst = [params[name] for name in Paralst_Unp_Names]
    Para_Pol_lst = [params[name] for name in Paralst_Pol_Names]
    jpsinorm = params["jpsinorm"]
    
    global Minuit_Counter, Time_Counter

    time_now = time.time() - time_start
    
    if(time_now > Time_Counter * 600):
        print('Runing Time:',round(time_now/60),'minutes. Cost function called total', Minuit_Counter, 'times.')
        Time_Counter = Time_Counter + 1
    
    Minuit_Counter = Minuit_Counter + 1
    
    Para_Unp_all = ParaManager_Unp(Para_Unp_lst)
    Para_Pol_all = ParaManager_Pol(Para_Pol_lst)
    Para_Comb = np.concatenate([Para_Unp_all, Para_Pol_all], axis=0)
    
    tPDF_pred = tPDF_theo(tPDF_data, Para=Para_Comb)
    GFF_pred = GFF_theo(GFF_data, Para=Para_Comb)
    PDF_pred = PDF_theo(PDF_data, Para=Para_Comb)
    GPD_pred = GPD_theo(GPD_data, Para=Para_Comb)
    
    pool = get_pool()

    # DVCS_pred_xBtQ = pd.concat(list(pool.map(partial(DVCSxsec_cost_xBtQ, Para_Unp = Para_Unp_all, Para_Pol = Para_Pol_all, P_order = 2), DVCSxsec_group_data)), ignore_index=True)
    # DVCS_HERA_pred_xBtQ = pd.concat(list(pool.map(partial(DVCSxsec_HERA_cost_xBtQ, Para_Unp = Para_Unp_all, Para_Pol = Para_Pol_all, P_order = 2), DVCSxsec_HERA_group_data)), ignore_index=True)
    # DVrhoPH1_pred_xBtQ = pd.concat(list(pool.map(partial(DVMPxsec_cost_xBtQ, Para_Unp = Para_Unp_all, xsec_norm = 1, meson = 1, p_order = 2), DVrhoPH1xsecL_group_data)), ignore_index=True)
    # DVrhoPZEUS_pred_xBtQ = pd.concat(list(pool.map(partial(DVMPxsec_cost_xBtQ, Para_Unp = Para_Unp_all, xsec_norm = 1, meson = 1, p_order = 2), DVrhoPZEUSxsecL_group_data)), ignore_index=True)
    # DVCS_Asym_pred_xBtQ = pd.concat(list(pool.map(partial(DVCSAsym_cost_xBtQ, Para_Unp = Para_Unp_all, Para_Pol = Para_Pol_all, P_order = 2), DVCSAsym_group_data)), ignore_index=True)

    # Instead of initial a parallelization for each task (shown above)
    # the following scripts collects them into a larger task
    # This would reduce overhead (due to pickling, scheduling, etc.)
    # Not necessary if each task is large (NOT the case here)
    
    porder = 2
    all_tasks_def = {
        "DVCSxsec": (
            partial(DVCSxsec_theo_xBtQ,
                    Para_Unp=Para_Unp_all,
                    Para_Pol=Para_Pol_all,
                    P_order=porder),
            DVCSxsec_group_data,
            "DVCSxsec.csv"
        ),
        "DVCSxsec_HERA": (
            partial(DVCSxsecHERA_theo_xBtQ,
                    Para_Unp=Para_Unp_all,
                    Para_Pol=Para_Pol_all,
                    P_order=porder),
            DVCSxsec_HERA_group_data,
            "DVCSxsec_HERA.csv"
        ),
        "DVrhoPH1": (
            partial(DVMPxsec_theo_xBtQ,
                    Para_Unp=Para_Unp_all,
                    xsec_norm=1,
                    meson=1,
                    p_order=porder),
            DVrhoPH1xsecL_group_data,
            "DVMPxsec.csv"
        ),
        "DVrhoPZEUS": (
            partial(DVMPxsec_theo_xBtQ,
                    Para_Unp=Para_Unp_all,
                    xsec_norm=1,
                    meson=1,
                    p_order=porder),
            DVrhoPZEUSxsecL_group_data,
            "DVMPxsec.csv"
        ),
        "DVJpsiPH1": (
            partial(DVMPxsec_theo_xBtQ,
                    Para_Unp=Para_Unp_all,
                    xsec_norm=jpsinorm,
                    meson=3,
                    p_order=porder),
            DVJpsiPH1xsec_group_data,
            "DVJpsiPxsec.csv"
        ),
        "DVJpsiPZEUS": (
            partial(DVMPxsec_theo_xBtQ,
                    Para_Unp=Para_Unp_all,
                    xsec_norm=jpsinorm,
                    meson=3,
                    p_order=porder),
            DVJpsiPZEUSxsec_group_data,
            "DVJpsiPZEUSxsec.csv"
        ),
        "DVCSxsec_New": (
            partial(DVCSxsec_theo_xBtQ,
                    Para_Unp=Para_Unp_all,
                    Para_Pol=Para_Pol_all,
                    P_order=porder),
            DVCSxsecNew_group_data,
            "DVCSxsec_New.csv"
        ),
        "DVCSAsym": (
            partial(DVCSAsym_theo_xBtQ,
                    Para_Unp=Para_Unp_all,
                    Para_Pol=Para_Pol_all,
                    P_order=porder),
            DVCSAsym_group_data,
            "DVCSAsym.csv"
        ),
    }
    selected_tasks = [
        "DVCSxsec",
        "DVCSxsec_HERA",
        "DVrhoPH1",
        "DVrhoPZEUS",
    ]
    if config.INC_JPSI:
        selected_tasks.append("DVJpsiPH1")
        #selected_tasks.append("DVJpsiPZEUS")
    
    all_tasks_exp = []
    task_names = []
    for name in selected_tasks:
        func, data, _ = all_tasks_def[name]
        all_tasks_exp.extend([(func, arg) for arg in data])
        task_names.extend([name] * len(data))

    all_results_exp = pool.map(simple_dispatch, all_tasks_exp)
    
    total_cost_exp = sum(df["cost"].sum() for df in all_results_exp if "cost" in df.columns)
    
    if config.Export_Mode:
        
        grouped_results = {}
        
        grouped_results["tPDF_pred"] = tPDF_pred
        grouped_results["GFF_pred"] = GFF_pred
        grouped_results["PDF_pred"] = PDF_pred
        grouped_results["GPD_pred"] = GPD_pred
        
        if SAVE_TO_FILE:
            Export_Frame_Append(tPDF_pred,"tPDFcomp.csv")
            Export_Frame_Append(GFF_pred,"GFFcomp.csv")
            Export_Frame_Append(PDF_pred,"PDFcomp.csv")
            Export_Frame_Append(GPD_pred,"GPDcomp.csv")
            
        start = 0
        
        for name in selected_tasks:
            func, data, filename = all_tasks_def[name]
            end = start + len(data)
            grouped_results[name] = pd.concat(all_results_exp[start:end], ignore_index=True)
            if SAVE_TO_FILE:
                Export_Frame_Append(grouped_results[name], filename)
            start = end
        
        total_points = 0
        total_chi2 = 0

        for name, df in grouped_results.items():
            datalen = len(df)
            totchi2 = df['cost'].sum()
            print(f'For task {name} with total {datalen} data points, total chi^2: {totchi2:.2f} and chi^2 per data point: {totchi2/datalen:.2f}')
            
            total_points += datalen
            total_chi2 += totchi2

        # Summary across all tasks
        print(f'\nOverall: total {total_points} data points, total chi^2: {total_chi2:.2f}, chi^2 per data point: {total_chi2/total_points:.2f}')

        return grouped_results
        
    return total_cost_exp + tPDF_pred['cost'].sum() + GFF_pred['cost'].sum() + PDF_pred['cost'].sum() + GPD_pred['cost'].sum()

def off_forward_fit_withH_withHt(Paralst_Unp, Paralst_Pol, Paralst_Aux=[1.0] * len(Paralst_Aux_Names), export_path = '.'):

    assert config.Export_Mode == False, "Make sure the Export_Mode is set to False in config.py before fitting"
    
    # Create dictionaries by zipping names and values
    params_unp = dict(zip(Paralst_Unp_Names, Paralst_Unp))
    params_pol = dict(zip(Paralst_Pol_Names, Paralst_Pol))
    params_aux = dict(zip(Paralst_Aux_Names, Paralst_Aux))
    
    params = {**params_unp, **params_pol, **params_aux}

    fit_off_forward = Minuit(cost_off_forward_withH_withHt, **params)
    
    fit_off_forward.errordef = 1

    norm_max = 1
    
    fit_off_forward.limits['Norm_HuV']     = (-norm_max,norm_max)
    fit_off_forward.limits['Norm_Hubar']   = (-norm_max,norm_max)
    fit_off_forward.limits['Norm_HdV']     = (-norm_max,norm_max)
    fit_off_forward.limits['Norm_Hdbar']   = (-norm_max,norm_max)
    fit_off_forward.limits['Norm_Hg']      = (-norm_max,norm_max)

    fit_off_forward.limits['Norm_Hubar_2'] = (-norm_max,norm_max)
    fit_off_forward.limits['Norm_Hdbar_2'] = (-norm_max,norm_max)
    fit_off_forward.limits['Norm_Hg_2']    = (-norm_max,norm_max)

    fit_off_forward.limits['Norm_EuV']     = (-norm_max,norm_max)
    fit_off_forward.limits['Norm_EdV']     = (-norm_max,norm_max)
    
    fit_off_forward.limits['Norm_HtuV']    = (-norm_max,norm_max)
    fit_off_forward.limits['Norm_Htubar']  = (-norm_max,norm_max)
    fit_off_forward.limits['Norm_HtdV']    = (-norm_max,norm_max)
    fit_off_forward.limits['Norm_Htdbar']  = (-norm_max,norm_max)
    fit_off_forward.limits['Norm_Htg']     = (-norm_max,norm_max)
    fit_off_forward.limits['Norm_EtuV']    = (-norm_max, norm_max)
    fit_off_forward.limits['Norm_EtdV']    = (-norm_max, norm_max)

    fit_off_forward.limits['alpha_HuV']   = (-2, 1.2)
    fit_off_forward.limits['alpha_Hubar'] = (-2, 1.2)
    fit_off_forward.limits['alpha_HdV']   = (-2, 1.2)
    fit_off_forward.limits['alpha_Hdbar'] = (-2, 1.2)
    fit_off_forward.limits['alpha_Hg']    = (-2, 1.2)
    fit_off_forward.limits['alpha_EuV']   = (-2, 1.2)

    # make the second set of parameters 'valence-like' bounded by small x <~ x **(-0.6)
    fit_off_forward.limits['alpha_Hubar_2'] = (-2, 0.6)
    fit_off_forward.limits['alpha_Hdbar_2'] = (-2, 0.6)
    fit_off_forward.limits['alpha_Hg_2']    = (-2, 0.6)

    fit_off_forward.limits['alpha_EuV']     = (-2, 1.2)
    fit_off_forward.limits['alpha_HtuV']    = (-2, 1.2)
    fit_off_forward.limits['alpha_Htubar']  = (-2, 1.2)
    fit_off_forward.limits['alpha_HtdV']    = (-2, 1.2)
    fit_off_forward.limits['alpha_Htdbar']  = (-2, 1.2)
    fit_off_forward.limits['alpha_Htg']     = (-2, 1.2)
    fit_off_forward.limits['alpha_EtuV']    = (-2, 1.2)

    beta_max = 20
    
    fit_off_forward.limits['beta_HuV']     = (0, beta_max)
    fit_off_forward.limits['beta_Hubar']   = (0, beta_max)
    fit_off_forward.limits['beta_HdV']     = (0, beta_max)
    fit_off_forward.limits['beta_Hdbar']   = (0, beta_max)
    fit_off_forward.limits['beta_Hg']      = (0, beta_max)
    fit_off_forward.limits['beta_EuV']     = (0, beta_max)
    
    fit_off_forward.limits['beta_Hubar_2'] = (0, beta_max)
    fit_off_forward.limits['beta_Hdbar_2'] = (0, beta_max)
    fit_off_forward.limits['beta_Hg_2']    = (0, beta_max)
    
    fit_off_forward.limits['beta_HtuV']    = (0, beta_max)
    fit_off_forward.limits['beta_Htubar']  = (0, beta_max)
    fit_off_forward.limits['beta_HtdV']    = (0, beta_max)
    fit_off_forward.limits['beta_Htdbar']  = (0, beta_max)
    fit_off_forward.limits['beta_Htg']     = (0, beta_max)
    fit_off_forward.limits['beta_EtuV']    = (0, beta_max)
    
    alpha_p_max = 5
    
    fit_off_forward.limits['alphap_HuV']   = (0, alpha_p_max)
    fit_off_forward.limits['Invm2_HuV']    = (0, alpha_p_max)
    fit_off_forward.limits['alphap_HdV']   = (0, alpha_p_max)
    fit_off_forward.limits['Invm2_HdV']    = (0, alpha_p_max)
    fit_off_forward.limits['Invm2_Hg']     = (0, alpha_p_max)
    fit_off_forward.limits['alphap_EuV']   = (0, alpha_p_max)

    fit_off_forward.limits['alphap_HtuV']  = (0, alpha_p_max)
    fit_off_forward.limits['alphap_HtdV']  = (0, alpha_p_max)
    fit_off_forward.limits['alphap_EtuV']  = (0, alpha_p_max)

    fit_off_forward.limits['R_E_Sea'] = (-10,10)
    fit_off_forward.limits['R_Et_Sea'] = (-100, 100)
    
    bmax = 15
    
    fit_off_forward.limits['bexp_Hg']    = (0, bmax)
    fit_off_forward.limits['bexp_HSea']  = (0, bmax)
    fit_off_forward.limits['bexp_HtSea'] = (0, bmax)
    
    Rmax = 10
    
    fit_off_forward.limits['R_Hu_xi2'] =(-Rmax,Rmax)
    fit_off_forward.limits['R_Hd_xi2'] =(-Rmax,Rmax)
    fit_off_forward.limits['R_Hg_xi2'] =(-Rmax,Rmax)
    fit_off_forward.limits['R_Eu_xi2'] =(-Rmax,Rmax)
    fit_off_forward.limits['R_Ed_xi2'] =(-Rmax,Rmax)
    fit_off_forward.limits['R_Eg_xi2'] =(-Rmax,Rmax)
    
    fit_off_forward.limits['R_Hu_xi4'] =(-Rmax,Rmax)
    fit_off_forward.limits['R_Hd_xi4'] =(-Rmax,Rmax)
    fit_off_forward.limits['R_Hg_xi4'] =(-Rmax,Rmax)
    fit_off_forward.limits['R_Eu_xi4'] =(-Rmax,Rmax)
    fit_off_forward.limits['R_Ed_xi4'] =(-Rmax,Rmax)
    fit_off_forward.limits['R_Eg_xi4'] =(-Rmax,Rmax)
    
    fit_off_forward.limits['R_Htu_xi2'] =(-Rmax,Rmax)
    fit_off_forward.limits['R_Htd_xi2'] =(-Rmax,Rmax)
    fit_off_forward.limits['R_Htg_xi2'] =(-Rmax,Rmax)
    fit_off_forward.limits['R_Etu_xi2'] =(-Rmax,Rmax)
    fit_off_forward.limits['R_Etd_xi2'] =(-Rmax,Rmax)
    fit_off_forward.limits['R_Etg_xi2'] =(-Rmax,Rmax)
    
    fit_off_forward.limits['R_Htu_xi4'] =(-Rmax,Rmax)
    fit_off_forward.limits['R_Htd_xi4'] =(-Rmax,Rmax)
    fit_off_forward.limits['R_Htg_xi4'] =(-Rmax,Rmax)
    fit_off_forward.limits['R_Etu_xi4'] =(-Rmax,Rmax)
    fit_off_forward.limits['R_Etd_xi4'] =(-Rmax,Rmax)
    fit_off_forward.limits['R_Etg_xi4'] =(-Rmax,Rmax)
    
    fixed_params = [
        #"Norm_HuV", "alpha_HuV", "beta_HuV", 
        #"Norm_Hubar", "alpha_Hubar", "beta_Hubar", 
        #"Norm_Hubar_2", "alpha_Hubar_2", "beta_Hubar_2",
        #"Norm_HdV", "alpha_HdV", "beta_HdV", 
        #"Norm_Hdbar", "alpha_Hdbar", "beta_Hdbar",
        #"Norm_Hdbar_2", "alpha_Hdbar_2", "beta_Hdbar_2",
        #"Norm_Hg", "alpha_Hg", "beta_Hg", 
        #"Norm_Hg_2", "alpha_Hg_2", "beta_Hg_2",
        "alphap_Hqbar", "alphap_Hg", #"Invm2_Hg",
        #"Norm_HtuV", "alpha_HtuV", "beta_HtuV",
        #"Norm_Htubar", "alpha_Htubar", "beta_Htubar", 
        #"Norm_HtdV", "alpha_HtdV", "beta_HtdV",
        #"Norm_Htdbar", "alpha_Htdbar", "beta_Htdbar",
        #"Norm_Htg", "alpha_Htg", "beta_Htg",
        "alphap_Htqbar", "alphap_Htg",
        "R_Htg_xi2", "R_Etg_xi2",
        "R_Htg_xi4", "R_Etg_xi4", "R_Htu_xi4", "R_Etu_xi4", "R_Htd_xi4", "R_Etd_xi4"]
    
    if not config.INC_JPSI:
        fixed_params.append("jpsinorm")
        
    for param in fixed_params:
        fit_off_forward.fixed[param] = True
    
    global Minuit_Counter, Time_Counter, time_start
    Minuit_Counter = 0
    Time_Counter = 1
    time_start = time.time()
    
    print("------------------------------------------")
    print("off forward fit starts, update in 10 mins")
    
    fit_off_forward.migrad(ncall=100000)
    fit_off_forward.hesse()
    
    print("off forward fit finished, see summary in /GUMP_Output")

    time_end = time.time() -time_start
    
    ndof_off_forward = (len(DVCSxsec_data.index) + len(DVCSxsec_HERA_data.index) 
                        + len(DVrhoPH1xsec_data.index) + len(DVrhoPZEUSxsec_data.index)
                         + len(tPDF_data.index) + len(GFF_data.index) + len(PDF_data.index) + len(GPD_data.index) - fit_off_forward.nfit)
    
    os.makedirs(os.path.join(export_path, 'GUMP_Output'), exist_ok=True)
    
    if config.INC_JPSI:
        ndof_off_forward = ndof_off_forward + len(DVJpsiPH1xsec_data.index)
        Exp_path = os.path.join(export_path,'GUMP_Output/off_forward_fit_withH_withHt_NLO_withJpsi.txt')
    else:
        Exp_path = os.path.join(export_path,'GUMP_Output/off_forward_fit_withH_withHt_NLO.txt')
    
    with open(Exp_path, 'w', encoding='utf-8') as f:
        print('Total running time: %.1f minutes. Total call of cost function: %3d.\n' % ( time_end/60, fit_off_forward.nfcn), file=f)
        print('The chi squared/d.o.f. is: %.2f / %3d ( = %.2f ).\n' % (fit_off_forward.fval, ndof_off_forward, fit_off_forward.fval/ndof_off_forward), file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*fit_off_forward.values, sep=", ", file = f)
        print(*fit_off_forward.errors, sep=", ", file = f)
        print(fit_off_forward.params, file = f)

    return fit_off_forward

Paralst_Unp_off_forward=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Unp_Off_forward_withH_withHt_NLO.csv'), header=None).to_numpy()[0]
Paralst_Pol_off_forward=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Pol_Off_forward_withH_withHt_NLO.csv'), header=None).to_numpy()[0]

Para_Unp_off_forward = ParaManager_Unp(Paralst_Unp_off_forward)
Para_Pol_off_forward = ParaManager_Pol(Paralst_Pol_off_forward[:-1]) # exclude jpsi_norm

Para_Comb_off_forward = np.concatenate([Para_Unp_off_forward, Para_Pol_off_forward], axis=0)