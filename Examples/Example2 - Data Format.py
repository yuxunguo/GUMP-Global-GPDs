# ===========================================================================
# In this example, we show how to use the gumpgpd package to 
# calculate predictions for custom kinematic points
# ===========================================================================

# Each data should be a pandas.DataFrame with certain columns
# Here we list the required columns for each observable
# the 'f' and 'delta f' columns are optional
# a 'cost' will be calcuated from (('f'-'pred f')/'delta f')**2
# Not having 'f' and 'delta f' will still gererate correction 'pred f'

PDF_data_names = ['x', 't', 'Q', 'spe', 'flv']
tPDF_data_names = ['x', 't', 'Q', 'spe', 'flv'] # The same as PDF
GPD_data_names = ['x', 'xi', 't', 'Q', 'spe', 'flv']
GFF_data_names = ['j', 't', 'Q', 'spe', 'flv']

DVCS_data_names = ['y', 'xB', 't', 'Q', 'phi', 'pol']
DVCSAsym_data_names = ['y', 'xB', 't', 'Q', 'phi', 'pol'] #The same as cross-section
DVCSHERA_data_names = ['y', 'xB', 't', 'Q', 'pol']
DVMP_data_names = ['y', 'xB', 't', 'Q']

PDF_da1ta_names_withf = ['x', 't', 'Q', 'spe', 'flv', 'f', 'delta f']
tPDF_data_names_withf = ['x', 't', 'Q', 'spe', 'flv', 'f', 'delta f'] # The same as PDF
GPD_data_names_withf = ['x', 'xi', 't', 'Q', 'spe', 'flv', 'f', 'delta f']
GFF_data_names_withf = ['j', 't', 'Q', 'spe', 'flv', 'f', 'delta f']

DVCS_data_names_withf = ['y', 'xB', 't', 'Q', 'phi', 'pol', 'f', 'delta f']
DVCSAsym_data_names_withf = ['y', 'xB', 't', 'Q', 'phi', 'pol', 'f', 'delta f'] #The same as cross-section
DVCSHERA_data_names_withf = ['y', 'xB', 't', 'Q', 'pol', 'f', 'delta f']
DVMP_data_names_withf = ['y', 'xB', 't', 'Q', 'f', 'delta f']

from gumpgpd.Minimizer import tPDF_theo, Para_Comb_off_forward
import pandas as pd
import numpy as np

if __name__ == '__main__':
    
    tPDFs = pd.DataFrame(columns=tPDF_data_names)

    tarr = np.linspace(-10., -0.0, 20)

    tPDFs['x'] = [0.1]*len(tarr)
    tPDFs['t'] = tarr
    tPDFs['Q'] = [2.0]*len(tarr)
    tPDFs['spe'] = [0]*len(tarr)
    tPDFs['flv'] = ['u']*len(tarr)

    print(tPDF_theo(tPDFs, Para = Para_Comb_off_forward))