from Parameters import ParaManager_Unp, ParaManager_Pol
from Observables import GPDobserv
from DVCS_xsec import dsigma_TOT, M, dsigma_DVCSINT_HERA
from multiprocessing import Pool
from functools import partial
import numpy as np
import pandas as pd
import csv
from  Plotting import plot_compare
from Minimizer import Q_threshold, xB_Cut, DVCSxsec_theo, CFF_theo, DVCSxsec_HERA_theo

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

DVCS_HERA_data = pd.read_csv('GUMPDATA/DVCSxsec_HERA.csv', header = None, names = ['y', 'xB', 't', 'Q', 'f', 'delta f', 'pol'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'f': float, 'delta f': float, 'pol': str})

def DVCSxsec_cost_xBtQ_plt(DVCSxsec_data_xBtQ: np.array, Para_Unp, Para_Pol):
    [xB, t, Q] = [DVCSxsec_data_xBtQ['xB'].iat[0], DVCSxsec_data_xBtQ['t'].iat[0], DVCSxsec_data_xBtQ['Q'].iat[0]]
    [HCFF, ECFF, HtCFF, EtCFF] = CFF_theo(xB, t, Q, Para_Unp, Para_Pol)
    DVCS_pred_xBtQ = np.array(list(map(partial(DVCSxsec_theo, CFF_input = [HCFF, ECFF, HtCFF, EtCFF]), np.array(DVCSxsec_data_xBtQ))))
    return DVCS_pred_xBtQ, HCFF, ECFF, HtCFF, EtCFF



Paralst_Unp     = [4.916737288099265, 0.2160974235037556, 3.2258424299202755, 2.3297704052669705, 0.1626461362289145, 1.1368777981444618, 6.801675174056078, -0.054561193804464884, 3.4011312710904424, 0.18167877602047033, 4.431135738906448, 3.5552156393490506, 0.2566106051374345, 1.0456780115928632, 6.797731044211278, 2.857047337247385, 1.0528415885845877, 7.386015035320446, 1.3668037066534142, 8.89892151499381, -0.06026608225708063, 3.414072497873425, 5.355066163072291, -0.023541114852060577, 0.9891204496981967, 0.9632557231947952, 0.05097257659617962, 0.006415775282922205, -2.365434779238983, 0.3199008853020352, 1.0, 9.003044705421427, -84.86742073956177, 1.0, 0.609691017424693, 0.08237033996473307, 0.0, -2.603673477179661, 29.812645500588264, 0.0]
Paralst_Pol     = [4.519624755533353, -0.24572862253056815, 3.033722770434178, 2.6225128805098703, 0.07490016975962657, 0.5199239290487521, 4.323496948268691, -0.7774985115127935, -0.7128670325051504, 0.21132587362841937, 3.2387492201569494, 4.447810664973923, -0.055480456634778005, 0.615449804616039, 2.0757673709446696, 0.24177484233130947, 0.6324341817221732, 2.7058935184763966, 1.1, 8.772598157795674, 0.7999999801488662, 7.290777470786635, 1.9914823940193018, -3.491397001954771, -0.4676673268175943, 1.2258081300470829, 6.366555784071109, 1.0, 1.697980818866844, 34.44515102261264, 1.0, -0.42961433607987104, -1.8949842275991569, 0.0, 3.4598966572448955, 31.12987072198461, 0.0]

Para_Unp_All    = ParaManager_Unp(Paralst_Unp)
Para_Pol_All    = ParaManager_Pol(Paralst_Pol)

DVCS_HERA_pred = np.array(list(map(partial(DVCSxsec_HERA_theo, Para_Unp = Para_Unp_All, Para_Pol = Para_Pol_All), np.array(DVCS_HERA_data))))

print(DVCS_HERA_pred)

"""
print(CFF_theo(0.36, -0.345, 2, Para_Unp_All, Para_Pol_All))

print(CFF_theo(0.48, -0.702, 2, Para_Unp_All, Para_Pol_All))

print(CFF_theo(0.60, -1.05, 2, Para_Unp_All, Para_Pol_All))
"""

"""
xsecgp0 = DVCSxsec_group_data[0]
print(xsecgp0)
xsecgp0UU = xsecgp0[xsecgp0['pol'] == 'UU']
xsecgp0LU = xsecgp0[xsecgp0['pol'] == 'LU']
xsecgp0UU_pred, xsecgp0HCFF, xsecgp0ECFF, xsecgp0HtCFF, xsecgp0EtCFF =DVCSxsec_cost_xBtQ_plt(xsecgp0UU, Para_Unp_All,Para_Pol_All )
xsecgp0LU_pred, xsecgp0HCFF, xsecgp0ECFF, xsecgp0HtCFF, xsecgp0EtCFF =DVCSxsec_cost_xBtQ_plt(xsecgp0LU, Para_Unp_All,Para_Pol_All )
print(xsecgp0HCFF, xsecgp0ECFF, xsecgp0HtCFF, xsecgp0EtCFF)
plot_compare(xsecgp0UU['phi'], xsecgp0UU['f'], xsecgp0UU['delta f'], xsecgp0UU_pred)
plot_compare(xsecgp0LU['phi'], xsecgp0LU['f'], xsecgp0LU['delta f'], xsecgp0LU_pred)
    
xsecgp1 = DVCSxsec_group_data[3]
print(xsecgp1)
xsecgp1UU = xsecgp1[xsecgp1['pol'] == 'UU']
xsecgp1LU = xsecgp1[xsecgp1['pol'] == 'LU']
xsecgp1UU_pred, xsecgp1HCFF, xsecgp1ECFF, xsecgp1HtCFF, xsecgp1EtCFF=DVCSxsec_cost_xBtQ_plt(xsecgp1UU, Para_Unp_All,Para_Pol_All )
xsecgp1LU_pred, xsecgp1HCFF, xsecgp1ECFF, xsecgp1HtCFF, xsecgp1EtCFF=DVCSxsec_cost_xBtQ_plt(xsecgp1LU, Para_Unp_All,Para_Pol_All )
print(xsecgp1HCFF, xsecgp1ECFF, xsecgp1HtCFF, xsecgp1EtCFF)
plot_compare(xsecgp1UU['phi'], xsecgp1UU['f'], xsecgp1UU['delta f'], xsecgp1UU_pred)
plot_compare(xsecgp1LU['phi'], xsecgp1LU['f'], xsecgp1LU['delta f'], xsecgp1LU_pred)
"""