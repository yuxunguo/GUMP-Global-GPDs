from Parameters import ParaManager_Unp, ParaManager_Pol
from Observables import GPDobserv
from DVCS_xsec import dsigma_TOT, M
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



Paralst_Unp     = [4.916737292838296, 0.21609742412427124, 3.225842428948964, 2.3297703926690523, 0.1626461360535492, 1.136877802260643, 6.801675190950625, 0.15, 3.401131271152364, 0.1816787754762963, 4.431135736130402, 3.5552156773610974, 0.25661060360399013, 1.0456782034751386, 6.797731039344151, 2.8570473278540613, 1.0528415882622442, 7.386015098049054, 1.366803743605326, 8.900263429138406, -0.06033266914934843, 3.414148460180483, 5.355305749933862, -0.023519612427915677, 0.9891302149568588, 0.9628202929560131, 0.05092644113226002, -1.5776402552027544, -1.4801463124210181, -0.4063877364366284, 1.0, 9.756083712781301, -2.517123014969496, 1.0, 0.36009592508488664, 0.3094580553615649, 0.0, -2.9347561013361294, 0.7682576430255452, 0.0, 3.8950305384906523]
Paralst_Pol     = [4.519577689535176, -0.24572491638001592, 3.033682816721097, 2.6224407663261124, 0.07493384516563739, 0.5198270626455179, 4.324519400543717, 0.15, -0.7128110922056157, 0.21136679320262397, 3.2385333530307276, 4.44744157941434, -0.05548105929333334, 0.6154600177069685, 2.075270725889884, 0.24182799850037726, 0.6323768258179756, 2.7063705128526023, 1.1, 8.800439448713657, 0.7999999844707935, 7.299453540750408, 1.9992792400214974, -3.4996836559679814, -1.0190999097674642, 1.3534199361725696, 9.206927785440778, 1.0, 2.727763897245076, 13.332261446995417, 1.0, -0.433394885607879, -2.1252724569087094, 0.0, 2.8237107713689484, 34.883815783642305, 0.0, 0.00039354937060587236]
  
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