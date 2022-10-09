from Parameters import ParaManager_Unp, ParaManager_Pol
from Observables import GPDobserv
from DVCS_xsec import dsigma_TOT, M
from multiprocessing import Pool
from functools import partial
import numpy as np
import pandas as pd
import csv
from  Plotting import plot_compare
from Minimizer import Q_threshold, xB_Cut

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

def DVCSxsec_theo(DVCSxsec_input: np.array, CFF_input: np.array):
    [y, xB, t, Q, phi, f, delta_f, pol] = DVCSxsec_input    
    [HCFF, ECFF, HtCFF, EtCFF] = CFF_input
    return dsigma_TOT(y, xB, t, Q, phi, pol, HCFF, ECFF, HtCFF, EtCFF)

def CFF_theo(xB, t, Q, Para_Unp, Para_Pol):
    x = 0
    xi = (1/(2 - xB) - (2*t*(-1 + xB))/(Q**2*(-2 + xB)**2))*xB
    H_E = GPDobserv(x, xi, t, Q, 1)
    Ht_Et = GPDobserv(x, xi, t, Q, -1)
    HCFF = H_E.CFF(Para_Unp[0])
    ECFF = H_E.CFF(Para_Unp[1])
    HtCFF = Ht_Et.CFF(Para_Pol[0])
    EtCFF = Ht_Et.CFF(Para_Pol[1])
    return [HCFF, ECFF, HtCFF, EtCFF]

def DVCSxsec_cost_xBtQ_plt(DVCSxsec_data_xBtQ: np.array, Para_Unp, Para_Pol):
    [xB, t, Q] = [DVCSxsec_data_xBtQ['xB'].iat[0], DVCSxsec_data_xBtQ['t'].iat[0], DVCSxsec_data_xBtQ['Q'].iat[0]]
    [HCFF, ECFF, HtCFF, EtCFF] = CFF_theo(xB, t, Q, Para_Unp, Para_Pol)
    DVCS_pred_xBtQ = np.array(list(map(partial(DVCSxsec_theo, CFF_input = [HCFF, ECFF, HtCFF, EtCFF]), np.array(DVCSxsec_data_xBtQ))))
    return DVCS_pred_xBtQ, HCFF, ECFF, HtCFF, EtCFF

Paralst_Unp     = [4.916737278558534, 0.21609742224882345, 3.2258424318470875, 2.329770430094072, 0.16264613656427143, 1.1368777899039304, 6.801675140509348, 0.15, 3.4011312651381522, 0.18167877581076208, 4.431135746101499, 3.555215564682872, 0.2566106067781364, 1.0456776263884748, 6.797731093197489, 2.8570473561879495, 1.0528415892407397, 7.386014909053386, 1.3668036322682287, 8.896085949094301, -0.060173964292647764, 3.4138299551287297, 5.354287423672972, -0.023560835433581156, 0.9891107045280738, 0.9640799351738663, 0.051014532186647726, 0.2968118806143138, -0.43662590407683344, 0.8677369129076947, 1.0, 1.7582631094833985, -19.08861189644988, 1.0]
Paralst_Pol     = [4.484385723114465, -0.24270256222840825, 3.0221829168556322, 2.5813599824017084, 0.0816248919687765, 0.4967525618264337, 4.6075515808683, 0.15, -0.7969022361097922, 0.1811292058016334, 3.4449542640145414, 4.889519039240955, -0.0426870622150197, 0.6810202231163998, 1.611235931883931, 0.22559133792436348, 0.6435402455187647, 2.547329752170809, 1.1, 8.984480420587369, 0.7999993675004156, 7.3209602005677334, 2.0650650293391335, -3.546802823027886, -1.8431865818413569, -0.21224310748166364, -0.9218467193680021, 1.0, 6.467347471008478, 70.27254575836811, 1.0]

Para_Unp_All    = ParaManager_Unp(Paralst_Unp)
Para_Pol_All    = ParaManager_Pol(Paralst_Pol)

with open('GUMP_Test/xbtQ.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(xBtQlst)

print(CFF_theo(0.36, -0.345, 2, Para_Unp_All, Para_Pol_All))

print(CFF_theo(0.48, -0.702, 2, Para_Unp_All, Para_Pol_All))

print(CFF_theo(0.60, -1.05, 2, Para_Unp_All, Para_Pol_All))

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