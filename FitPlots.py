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

GFF_data = pd.read_csv('GUMPDATA/GFFdata_Quark.csv',       header = None, names = ['j', 't', 'Q', 'f', 'delta f', 'spe', 'flv'],        dtype = {'j': int, 't': float, 'Q': float, 'f': float, 'delta f': float,'spe': int, 'flv': str})
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
    [HCFF, ECFF, HtCFF, EtCFF] = CFF_theo(xB, t, Q, Para_Unp, Para_Pol) # scalar for each of them
    # DVCS_pred_xBtQ = np.array(list(map(partial(DVCSxsec_theo, CFF_input = [HCFF, ECFF, HtCFF, EtCFF]), np.array(DVCSxsec_data_xBtQ))))
    DVCS_pred_xBtQ = DVCSxsec_theo(DVCSxsec_data_xBtQ, CFF_input = [HCFF, ECFF, HtCFF, EtCFF])
    return DVCS_pred_xBtQ, HCFF, ECFF, HtCFF, EtCFF


if __name__ == '__main__':
    pool = Pool()
    Paralst_Unp     = [4.923245780014504, 0.21630849765192428, 3.22890722499297, 2.3489354011979526, 0.16346585165260322, 1.135711231720756, 6.897907293004202, 0.15, 3.3570431064205124, 0.18430950949075475, 4.4164330846199, 3.474060988244228, 0.24911513048067477, 1.051977198682268, 6.546615550866641, 2.8639557573930126, 1.052330303667227, 7.412339020128528, 0.15, 0.16051495538648822, 0.9163110652883968, 1.0193962830435552, 0.41274932784170376, -0.1977991308710107, -1.2564920365188517, 1.067457804938632, -19.770730039892825, 1.0, 10.069595476994552, -3.2467026195137683, 1.0, -0.32089849276114984, 5.693312161571689, 0.0, -2.926022924238427, 0.6575758917146275, 0.0, 3.7811389938079354]
    Paralst_Pol     = [4.523625044880523, -0.24625833924504925, 3.0349443428078073, 2.6130504668028984, 0.07655463228254533, 0.5156744483888329, 4.371557716864458, 0.15, -0.7111561056489031, 0.2110536573461057, 3.2394039683470357, 4.37109681578387, -0.05648412944234933, 0.6133679635068163, 2.0897772013914437, 0.24322592280875213, 0.6308477000598409, 2.7182293377476165, 0.15, 8.93993973044334, 0.7999950847932062, 7.329257777405136, 2.043840761349927, -3.531387135862607, -1.267179457495525, 5.110641851061494, 42.0289770774691, 1.0, 2.092997157826637, -2.1779671813414163, 1.0, -1.5931689979366248, -12.338040498859877, 0.0, 2.6588214837192625, 36.25529526389492, 0.0, 2.805380264758739e-06]
    
    Para_Unp_All    = ParaManager_Unp(Paralst_Unp)
    Para_Pol_All    = ParaManager_Pol(Paralst_Pol)

    DVCS_HERA_pred = np.transpose([DVCSxsec_HERA_theo(DVCS_HERA_data, Para_Unp=Para_Unp_All, Para_Pol=Para_Pol_All)])
    with open("GUMP_Results/HERAxsec.csv","w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(DVCS_HERA_pred)

    """
    print(CFF_theo(0.36, -0.345, 2, Para_Unp_All, Para_Pol_All))

    print(CFF_theo(0.48, -0.702, 2, Para_Unp_All, Para_Pol_All))

    print(CFF_theo(0.60, -1.05, 2, Para_Unp_All, Para_Pol_All))
    """

    xsecgp0 = DVCSxsec_group_data[0]
    #print(xsecgp0)
    xsecgp0UU = xsecgp0[xsecgp0['pol'] == 'UU']
    xsecgp0LU = xsecgp0[xsecgp0['pol'] == 'LU']
    xsecgp0UU_pred, xsecgp0HCFF, xsecgp0ECFF, xsecgp0HtCFF, xsecgp0EtCFF =DVCSxsec_cost_xBtQ_plt(xsecgp0UU, Para_Unp_All,Para_Pol_All )
    xsecgp0LU_pred, xsecgp0HCFF, xsecgp0ECFF, xsecgp0HtCFF, xsecgp0EtCFF =DVCSxsec_cost_xBtQ_plt(xsecgp0LU, Para_Unp_All,Para_Pol_All )

    print(xsecgp0HCFF, xsecgp0ECFF, xsecgp0HtCFF, xsecgp0EtCFF)
    #plot_compare(xsecgp0UU['phi'], xsecgp0UU['f'], xsecgp0UU['delta f'], xsecgp0UU_pred)
    #plot_compare(xsecgp0LU['phi'], xsecgp0LU['f'], xsecgp0LU['delta f'], xsecgp0LU_pred)
    
    xsecUUgp0 = np.transpose(np.array([xsecgp0UU['phi'], xsecgp0UU['f'], xsecgp0UU['delta f'], xsecgp0UU_pred]))
    xsecLUgp0 = np.transpose(np.array([xsecgp0LU['phi'], xsecgp0LU['f'], xsecgp0LU['delta f'], xsecgp0LU_pred]))

    with open("GUMP_Results/UUxsec.csv","w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(xsecUUgp0)

    with open("GUMP_Results/LUxsec.csv","w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(xsecLUgp0)

    """
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
