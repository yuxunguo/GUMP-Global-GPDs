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
    Paralst_Unp     = [4.922534295590349, 0.21634143179981935, 3.228610042709383, 2.3482459294135394, 0.16344275580252743, 1.1357400684186936, 6.895402708196721, 0.15, 3.3586572239218433, 0.1842426100086847, 4.417573382186983, 3.479117270371762, 0.24917750176639988, 1.0519216118153967, 6.551152830038727, 2.8642811429092463, 1.0523058466461475, 7.412779748354998, 0.15, 0.17039078923369722, 0.9117778159622452, 1.0586925188952074, 0.43575743065323375, -0.209955152625657, -0.5256575429614562, -1.3600129447064717, -19.770730039892825, 1.0, 10.267159566549998, -3.2467026195137683, 1.0, 0.37811459327629876, 5.693312161571689, 0.0, -2.1297383785670863, 0.6575758917146275, 0.0, 4.600577817679961]
    Paralst_Pol     = [4.51444669636376, -0.24542434834709703, 3.0318516798470454, 2.6220577389407684, 0.0765192697023151, 0.5149009611155857, 4.374109637271134, 0.15, -0.7101235808764668, 0.21248116095909975, 3.232764463064717, 4.4410688663255105, -0.055617289814115935, 0.6147308342956928, 2.077852173660539, 0.24325052551974666, 0.6308206072396163, 2.7184278210766615, 0.15, 8.786112107139164, 0.7999996653919439, 7.2970857442157655, 1.9946659519505927, -3.496916052403074, -22.311808008911193, 0.12347264320433687, 42.0289770774691, 1.0, -3.895036041601817, -2.1779671813414163, 1.0, -0.12885271382282817, -12.338040498859877, 0.0, 0.7759791414166142, 36.25529526389492, 0.0, 0.04486971215711455]
    
    Para_Unp_All    = ParaManager_Unp(Paralst_Unp)
    Para_Pol_All    = ParaManager_Pol(Paralst_Pol)

    print(CFF_theo(0.36, -0.345, 2, Para_Unp_All, Para_Pol_All))

    print(CFF_theo(0.48, -0.702, 2, Para_Unp_All, Para_Pol_All))

    print(CFF_theo(0.60, -1.05, 2, Para_Unp_All, Para_Pol_All))

    """
    DVCS_HERA_pred = np.transpose([DVCSxsec_HERA_theo(DVCS_HERA_data, Para_Unp=Para_Unp_All, Para_Pol=Para_Pol_All)])
    with open("GUMP_Results/HERAxsec.csv","w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(DVCS_HERA_pred)

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
