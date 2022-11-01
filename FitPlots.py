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
    [HCFF, ECFF, HtCFF, EtCFF] = CFF_theo(xB, t, Q, Para_Unp, Para_Pol)
    DVCS_pred_xBtQ = np.array(list(map(partial(DVCSxsec_theo, CFF_input = [HCFF, ECFF, HtCFF, EtCFF]), np.array(DVCSxsec_data_xBtQ))))
    return DVCS_pred_xBtQ, HCFF, ECFF, HtCFF, EtCFF


if __name__ == '__main__':
    pool = Pool()
    Paralst_Unp     = [4.923217839900655, 0.2163060129576606, 3.228925750381223, 2.3493354261241612, 0.16346679672382392, 1.1357132245889958, 6.897969180700207, 0.15, 3.3570782094516596, 0.18430549565759557, 4.416316231277839, 3.4721988925548706, 0.24911311573644498, 1.0519743916576973, 6.5462925888590595, 2.863953531291127, 1.0523304954325372, 7.412350083916277, 0.15, 6.276774021060197, 0.11670360566985849, 3.547396702804653, 4.587715450527496, -0.7142906394782781, 0.7632839180070121, 2.0763504665527757, 1.1274957356797919, 1.497966434121736, -3.2408671697246416, 5.699905821036576, 1.0, 7.835285455617766, -13.73529004595495, 1.0, 0.8701104635432858, -1.4867232698056145, 0.0, -2.1710158707342946, 3.3441086795926793, 0.0, 4.673094489772017]
    Paralst_Pol     = [4.5143494322720334, -0.2454214913083923, 3.0318031020006058, 2.6220176153953245, 0.07657586611377216, 0.5146695211814358, 4.37635963882342, 0.15, -0.7101347985853508, 0.21247704285712699, 3.2328291823725634, 4.441200599742609, -0.05562201795594495, 0.6147236921885604, 2.0777565438620855, 0.24341647746990971, 0.6306311366014787, 2.719806232443027, 0.15, 8.78184184809053, 0.7999999719162574, 7.296335404521038, 1.9934319167219052, -3.4960543432322893, -1.2776456807843937, 4.514881584041937, 35.4654224562875, 1.0, 1.0339060576780428, 9.430399080989018, 1.0, -1.3189468104713455, -9.521150279563958, 0.0, 2.978128102747831, 32.67808817882311, 0.0, 0.026774585332154932]
     
    Para_Unp_All    = ParaManager_Unp(Paralst_Unp)
    Para_Pol_All    = ParaManager_Pol(Paralst_Pol)

    DVCS_HERA_pred = np.array(list(pool.map(partial(DVCSxsec_HERA_theo, Para_Unp = Para_Unp_All, Para_Pol = Para_Pol_All), np.array(DVCS_HERA_data))))

    print(DVCS_HERA_pred)
    """
    print(CFF_theo(0.36, -0.345, 2, Para_Unp_All, Para_Pol_All))

    print(CFF_theo(0.48, -0.702, 2, Para_Unp_All, Para_Pol_All))

    print(CFF_theo(0.60, -1.05, 2, Para_Unp_All, Para_Pol_All))
    """

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

    with open("GUMP_test/UUxsec.csv","w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(xsecUUgp0)

    with open("GUMP_test/LUxsec.csv","w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(xsecLUgp0)

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
