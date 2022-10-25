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


if __name__ == '__main__':
    pool = Pool()
    Paralst_Unp     = [4.922770899728711, 0.21631245457726278, 3.2286340744892907, 2.348399185175322, 0.16365640346071564, 1.1354420505970975, 6.900555781000984, 0.15, 3.3576171024691988, 0.18433322801878216, 4.417047711571693, 3.4784962312911554, 0.24919405603782252, 1.0519014521338468, 6.550675975040194, 2.8569361638914623, 1.052850395852861, 7.3858686738696075, 1.3667990030964654, 11.428596979560261, -0.14503367665445221, 3.758714482921486, 5.682818126206609, -0.04238153011014113, 0.9803227812334159, 0.4586142799424245, 0.09122382463597081, 0.5174928633254026, -3.6601890894524987, 4.4047879899243005, 1.0, 6.068048827873447, 30.197110804859197, 1.0, 1.1390141973542285, -1.542434376217511, 0.0, -1.6034835605512132, -10.27405958258355, 0.0, 5.230500613651823]
    Paralst_Pol     = [4.519500903078374, -0.24572273380859522, 3.0336506651929422, 2.6222628900332507, 0.07497508649129046, 0.5197103475539158, 4.325734887333323, 0.15, -0.7127190055054058, 0.21139941814694918, 3.2384085342954885, 4.446272032794327, -0.055483176661642805, 0.6154810679003333, 2.074752893822682, 0.24189131037183043, 0.6323075664721904, 2.7069132220353342, 1.1, 8.795795400853171, 0.7999999981931851, 7.29806279081493, 1.9980251635965138, -3.4981561238130356, -0.6401207838946621, 3.6721190840512543, 41.05312601873508, 1.0, 1.139988840989931, 10.38510432334534, 1.0, -1.1225695849394057, -11.859558195586501, 0.0, 3.052120336919783, 32.4062446087123, 0.0, 0.0063367065778857645]
    
    Para_Unp_All    = ParaManager_Unp(Paralst_Unp)
    Para_Pol_All    = ParaManager_Pol(Paralst_Pol)

    DVCS_HERA_pred = np.array(list(pool.map(partial(DVCSxsec_HERA_theo, Para_Unp = Para_Unp_All, Para_Pol = Para_Pol_All), np.array(DVCS_HERA_data))))

    #print(*DVCS_HERA_pred,sep='\n')

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
