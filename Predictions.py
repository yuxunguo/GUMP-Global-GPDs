from Parameters import ParaManager_Unp, ParaManager_Pol
from Observables import GPDobserv
from DVCS_xsec import dsigma_TOT, dsigma_DVCS_HERA, M
import numpy as np
import pandas as pd
import csv 
from Minimizer import PDF_theo, tPDF_theo, GFF_theo, CFF_theo
from Minimizer import DVCSxsec_theo, DVCSxsec_cost_xBtQ, DVCSxsec_HERA_theo
from multiprocessing import Pool

df_Para = pd.read_csv('GUMP_Params/params.csv', index_col=0)
para_list_unp = df_Para['value'][:38].to_numpy()
para_list_pol = df_Para['value'][38:].to_numpy()

Para_Unp = ParaManager_Unp(para_list_unp)
Para_Pol = ParaManager_Pol(para_list_pol)
Para_All = np.concatenate([Para_Unp, Para_Pol], axis=0)


def PDF(x, t, Q, flv, spe):
    '''
    Return parton distribution function (PDF) at a given point.

    Args:
        x: numpy float array. 
        t: numpy float array.  
        Q: numpy float array. 
        flv: an array of string.
            flv is the flavor. It can be 'u', 'd', 'g', 'NS', or 'S'
        spe: numpy integer array
            spe is the "species."
            0 means H
            1 means E
            2 means Ht
            3 means Et
        
    Returns:
        _pdf: numpy float array.
            Parton distribution function.
    '''
    x = np.array(x)
    t = np.array(t)
    Q = np.array(Q)
    flv = np.array(flv)
    spe = np.array(spe)
 
    xi = 0

    p = np.where(spe<=1, 1, -1)

    '''
    if(spe == 0 or spe == 1):
       p =  1

    if(spe == 2 or spe == 3):
        p = -1
    '''
    # Para: (4, 2, 5, 1, 4)

    Para_spe = Para_All[spe] # fancy indexing. Output (N, 3, 5, 1, 5)
    _PDF_theo = GPDobserv(x, xi, t, Q, p)
    _pdf = _PDF_theo.tPDF(flv, Para_spe)  # array length N

    return _pdf

tPDF = PDF

def GPD(x, xi, t, Q, flv, spe):
    '''
    Return parton distribution function (PDF) at a given point.

    Args:
        x: numpy float array. 
        t: numpy float array.  
        Q: numpy float array. 
        flv: an array of string.
            flv is the flavor. It can be 'u', 'd', 'g', 'NS', or 'S'
        spe: numpy integer array
            spe is the "species."
            0 means H
            1 means E
            2 means Ht
            3 means Et
        
    Returns:
        _pdf: numpy float array.
            Parton distribution function.
    '''
    x = np.array(x)
    xi = np.array(xi)
    t = np.array(t)
    Q = np.array(Q)
    flv = np.array(flv)
    spe = np.array(spe)

    p = np.where(spe<=1, 1, -1)

    '''
    if(spe == 0 or spe == 1):
       p =  1

    if(spe == 2 or spe == 3):
        p = -1
    '''
    # Para: (4, 2, 5, 1, 4)

    Para_spe = Para_All[spe] # fancy indexing. Output (N, 3, 5, 1, 5)
    _GPD_theo = GPDobserv(x, xi, t, Q, p)
    _GPD = _GPD_theo.GPD(flv, Para_spe)  # array length N

    return _GPD

def GFF(j, t, Q, flv, spe):
    '''
    Return Generalized Form Factors

    Args:
        j: numpy int array
        t: numpy float array
        Q: numpy float array
        flv: an array of string.
            flv is the flavor. It can be 'u', 'd', 'g', 'NS', or 'S'
        spe: numpy array of integer
            spe is the "species."
            0 means H
            1 means E
            2 means Ht
            3 means Et
    
    Returns:
        _gff: generalized form factors. numpy float array
    '''

    j = np.array(j)
    t = np.array(t)
    Q = np.array(Q)
    flv = np.array(flv)
    spe = np.array(spe)
    x = 0
    xi = 0   
    '''
    if(spe == 0 or spe == 1):
        p = 1

    if(spe == 2 or spe == 3):
        p = -1
    '''
    p = np.where(spe<=1, 1, -1)
   

    Para_spe = Para_All[spe] # fancy indexing. Output (N, 3, 5, 1, 5)
    _GFF_theo = GPDobserv(x, xi, t, Q, p)
    _gff = _GFF_theo.GFFj0(j, flv, Para_spe) # (N)
    
    return _gff


def CFF(xB, t, Q):
    '''
    CFF

    Args
        xB: numpy float array
        t: numpy float array
        Q: numpy float array

    Returns
        [ HCFF, ECFF, HtCFF, EtCFF ], each of which is a numpy array
    '''
    xB  = np.array(xB)
    t   = np.array(t)
    Q   = np.array(Q)
    return CFF_theo(xB, t, Q, np.expand_dims(Para_Unp, axis=0), np.expand_dims(Para_Pol, axis=0))


def DVCSxsec(y, xB, t, Q, phi, pol):
    '''
    DVCS cross section

    Args
        y: numpy float array
        xB: numpy float array
        t: numpy float array
        Q: numpy float array
        phi: numpy float array
        pol: a numpy array of str
            pol can be 'UU', 'LU', 'UL', 'LL',
                'UTin', 'LTin', 'UTout', or 'LTout'

    Returns
        The DVCS cross section. Float or numpy array
    '''
    y   = np.array(y)
    xB  = np.array(xB)
    t   = np.array(t)
    Q   = np.array(Q)
    phi = np.array(phi)
    pol = np.array(pol)

    [HCFF, ECFF, HtCFF, EtCFF] = CFF(xB, t, Q)
    return dsigma_TOT(y, xB, t, Q, phi, pol, HCFF, ECFF, HtCFF, EtCFF)

if __name__ == '__main__':
    pool = Pool()

    x = np.exp(np.linspace(np.log(0.005), np.log(0.6), 100, dtype = float))

    uhPDF = PDF(x,[0.0],[2.0],['u'],[0])
    ubarhPDF = -PDF(-x,[0.0],[2.0],['u'],[0])
    dhPDF = PDF(x,[0.0],[2.0],['d'],[0])
    dbarhPDF = -PDF(-x,[0.0],[2.0],['d'],[0])

    with open("GUMP_Results/H_PDF.csv","w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([x,uhPDF,ubarhPDF,dhPDF,dbarhPDF]))

    uePDF = PDF(x,[0.0],[2.0],['u'],[1])
    ubarePDF = -PDF(-x,[0.0],[2.0],['u'],[1])
    dePDF = PDF(x,[0.0],[2.0],['d'],[1])
    dbarePDF = -PDF(-x,[0.0],[2.0],['d'],[1])

    with open("GUMP_Results/E_PDF.csv","w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([x,uePDF,ubarePDF,dePDF,dbarePDF]))

    uhtPDF = PDF(x,[0.0],[2.0],['u'],[2])
    ubarhtPDF = PDF(-x,[0.0],[2.0],['u'],[2])
    dhtPDF = PDF(x,[0.0],[2.0],['d'],[2])
    dbarhtPDF = PDF(-x,[0.0],[2.0],['d'],[2])

    with open("GUMP_Results/Ht_PDF.csv","w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([x,uhtPDF,ubarhtPDF,dhtPDF,dbarhtPDF]))

    uetPDF = PDF(x,[0.0],[2.0],['u'],[3])
    ubaretPDF = PDF(-x,[0.0],[2.0],['u'],[3])
    detPDF = PDF(x,[0.0],[2.0],['d'],[3])
    dbaretPDF = PDF(-x,[0.0],[2.0],['d'],[3])

    with open("GUMP_Results/Et_PDF.csv","w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([x,uetPDF,ubaretPDF,detPDF,dbaretPDF]))

    xgpd = np.concatenate((np.linspace(-0.6, -0.225, 16, dtype = float),np.linspace(-0.2, -0.1025, 40, dtype = float),np.linspace(-0.1, 0.1, 188, dtype = float),np.linspace(0.1025, 0.20, 40, dtype = float),np.linspace(0.225, 0.6, 16, dtype = float)))

    uhGPD = GPD(xgpd,[0.1],[0.0],[2.0],['u'],[0])
    dhGPD = GPD(xgpd,[0.1],[0.0],[2.0],['d'],[0])

    with open("GUMP_Results/H_GPD.csv","w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([xgpd,uhGPD,dhGPD]))

    ueGPD = GPD(xgpd,[0.1],[0.0],[2.0],['u'],[1])
    deGPD = GPD(xgpd,[0.1],[0.0],[2.0],['d'],[1])

    with open("GUMP_Results/E_GPD.csv","w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([xgpd,ueGPD,deGPD]))

    uhtGPD = GPD(xgpd,[0.1],[0.0],[2.0],['u'],[2])
    dhtGPD = GPD(xgpd,[0.1],[0.0],[2.0],['d'],[2])

    with open("GUMP_Results/Ht_GPD.csv","w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([xgpd,uhtGPD,dhtGPD]))

    uetGPD = GPD(xgpd,[0.1],[0.0],[2.0],['u'],[3])
    detGPD = GPD(xgpd,[0.1],[0.0],[2.0],['d'],[3])

    with open("GUMP_Results/Et_GPD.csv","w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([xgpd,uetGPD,detGPD]))
