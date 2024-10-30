from Parameters import ParaManager_Unp, ParaManager_Pol
from Observables import GPDobserv
from DVCS_xsec import dsigma_TOT, dsigma_DVCS_HERA, M
import numpy as np
import pandas as pd
import csv 
from Minimizer import PDF_theo, tPDF_theo, GFF_theo, CFF_theo
from Minimizer import DVCSxsec_theo, DVCSxsec_cost_xBtQ, DVCSxsec_HERA_theo, DVJpsiPH1xsec_group_data
from multiprocessing import Pool
import time
from DVMP_xsec import dsigma_Jpsi_dt, M_jpsi
from functools import partial
import os
dir_path = os.path.dirname(os.path.realpath(__file__))


Paralst_Unp=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Unp.csv'), header=None).to_numpy()[0]
Paralst_Pol=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Pol.csv'), header=None).to_numpy()[0]

jpsinorm = Paralst_Unp[-2]

Para_Unp = ParaManager_Unp(np.array(Paralst_Unp[:-2]))
Para_Pol = ParaManager_Pol(np.array(Paralst_Pol))
Para_All = np.concatenate([Para_Unp, Para_Pol], axis=0)

def PDF_theo_s(x, t, Q, p, flv, Para, p_order):
    _PDF_theo = GPDobserv(x, 0, t, Q, p)
    return _PDF_theo.tPDF(flv, Para, p_order)

tPDF_theo_s = PDF_theo_s

def GPD_theo_s(x, xi, t, Q, p, flv, Para, p_order):
    _GPD_theo = GPDobserv(x, xi, t, Q, p)
    return _GPD_theo.GPD(flv, Para, p_order)

def rratio_theo_s(xi, t, Q, p, flv, Para, p_order):
    _GPD_theo = GPDobserv(xi, xi, t, Q, p)
    return _GPD_theo.GPD(flv, Para, p_order)/_GPD_theo.tPDF(flv, Para, p_order)

def HTFF_theo_jpsi(xB, t, Q, Para_spe, p_order = 1, muset =1, flv = 'All'):
    x = 0
    xi = (1/(2 - xB) - (2*t*(-1 + xB))/((Q**2 + M_jpsi**2)*(-2 + xB)**2))*xB
    H_E = GPDobserv(x, xi, t, Q, 1)
    HTFF_jpsi = H_E.TFF(Para_Unp[..., 0, :, :, :, :], muset * Q, 3, p_order, flv)
    
    return HTFF_jpsi

def TFF_theo_jpsi(xB, t, Q, Para_spe, p_order = 1, muset =1, flv = 'All'):
    x = 0
    xi = (1/(2 - xB) - (2*t*(-1 + xB))/((Q**2 + M_jpsi**2)*(-2 + xB)**2))*xB
    H_E = GPDobserv(x, xi, t, Q, 1)
    HTFF_jpsi = H_E.TFF(Para_Unp[..., 0, :, :, :, :], muset * Q, 3, p_order, flv)
    ETFF_jpsi = H_E.TFF(Para_Unp[..., 1, :, :, :, :], muset * Q, 3, p_order, flv)
    
    return  [ HTFF_jpsi, ETFF_jpsi ]

def DVjpsiPxsec_theo(DVjpsiPxsec_input: pd.DataFrame, TFF_jpsi_input: np.array):
    y = DVjpsiPxsec_input['y'].to_numpy()
    xB = DVjpsiPxsec_input['xB'].to_numpy()
    t = DVjpsiPxsec_input['t'].to_numpy()
    Q = DVjpsiPxsec_input['Q'].to_numpy()    
    [HTFF_jpsi, ETFF_jpsi] = TFF_jpsi_input
    return dsigma_Jpsi_dt(y, xB, t, Q, 0, HTFF_jpsi, ETFF_jpsi)

def DVjpsiPxsec_theo_xBtQ(DVjpsiPxsec_data_xBtQ: pd.DataFrame, Para_Unp, xsec_norm, p_order = 2, muset = 1, flv = 'All'):
    [xB, t, Q] = [DVjpsiPxsec_data_xBtQ['xB'].iat[0], DVjpsiPxsec_data_xBtQ['t'].iat[0], DVjpsiPxsec_data_xBtQ['Q'].iat[0]] 
    [HTFF_jpsi, ETFF_jpsi] = TFF_theo_jpsi(xB, t, Q, Para_Unp, p_order, muset, flv)
    DVjpsiP_pred_xBtQ = DVjpsiPxsec_theo(DVjpsiPxsec_data_xBtQ, TFF_jpsi_input = [HTFF_jpsi, ETFF_jpsi]) * xsec_norm**2
    return DVjpsiP_pred_xBtQ
'''
def DVjpsiPxsec_theo_scalar(y: float, xB: float, t: float, Q: float, Para_Unp, xsec_norm, p_order = 2, muset = 1, flv = 'All'):
    [HTFF_jpsi, ETFF_jpsi] = TFF_theo_jpsi(xB, t, Q, Para_Unp, p_order, muset, flv)
    return dsigma_Jpsi_dt(y, xB, t, Q, 0, HTFF_jpsi, ETFF_jpsi)* xsec_norm**2
'''

"""
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
"""

if __name__ == '__main__':
    pool = Pool()
    Para_spe = Para_All[0]
    '''
    # Test of LO ImCFF and quark GPD evolved to mu =5 GeV
    x=0.0001
    _GPD_theo = GPDobserv(x,x,0.0,5.0,1)
    _GPD_theo2 = GPDobserv(-x,x,0.0,5.0,1)
    ts=time.time()
    CFF = _GPD_theo.CFF(Para_spe,5.0)
    print(CFF)
    ts=time.time()
    gpd1 = (_GPD_theo.GPD('u',Para_spe))* (2/3) ** 2
    gpd2 = (_GPD_theo2.GPD('u',Para_spe))* (2/3) ** 2
    gpd3 = (_GPD_theo.GPD('d',Para_spe))* (1/3) ** 2
    gpd4 = (_GPD_theo2.GPD('d',Para_spe))* (1/3) ** 2

    print(np.pi*(gpd1-gpd2+gpd3-gpd4))

    # Test of LO ImTFF and gluon GPD evolved to mu = 5 GeV
    x=0.0001
    _GPD_theo = GPDobserv(x,x,0.0,5.0,1)
    ts=time.time()
    TFF = _GPD_theo.TFF(Para_spe,5.0,3)
    print(TFF)
    gpd1 = (_GPD_theo.GPD('g',Para_spe))
    f_jpsi= 0.406
    CF=4/3
    NC=3
    prefact = np.pi * 3 * f_jpsi / NC /x * 2/3
    print(prefact*gpd1)


    # Test of two methods of calculating TFF evolved to mu =5 GeV
    x=0.0001
    _GPD_theo = GPDobserv(x,x,0.0,5.0,1)
    TFF1 = _GPD_theo.TFFNLO(Para_spe,5.0, meson = 3, flv ='All')
    print(TFF1)
    TFF2 = _GPD_theo.TFFNLO_evMom(Para_spe,5.0, meson = 3, flv ='All')
    print(TFF2)
    print(TFF2-TFF1)
    '''
    #
    # Plotting results of the paper
    #
    

    # Comparing PDF with the global extraction of PDF

    x = np.exp(np.linspace(np.log(0.0001), np.log(0.05), 100, dtype = float))
    
    pdflst = np.array([PDF_theo_s(x_i,0.,2.,1,'g',Para_spe, 2) for x_i in x ])
    
    with open(os.path.join(dir_path,"GUMP_Results/Smallx_PDF.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([x,pdflst]))
        
    # Comparing GPD with the global extraction of PDF

    ts=time.time()
    x = np.exp(np.linspace(np.log(0.0014), np.log(0.05), 320, dtype = float))
    
    gpdlst = np.array([GPD_theo_s(x_i,0.002,0.,2.,1,'g',Para_spe, 2) for x_i in x ]).flatten()
    
    with open(os.path.join(dir_path,"GUMP_Results/Smallx_GPD.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([x,gpdlst]))
    print(time.time()-ts)

    # Ploting the R ratio

    x = np.exp(np.linspace(np.log(0.0001), np.log(0.01), 20, dtype = float))
    
    qlst = np.linspace(2,8, 20, dtype = float)
    
    xmesh, qmesh = np.meshgrid(x,qlst)
    
    xmeshflat = xmesh.flatten()
    
    qmeshflat = qmesh.flatten()

    rrat2dlst = np.array([rratio_theo_s(x_i,0.,q_i,1,'g',Para_spe, 2) for x_i,q_i in zip(xmeshflat,qmeshflat) ]).flatten()
    with open(os.path.join(dir_path,"GUMP_Results/Rrat2D.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([xmeshflat,qmeshflat,rrat2dlst]))

    '''
    # Comparing the cross-sections
    
    DVjpsiPH1_xBtQ_theo = np.array(list(pool.map(partial(DVjpsiPxsec_theo_xBtQ, Para_Unp = Para_Unp, xsec_norm = jpsinorm, p_order = 2, muset = 1, flv = 'All'), DVJpsiPH1xsec_group_data))).flatten()

    DVjpsiPH1_xBtQ_theo_mu_1 = np.array(list(pool.map(partial(DVjpsiPxsec_theo_xBtQ, Para_Unp = Para_Unp, xsec_norm = jpsinorm, p_order = 2, muset = np.sqrt(0.5), flv = 'All'), DVJpsiPH1xsec_group_data))).flatten()
    
    DVjpsiPH1_xBtQ_theo_mu_2 = np.array(list(pool.map(partial(DVjpsiPxsec_theo_xBtQ, Para_Unp = Para_Unp, xsec_norm = jpsinorm, p_order = 2, muset = np.sqrt(2), flv = 'All'), DVJpsiPH1xsec_group_data))).flatten()  
    
    DVjpsiPH1_xBtQ_theo_mu_3 = np.array(list(pool.map(partial(DVjpsiPxsec_theo_xBtQ, Para_Unp = Para_Unp, xsec_norm = jpsinorm, p_order = 2, muset = np.sqrt(0.75), flv = 'All'), DVJpsiPH1xsec_group_data))).flatten()
    
    DVjpsiPH1_xBtQ_theo_mu_4 = np.array(list(pool.map(partial(DVjpsiPxsec_theo_xBtQ, Para_Unp = Para_Unp, xsec_norm = jpsinorm, p_order = 2, muset = np.sqrt(1.5), flv = 'All'), DVJpsiPH1xsec_group_data))).flatten()  

    DVjpsiPH1_xBtQ_theo_mu_5 = np.array(list(pool.map(partial(DVjpsiPxsec_theo_xBtQ, Para_Unp = Para_Unp, xsec_norm = jpsinorm, p_order = 2, muset = np.sqrt(0.9), flv = 'All'), DVJpsiPH1xsec_group_data))).flatten()
    
    DVjpsiPH1_xBtQ_theo_mu_6 = np.array(list(pool.map(partial(DVjpsiPxsec_theo_xBtQ, Para_Unp = Para_Unp, xsec_norm = jpsinorm, p_order = 2, muset = np.sqrt(1.25), flv = 'All'), DVJpsiPH1xsec_group_data))).flatten()  

    DVJpsiPH1xsec_group_data_shape = np.array(DVJpsiPH1xsec_group_data).shape
    
    with open(os.path.join(dir_path,"GUMP_Results/dvjpsiph1data.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.array(DVJpsiPH1xsec_group_data).reshape(DVJpsiPH1xsec_group_data_shape[0],-1))
    
    with open(os.path.join(dir_path,"GUMP_Results/dvjpsiph1theo.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([DVjpsiPH1_xBtQ_theo,DVjpsiPH1_xBtQ_theo_mu_1,DVjpsiPH1_xBtQ_theo_mu_2,DVjpsiPH1_xBtQ_theo_mu_3,DVjpsiPH1_xBtQ_theo_mu_4,DVjpsiPH1_xBtQ_theo_mu_5,DVjpsiPH1_xBtQ_theo_mu_6]))

    # Decomposing into different orders and flavors
    # mulst =np.linspace(4,40, 20, dtype = float)
    qlst = np.linspace(2,20, 60, dtype = float)
    
    xb = 0.002
    
    TFFq1=np.array([HTFF_theo_jpsi(xb,-0.05,q_i,Para_spe, 1, 1, 'q') for q_i in qlst ]).flatten()
    TFFq2=np.array([HTFF_theo_jpsi(xb,-0.05,q_i,Para_spe, 2, 1, 'q') for q_i in qlst ]).flatten()
    TFFg1=np.array([HTFF_theo_jpsi(xb,-0.05,q_i,Para_spe, 1, 1, 'g') for q_i in qlst ]).flatten()
    TFFg2=np.array([HTFF_theo_jpsi(xb,-0.05,q_i,Para_spe, 2, 1, 'g') for q_i in qlst ]).flatten()
    
    TFFfull=np.array([HTFF_theo_jpsi(xb,-0.05,q_i,Para_spe, 2, 1, 'All') for q_i in qlst ]).flatten()
  
    with open(os.path.join(dir_path,"GUMP_Results/TFFqg12xb1.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([qlst,TFFfull,TFFq1,TFFq2,TFFg1,TFFg2]))
        
    xb = 0.005
    
    TFFq1=np.array([HTFF_theo_jpsi(xb,-0.05,q_i,Para_spe, 1, 1, 'q') for q_i in qlst ]).flatten()
    TFFq2=np.array([HTFF_theo_jpsi(xb,-0.05,q_i,Para_spe, 2, 1, 'q') for q_i in qlst ]).flatten()
    TFFg1=np.array([HTFF_theo_jpsi(xb,-0.05,q_i,Para_spe, 1, 1, 'g') for q_i in qlst ]).flatten()
    TFFg2=np.array([HTFF_theo_jpsi(xb,-0.05,q_i,Para_spe, 2, 1, 'g') for q_i in qlst ]).flatten()
    
    TFFfull=np.array([HTFF_theo_jpsi(xb,-0.05,q_i,Para_spe, 2, 1, 'All') for q_i in qlst ]).flatten()
  
    with open(os.path.join(dir_path,"GUMP_Results/TFFqg12xb2.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([qlst,TFFfull,TFFq1,TFFq2,TFFg1,TFFg2]))
    '''
    
    '''
    x = np.exp(np.linspace(np.log(0.005), np.log(0.6), 100, dtype = float))

    uhPDF = PDF(x,[0.0],[2.0],['u'],[0])
    ubarhPDF = -PDF(-x,[0.0],[2.0],['u'],[0])
    dhPDF = PDF(x,[0.0],[2.0],['d'],[0])
    dbarhPDF = -PDF(-x,[0.0],[2.0],['d'],[0])

    with open(os.path.join(dir_path,"GUMP_Results/H_PDF.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([x,uhPDF,ubarhPDF,dhPDF,dbarhPDF]))

    uePDF = PDF(x,[0.0],[2.0],['u'],[1])
    ubarePDF = -PDF(-x,[0.0],[2.0],['u'],[1])
    dePDF = PDF(x,[0.0],[2.0],['d'],[1])
    dbarePDF = -PDF(-x,[0.0],[2.0],['d'],[1])

    with open(os.path.join(dir_path,"GUMP_Results/E_PDF.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([x,uePDF,ubarePDF,dePDF,dbarePDF]))

    uhtPDF = PDF(x,[0.0],[2.0],['u'],[2])
    ubarhtPDF = PDF(-x,[0.0],[2.0],['u'],[2])
    dhtPDF = PDF(x,[0.0],[2.0],['d'],[2])
    dbarhtPDF = PDF(-x,[0.0],[2.0],['d'],[2])

    with open(os.path.join(dir_path,"GUMP_Results/Ht_PDF.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([x,uhtPDF,ubarhtPDF,dhtPDF,dbarhtPDF]))

    uetPDF = PDF(x,[0.0],[2.0],['u'],[3])
    ubaretPDF = PDF(-x,[0.0],[2.0],['u'],[3])
    detPDF = PDF(x,[0.0],[2.0],['d'],[3])
    dbaretPDF = PDF(-x,[0.0],[2.0],['d'],[3])
    
    with open(os.path.join(dir_path,"GUMP_Results/Et_PDF.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([x,uetPDF,ubaretPDF,detPDF,dbaretPDF]))
    '''
    '''
    xgpd = np.concatenate((np.linspace(-0.6,-0.33,28,dtype = float),np.linspace(-0.32,0.32,66,dtype = float),np.linspace(0.33,0.6,28,dtype = float)))

    uhGPD = GPD(xgpd,[0.33],[-0.5],[2.0],['u'],[0])
    dhGPD = GPD(xgpd,[0.33],[-0.5],[2.0],['d'],[0])

    with open(os.path.join(dir_path,"GUMP_Results/H_GPD.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([xgpd,uhGPD,dhGPD]))

    ueGPD = GPD(xgpd,[0.33],[-0.5],[2.0],['u'],[1])
    deGPD = GPD(xgpd,[0.33],[-0.5],[2.0],['d'],[1])

    with open(os.path.join(dir_path,"GUMP_Results/E_GPD.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([xgpd,ueGPD,deGPD]))

    uhtGPD = GPD(xgpd,[0.33],[-0.5],[2.0],['u'],[2])
    dhtGPD = GPD(xgpd,[0.33],[-0.5],[2.0],['d'],[2])

    with open(os.path.join(dir_path,"GUMP_Results/Ht_GPD.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([xgpd,uhtGPD,dhtGPD]))

    uetGPD = GPD(xgpd,[0.33],[-0.5],[2.0],['u'],[3])
    detGPD = GPD(xgpd,[0.33],[-0.5],[2.0],['d'],[3])

    with open(os.path.join(dir_path,"GUMP_Results/Et_GPD.csv"),"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([xgpd,uetGPD,detGPD]))
    '''