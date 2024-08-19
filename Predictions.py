from Parameters import ParaManager_Unp, ParaManager_Pol
from Observables import GPDobserv
from DVCS_xsec import dsigma_TOT, dsigma_DVCS_HERA, M
import numpy as np
import pandas as pd
import csv 
from Minimizer import PDF_theo, tPDF_theo, GFF_theo, CFF_theo
from Minimizer import DVCSxsec_theo, DVCSxsec_cost_xBtQ, DVCSxsec_HERA_theo
from multiprocessing import Pool
import time
'''
df_Para = pd.read_csv('GUMP_Params/params.csv', index_col=0)
para_list_unp = df_Para['value'][:38].to_numpy()
para_list_pol = df_Para['value'][38:].to_numpy()

Para_Unp = ParaManager_Unp(para_list_unp)
Para_Pol = ParaManager_Pol(para_list_pol)
Para_All = np.concatenate([Para_Unp, Para_Pol], axis=0)
'''
Paralst_Pol     = [4.833430384423373, -0.26355746727810136, 3.1855567245326317, 2.1817250267982997, 0.06994083000560514, 0.5376473088622284, 4.22898219488582, 0.15, -0.663583721889865, 0.24767388786943867, 3.5722668493718626, 0.5420415127277624, -0.08640413690298866, 0.4946733452347538, 2.553713733867575, 0.24307061469378405, 0.6309890923077655, 2.716624295877619, 0.15, 7.99299605623125, 0.799997370438831, 6.415448025778247, 2.0758963463111515, -2.407059919688728, 37.65971219196447, 0.24589373380232807, 1.6561364171210822, 0.0, 2.6840962695831894, 37.58453653636456, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.852441955678458]
Paralst_Unp = [4.92252245341075, 0.21632833928300776, 3.228525762889928, 2.347470994624827, 0.16344460105600744, 1.135739437288775, 6.893895640954224, 0.15, 3.358767931921898, 0.1842893653407356, 4.417802345266761, 3.4816671934041685, 0.2491737223289409, 1.0519258916411531, 6.553873836594824, 1.8318696701278339, 1.0965234601821583, 9.99999305383342, 0.15, 0.1813228421702434, 0.9068471909677753, 1.1018931174030364, 0.4607676086634599, -0.22341404954304522, 0.7683213780361391, 0.22948701913308733, -2.638627981453611, -0.13980648369003512, 0.7985103392773935, 3.404262017724412, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.44764738950069, 1.7955307337138424]

Para_Unp = ParaManager_Unp(np.array(Paralst_Unp))
Para_Pol = ParaManager_Pol(np.array(Paralst_Pol))
Para_All = np.concatenate([Para_Unp, Para_Pol], axis=0)
"""
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
"""

if __name__ == '__main__':
    pool = Pool()
    '''
    x  =  0.1
    xi =  0.3
    t  = -0.5
    print(GPD([x,-x],[xi],[t],[2.0],['u'],[0]))
    print(GPD([x,-x],[xi],[t],[2.0],['d'],[0]))

    print(GPD([x,-x],[xi],[t],[2.0],['u'],[2]))
    print(GPD([x,-x],[xi],[t],[2.0],['d'],[2]))

    print(GPD([x,-x],[xi],[t],[2.0],['u'],[1]))
    print(GPD([x,-x],[xi],[t],[2.0],['d'],[1]))

    print(GPD([x,-x],[xi],[t],[2.0],['u'],[3]))
    print(GPD([x,-x],[xi],[t],[2.0],['d'],[3]))
    '''
    Para_spe = Para_All[0] 
    _GPD_theo = GPDobserv(0.1,0.1,-1.0,3.0,1)
    ts=time.time()
    _GPD1 = _GPD_theo.TFF(Para_spe,3,2,1)
    print(_GPD1)
    print(time.time()-ts)

    ts=time.time()    
    _GPD2 = _GPD_theo.TFFNLO_evMom(Para_spe,3)    
    print(_GPD2)
    print(time.time()-ts)
    '''
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
    '''
    '''
    xgpd = np.concatenate((np.linspace(-0.6,-0.33,28,dtype = float),np.linspace(-0.32,0.32,66,dtype = float),np.linspace(0.33,0.6,28,dtype = float)))

    uhGPD = GPD(xgpd,[0.33],[-0.5],[2.0],['u'],[0])
    dhGPD = GPD(xgpd,[0.33],[-0.5],[2.0],['d'],[0])

    with open("GUMP_Results/H_GPD.csv","w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([xgpd,uhGPD,dhGPD]))

    ueGPD = GPD(xgpd,[0.33],[-0.5],[2.0],['u'],[1])
    deGPD = GPD(xgpd,[0.33],[-0.5],[2.0],['d'],[1])

    with open("GUMP_Results/E_GPD.csv","w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([xgpd,ueGPD,deGPD]))

    uhtGPD = GPD(xgpd,[0.33],[-0.5],[2.0],['u'],[2])
    dhtGPD = GPD(xgpd,[0.33],[-0.5],[2.0],['d'],[2])

    with open("GUMP_Results/Ht_GPD.csv","w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([xgpd,uhtGPD,dhtGPD]))

    uetGPD = GPD(xgpd,[0.33],[-0.5],[2.0],['u'],[3])
    detGPD = GPD(xgpd,[0.33],[-0.5],[2.0],['d'],[3])

    with open("GUMP_Results/Et_GPD.csv","w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(np.transpose([xgpd,uetGPD,detGPD]))
    '''