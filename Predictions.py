from Parameters import ParaManager_Unp, ParaManager_Pol
from Observables import GPDobserv
from DVCS_xsec import dsigma_TOT, dsigma_DVCS_HERA, M
import numpy as np
import pandas as pd
from Minimizer import PDF_theo, tPDF_theo, GFF_theo, CFF_theo
from Minimizer import DVCSxsec_theo, DVCSxsec_cost_xBtQ, DVCSxsec_HERA_theo

df_Para = pd.read_csv('GUMP_Paras/params.csv', index_col=0)
para_list_unp = df_Para['values'][:38].to_numpy()
para_list_pol = df_Para['values'][:38].to_numpy()

Para_Unp = ParaManager_Unp(para_list_unp)
Para_Pol = ParaManager_Pol(para_list_pol)
Para_All = np.stack([Para_Unp, Para_Pol], axis=0)


def PDF(x, t, Q, flv, spe):
    '''
    Return parton distribution function (PDF) at a given point.

    Args:
        x: float or numpy array. 
        t: float or numpy array. 
        Q: float or numpy array. 
        flv: string or an array of string.
            flvs is the flavor. It can be 'u', 'd', 'g', 'NS', or 'S'
        spe: integer or numpy array of integer
            spes is the "species."
            0 means H
            1 means E
            2 means Ht
            3 means Et
        
    Returns:
        _pdf: Float or numpy array, depending on the input.
            Parton distribution function.
    '''
    # [x, t, Q, f, delta_f, spe, flv] = PDF_input
    _is_scalar = ( np.isscalar(x) and np.isscalar(t) and np.isscalar(Q)\
        and np.isscalar(flv) and np.isscalar(spe) )

    x = np.array(x)
    t = np.array(t)
    Q = np.array(Q)
    flv = np.array(flv)
    spe = np.array(spe)
 
    xi = 0

    p = np.where(spes<=1, 1, -1)

    '''
    if(spe == 0 or spe == 1):
       p =  1

    if(spe == 2 or spe == 3):
        p = -1
    '''
    # Para: (4, 2, 5, 1, 4)

    Para_spe = Para_All[spes] # fancy indexing. Output (N, 3, 5, 1, 5)
    _PDF_theo = GPDobserv(x, xi, t, Q, p)
    _pdf = _PDF_theo.tPDF(flvs, Para_spe)  # array length N

    if _is_scalar:
        _pdf = _pdf.item()

    return _pdf

tPDF = PDF


def GFF_theo(j, t, Q, flv, spe):
    '''
    Return Generalized Form Factors

    Args:
        j: int or numpy int array
        t: float or numpy array
        Q:float or numpy array
        flv: string or an array of string.
            flvs is the flavor. It can be 'u', 'd', 'g', 'NS', or 'S'
        spe: integer or numpy array of integer
            spes is the "species."
            0 means H
            1 means E
            2 means Ht
            3 means Et
    
    Returns:
        _gff: generalized form factors. Float or numpy array
    '''
    _is_scalar = ( np.isscalar(j) and np.isscalar(t) and np.isscalar(Q)\
        and np.isscalar(flv) and np.isscalar(spe) )

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
    p = np.where(spes<=1, 1, -1)
   

    Para_spe = Para_All[spes] # fancy indexing. Output (N, 3, 5, 1, 5)
    _GFF_theo = GPDobserv(x, xi, t, Q, p)
    _gff = _GFF_theo.GFFj0(j, flv, Para_spe) # (N)

    if _is_scalar:
        _gff = _gff.item()
    
    return _gff


def CFF(xB, t, Q):
    '''
    CFF

    Args
        xB: float or numpy array
        t: float or numpy array
        Q: float or numpy array

    Returns
        [ HCFF, ECFF, HtCFF, EtCFF ], each of which can be a scalar or numpy array
    '''
    xB  = np.array(xB)
    t   = np.array(t)
    Q   = np.array(Q)
    if np.ndim(xB*t*Q)==0:
        return CFF_theo(xB, t, Q, Para_Unp, Para_Pol)
    return CFF_theo(xB, t, Q, np.expand_dims(Para_Unp, axis=0), np.expand_dims(Para_Pol, axis=0))


def DVCSxsec(y, xB, t, Q, phi, pol):
    '''
    DVCS cross section

    Args
        y: float or numpy array
        xB: float or numpy array
        t: float or numpy array
        Q: float or numpy array
        phi: float or numpy array
        pol: string or a numpy array of str
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


def DVCSxsec_HERA(y, xB, t, Q, pol):
    '''
    The total DVCS cross-section integrated over phi

    Args
        y: float or numpy array
        xB: float or numpy array
        t: float or numpy array
        Q: float or numpy array
        phi: float or numpy array
        pol: string or a numpy array of str
            pol can be 'UU', 'LU', 'UL', 'LL',
                'UTin', 'LTin', 'UTout', or 'LTout'

    Returns
        The total DVCS cross-section integrated over phi. Float or numpy array
    '''
    y   = np.array(y)
    xB  = np.array(xB)
    t   = np.array(t)
    Q   = np.array(Q)
    pol = np.array(pol)

    [HCFF, ECFF, HtCFF, EtCFF] = CFF(xB, t, Q)
    return dsigma_DVCS_HERA(y, xB, t, Q, pol, HCFF, ECFF, HtCFF, EtCFF)