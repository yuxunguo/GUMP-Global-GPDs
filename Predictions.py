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


def PDF(xs, ts, Qs, flvs, spes):
    '''
    Return parton distribution function (PDF) at a given point.

    Args:
        xs: float or numpy array. 
        ts: float or numpy array. 
        Qs: float or numpy array. 
        flvs: string or an array of string.
            flvs is the flavor. It can be 'u', 'd', 'g', 'NS', or 'S'
        spes: integer or numpy array of integer
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
    _is_scalar = ( np.isscalar(xs) and np.isscalar(ts) and np.isscalar(Qs)\
        and np.isscalar(flvs) and np.isscalar(spes) )

    xs = np.array(xs)
    ts = np.array(ts)
    Qs = np.array(Qs)
    flvs = np.array(flvs)
    spes = np.array(spes)
 
    xi = 0

    ps = np.where(spes<=1, 1, -1)

    '''
    if(spe == 0 or spe == 1):
       p =  1

    if(spe == 2 or spe == 3):
        p = -1
    '''
    # Para: (4, 2, 5, 1, 4)

    Para_spe = Para_All[spes] # fancy indexing. Output (N, 3, 5, 1, 5)
    _PDF_theo = GPDobserv(xs, xi, ts, Qs, ps)
    _pdf = _PDF_theo.tPDF(flvs, Para_spe)  # array length N

    if _is_scalar:
        _pdf = _pdf.item()

    return _pdf

tPDF = PDF


def GFF_theo(js, ts, Qs, flvs, spes):
    '''
    Return Generalized Form Factors

    Args:
        js: int or numpy int array
        ts: float or numpy array
        Qs:float or numpy array
        flvs: string or an array of string.
            flvs is the flavor. It can be 'u', 'd', 'g', 'NS', or 'S'
        spes: integer or numpy array of integer
            spes is the "species."
            0 means H
            1 means E
            2 means Ht
            3 means Et
    
    Returns:
        _gff: generalized form factors. Float or numpy array
    '''
    _is_scalar = ( np.isscalar(js) and np.isscalar(ts) and np.isscalar(Qs)\
        and np.isscalar(flvs) and np.isscalar(spes) )

    js = np.array(js)
    ts = np.array(ts)
    Qs = np.array(Qs)
    flvs = np.array(flvs)
    spes = np.array(spes)
    x = 0
    xi = 0   
    '''
    if(spe == 0 or spe == 1):
        p = 1

    if(spe == 2 or spe == 3):
        p = -1
    '''
    ps = np.where(spes<=1, 1, -1)
   

    Para_spe = Para_All[spes] # fancy indexing. Output (N, 3, 5, 1, 5)
    _GFF_theo = GPDobserv(x, xi, ts, Qs, ps)
    _gff = _GFF_theo.GFFj0(js, flvs, Para_spe) # (N)

    if _is_scalar:
        _gff = _gff.item()
    
    return _gff