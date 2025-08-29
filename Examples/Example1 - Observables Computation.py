# ===========================================================================
# In this code, we provide a example of how to use the gumpgpd package to 
# calculate predictions for certain observables at certain kinematic points
# ===========================================================================

from gumpgpd.Minimizer import *

if __name__ == '__main__':

    # Might take a few minutes for large data set
    
    PDF_pred = PDF_theo(PDF_data,    Para = Para_Comb_off_forward)
    GPD_pred = GPD_theo(GPD_data,    Para = Para_Comb_off_forward)
    tPDF_pred = tPDF_theo(tPDF_data, Para = Para_Comb_off_forward)
    GFF_pred = GFF_theo(GFF_data,    Para = Para_Comb_off_forward)
    
    # Use pool for parallel, other wise will be slow
    pool = get_pool()
    
    # Set P_order = 2 for NLO accuarcy, meson = 1 for rho meson, xsec_norm = 1 for rho-meson cross-section normalization
    DVCS_pred_xBtQ       = pd.concat(list(pool.map(partial(DVCSxsec_cost_xBtQ,      Para_Unp = Para_Unp_off_forward, Para_Pol = Para_Pol_off_forward, P_order = 2), DVCSxsec_group_data)), ignore_index=True)
    DVCS_HERA_pred_xBtQ  = pd.concat(list(pool.map(partial(DVCSxsec_HERA_cost_xBtQ, Para_Unp = Para_Unp_off_forward, Para_Pol = Para_Pol_off_forward, P_order = 2), DVCSxsec_HERA_group_data)), ignore_index=True)
    DVCS_Asym_pred_xBtQ  = pd.concat(list(pool.map(partial(DVCSAsym_cost_xBtQ,      Para_Unp = Para_Unp_off_forward, Para_Pol = Para_Pol_off_forward, P_order = 2), DVCSAsym_group_data)), ignore_index=True)
    DVrhoPH1_pred_xBtQ   = pd.concat(list(pool.map(partial(DVMPxsec_cost_xBtQ,      Para_Unp = Para_Unp_off_forward, xsec_norm = 1, meson = 1, p_order = 2), DVrhoPH1xsecL_group_data)), ignore_index=True)
    DVrhoPZEUS_pred_xBtQ = pd.concat(list(pool.map(partial(DVMPxsec_cost_xBtQ,      Para_Unp = Para_Unp_off_forward, xsec_norm = 1, meson = 1, p_order = 2), DVrhoPZEUSxsecL_group_data)), ignore_index=True)
    