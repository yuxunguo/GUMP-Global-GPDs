# ===========================================================================
# In this example, we show how to use the gumpgpd package to 
# calculate predictions for certain observables at default kinematic points
#
# Please find in example 2 how to calculate for custom kinematic points
# ===========================================================================

from gumpgpd.Minimizer import *

if __name__ == '__main__':
    
    # The overhead for initizalization could take quite a few minutes
    # Then each task should take a few seconds to minutes to complete
    print('The overhead for initizalization could take quite a few minutes')
    print('Then each task should take a few seconds to minutes to complete')
    #===================== PDF calculation ===============================
    PDF_pred = PDF_theo(PDF_data,    Para = Para_Comb_off_forward)
    print("PDF calculation finished the results read:")
    print(PDF_pred.head())
    print('-----------------------------------------------------------------')
    
    #====================== GPD calculation ===============================
    GPD_pred = GPD_theo(GPD_data,    Para = Para_Comb_off_forward)
    print("GPD calculation finished the results read:")
    print(GPD_pred.head())
    print('-----------------------------------------------------------------')

    #====================== tPDF calculation ===============================
    tPDF_pred = tPDF_theo(tPDF_data, Para = Para_Comb_off_forward)
    print("tPDF calculation finished the results read:")
    print(tPDF_pred.head())
    print('-----------------------------------------------------------------')

    #====================== GFF calculation ================================
    GFF_pred = GFF_theo(GFF_data,    Para = Para_Comb_off_forward)
    print("GFF calculation finished the results read:")
    print(GFF_pred.head())
    print('-----------------------------------------------------------------')

    # Set P_order = 2 for NLO accuarcy, meson = 1 for rho meson, xsec_norm = 1 for rho-meson cross-section normalization
    #====================== DVCS calculation ===============================
    DVCS_pred_xBtQ       = DVCSxsec_theo(DVCSxsec_data,
                                         Para_Unp = Para_Unp_off_forward, 
                                         Para_Pol = Para_Pol_off_forward, 
                                         P_order = 2)
    print("DVCS calculation finished the results read:")
    print(DVCS_pred_xBtQ.head())
    print('-----------------------------------------------------------------')

    #====================== DVCS_HERA calculation ===============================
    DVCS_HERA_pred_xBtQ  = DVCSxsecHERA_theo(DVCSxsec_HERA_data, 
                                             Para_Unp = Para_Unp_off_forward, 
                                             Para_Pol = Para_Pol_off_forward,
                                             P_order = 2)
    print("DVCS_HERA calculation finished the results read:")
    print(DVCS_HERA_pred_xBtQ.head())
    print('-----------------------------------------------------------------')

    #====================== DVCS_Asym calculation ===============================
    DVCS_Asym_pred_xBtQ  = DVCSAsym_theo(DVCSAsym_data, 
                                         Para_Unp = Para_Unp_off_forward, 
                                         Para_Pol = Para_Pol_off_forward, 
                                         P_order = 2)
    print("DVCS Asym calculation finished the results read:")
    print(DVCS_Asym_pred_xBtQ.head())
    print('-----------------------------------------------------------------')

    #====================== DVrhoP H1 calculation ===============================
    DVrhoPH1_pred_xBtQ   = DVMPxsec_theo(DVrhoPH1xsecL_data, 
                                         Para_Unp = Para_Unp_off_forward, 
                                         xsec_norm = 1, meson = 1, p_order = 2)
    
    print("DVrhoP H1 calculation finished the results read:")
    print(DVrhoPH1_pred_xBtQ.head())
    print('-----------------------------------------------------------------')

    #====================== DVrhoP ZEUS calculation ===============================
    DVrhoPZEUS_pred_xBtQ = DVMPxsec_theo(DVrhoPZEUSxsecL_data, 
                                         Para_Unp = Para_Unp_off_forward, 
                                         xsec_norm = 1, meson = 1, p_order = 2)
    print("DVrhoP ZEUS calculation finished the results read:")
    print(DVrhoPZEUS_pred_xBtQ.head())
    print('-----------------------------------------------------------------')
