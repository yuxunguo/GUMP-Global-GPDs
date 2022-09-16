"""

Extract parameters from the PDF

"""
import numpy as np
import pandas as pd
from iminuit import Minuit

from DVCS_cross_section import M

#Using the ansatz norm * x ** (- alpha) * (1 - x) ** (beta) to fit PDF (equivalent to the moment space ansatz norm* beta( s- alpha, 1 + beta))
def PDF_Ansatz(x:float, norm: float, alpha: float, beta: float):
    return norm * x ** (- alpha) * (1 - x) ** (beta)

#PDF fit class, read the data from the file and run chi2 minimization using iminuit
class PDFFit(object) :
    def __init__(self, data_Path) -> None:
        self.data = pd.read_csv(data_Path, names = ["x", "f(x)","delta f(x)"])

    def chi2(self, norm: float, alpha: float, beta: float):
        PDF_Pred = PDF_Ansatz(self.data["x"],norm, alpha, beta)
        return np.sum(((PDF_Pred - self.data["f(x)"])/ self.data["delta f(x)"]) ** 2 )

    def minuit_PDF(self):

        m = Minuit(self.chi2, norm = 1 , alpha = 1, beta =2)
        m.errordef = Minuit.LEAST_SQUARES
        m.migrad()
        m.hesse()
        return m

uV_Unp_f = PDFFit("PDFDATA/uV_Unp.csv")
uV_Unp = np.array(uV_Unp_f.minuit_PDF().values)

dV_Unp_f = PDFFit("PDFDATA/dV_Unp.csv")
dV_Unp = np.array(dV_Unp_f.minuit_PDF().values)

ubar_Unp_f = PDFFit("PDFDATA/ubar_Unp.csv")
ubar_Unp = np.array(ubar_Unp_f.minuit_PDF().values)

dbar_Unp_f = PDFFit("PDFDATA/dbar_Unp.csv")
dbar_Unp = np.array(dbar_Unp_f.minuit_PDF().values)

g_Unp_f = PDFFit("PDFDATA/g_Unp.csv")
g_Unp = np.array(g_Unp_f.minuit_PDF().values)

uV_Pol_f = PDFFit("PDFDATA/uV_Pol.csv")
uV_Pol = np.array(uV_Pol_f.minuit_PDF().values)

print(uV_Pol_f.minuit_PDF().params)

dV_Pol_f = PDFFit("PDFDATA/dV_Pol.csv")
dV_Pol = np.array(dV_Pol_f.minuit_PDF().values)

print(dV_Pol_f.minuit_PDF().params)

qbar_Pol_f = PDFFit("PDFDATA/qbar_Pol.csv")
qbar_Pol = np.array(qbar_Pol_f.minuit_PDF().values)

print(qbar_Pol_f.minuit_PDF().params)

g_Pol_f = PDFFit("PDFDATA/g_Pol.csv")
g_Pol = np.array(g_Pol_f.minuit_PDF().values)
print(g_Pol_f.minuit_PDF().params)