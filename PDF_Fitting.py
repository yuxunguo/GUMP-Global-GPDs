"""

Extract parameters from the PDF

"""
import numpy as np
import pandas as pd
from iminuit import Minuit

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
        m.migrad()
        m.hesse()
        return m
    
    def PDFparam(self):
        m = self.minuit_PDF()
        return np.transpose([np.array(m.values), np.array(m.errors)])


uV_Unp = PDFFit("PDFDATA/uV_Unp.csv")

dV_Unp = PDFFit("PDFDATA/dV_Unp.csv")

ubar_Unp = PDFFit("PDFDATA/ubar_Unp.csv")

dbar_Unp = PDFFit("PDFDATA/dbar_Unp.csv")

g_Unp = PDFFit("PDFDATA/g_Unp.csv")

uV_Pol = PDFFit("PDFDATA/uV_Pol.csv")

dV_Pol = PDFFit("PDFDATA/dV_Pol.csv")

qbar_Pol = PDFFit("PDFDATA/qbar_Pol.csv")

g_Pol = PDFFit("PDFDATA/g_Pol.csv")

print(g_Unp.minuit_PDF().params)

print(qbar_Pol.minuit_PDF().params)
