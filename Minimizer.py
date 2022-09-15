from Observables import GPDobserv
from Parameters import ParaManager
import numpy as np

Init_Para_All = ParaManager([])[0]

TestGPD = GPDobserv(0.1, 0.1, 0, 5, 1)

import time

start = time.time()
print(TestGPD.CFF(Init_Para_All))
end = time.time()
print(end - start)

start = time.time()
print(TestGPD.GPD(Init_Para_All) * np.pi)
end = time.time()
print(end - start)

start = time.time()
print(TestGPD.tPDF(Init_Para_All) * np.pi)
end = time.time()
print(end - start)
