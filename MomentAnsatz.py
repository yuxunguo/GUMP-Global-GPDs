"""

Comments Here

"""


inv_Mellin_intercept = 0.5

inv_Mellin_cutoff = 10

import time

from wolframclient.evaluation import WolframLanguageSession

from wolframclient.language import wl

session = WolframLanguageSession()

session.evaluate('MomentAnsatz[j_,k_,t_,n_,\[Alpha]_,\[Beta]_,\[Alpha]p_]:=n*Beta[j+1-\[Alpha],1+\[Beta]] (j+1-k-\[Alpha])/(j+1-k-\[Alpha]-\[Alpha]p*t);')

session.evaluate('reS=0.5')

session.evaluate('Cutoff = Infinity')

session.evaluate('tPDFInVMel[x_?NumericQ,t_?NumericQ,n_?NumericQ,\[Alpha]_?NumericQ,\[Beta]_?NumericQ,\[Alpha]p_?NumericQ]:=tPDFInVMel[x,t,n,\[Alpha],\[Beta],\[Alpha]p]=Re[NIntegrate[I/(2\[Pi] I) MomentAnsatz[reS-1+I imS,0,t,n,\[Alpha],\[Beta],\[Alpha]p]x^(-(reS+I imS)),{imS,-Cutoff,Cutoff}]]')

def t_PDF_Ansatz3():

    return session.evaluate('tPDFInVMel[0.5 , 0 , 1 , 0.1 , 3 , 1.5]')

start_time = time.time()
print(t_PDF_Ansatz3())
print("Third Evaluation of t-dependent PDF--- %s seconds ---" % (time.time() - start_time))
