
Quick start
===========

Installation
------------
We are working to deliver this program as a python package once all the submodules are settled.
At this point the recommended installation method is to download the source file directly at `GUMP GitHub Page <https://github.com/yuxunguo/GUMP-Global-GPDs>`_,
and work in the root folder.


Parameters and model setting
----------------------------
To start with, you will need a set of parameter as well as models for the GPDs. 
In the GUMP framework, we consider modeling in the conformal moment space which have many benefits.
The details are documented in the :ref:`Parameters module`. 
In genearl, a GPD model should take a set of parameter and generate moments in the conformal spin j space for all flavors, since they will be mixed under evolutions.

For a quick start, we suggest using the GUMP parameterization and the best-fit parameters we obtained in previous work from 
`DVMP <https://inspirehep.net/literature/2833822>`_ and `DVCS <https://inspirehep.net/literature/2632776>`_ analyses.
These parameters are stored within the package:

.. code-block:: py
     :name: parameters input

     import pandas as pd
     import os
     dir_path = os.path.dirname(os.path.realpath(__file__))

     Paralst_Unp=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Unp.csv'), header=None).to_numpy()[0]
     Paralst_Pol=pd.read_csv(os.path.join(dir_path,'GUMP_Params/Para_Pol.csv'), header=None).to_numpy()[0]      

These parameter shall be fed to the parameters managers:

.. code-block:: py
     :name: parameters managers
     
     from Parameters import ParaManager_Unp, ParaManager_Pol
     Para_Unp = ParaManager_Unp(np.array(Paralst_Unp[:-2]))
     Para_Pol = ParaManager_Pol(np.array(Paralst_Pol))

Note that there are two parameters not needed for the GPDs (they are parameters for the cross-sections only).
So they shall not be fed to the parameter manager.
You can make your own model and modfies the parameter managers correspondingly.

Calculation of GPDs and observables
-----------------------------------
With the above model we can in principle calcualte anythings that the :ref:`Observables module` allow to do. 
In the following, we present some simple examples that calculate the obserbales of interestes.

Leading-order Compton Form factors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For instance, we can use ``Observables.GPDobserv.CFF()`` to calculate the Compton form factors (CFFs) 
at leading order (next-to-leading order working in progress). 
And we can test that the imaginary part of the leading order CFF agree with the GPD at :math:`x=\xi` as shown in the following:

.. code-block:: py
     :name: CFF LO

     from Observables import GPDobserv

     # Test of LO ImCFF and quark GPD evolved to mu =5 GeV
      
     Para_H = Para_Unp[0]  # Para_Unp = (Para_H ,Para_E) for the H and E GPDs respectively
     x=0.0001
     _GPD_theo = GPDobserv(x,x,0.0,5.0,1)  # Each obserbales requires (x,xi,t,mu,p)
     _GPD_theo2 = GPDobserv(-x,x,0.0,5.0,1)

     CFF = _GPD_theo.CFF(Para_spe,5.0)

     print(CFF)

     gpd1 = (_GPD_theo.GPD('u',Para_spe))* (2/3) ** 2
     gpd2 = (_GPD_theo2.GPD('u',Para_spe))* (2/3) ** 2
     gpd3 = (_GPD_theo.GPD('d',Para_spe))* (1/3) ** 2
     gpd4 = (_GPD_theo2.GPD('d',Para_spe))* (1/3) ** 2

     print(np.pi*(gpd1-gpd2+gpd3-gpd4))

Leading-order Transition Form factors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same for the LO Transition form factors (TFFs) with ``Observables.GPDobserv.TFF()``. 
Note that in our definition, the TFFs absorp many prefactors like the :math:`1/N_c` color factor and the meson decay constant
:math:`f_{\phi}` and the charge factor :math:`e_c=2/3` for charm quark, and so on. 

.. code-block:: py
     :name: TFF LO

     # Test of LO ImTFF and gluon GPD evolved to mu = 5 GeV

     Para_H = Para_Unp[0]
     x=0.0001

     _GPD_theo = GPDobserv(x,x,0.0,5.0,1)
     TFF = _GPD_theo.TFF(Para_spe,5.0,3)
     print(TFF)

     gpd1 = (_GPD_theo.GPD('g',Para_spe))
     f_jpsi= 0.406
     CF=4/3
     NC=3
     prefact = np.pi * 3 * f_jpsi / NC /x * 2/3

     print(prefact*gpd1)

NLO Transition Form factors
~~~~~~~~~~~~~~~~~~~~~~~~~~~
We can also calculate TFF at next-to-leading order (NLO) where we currently only has singlet and gluon contributions (non-singlet working in progress).
Three functions can do this ``Observables.GPDobserv.TFF()`` with ``p_order =2`` is equivalent to ``Observables.GPDobserv.TFFNLO()`` .
Whereas the ``Observables.GPDobserv.TFF_evMom()`` uses the eolved moment method that provides a cross-check. 

The results are virtually the same:

.. code-block:: py
     :name: TFF NLO

     # Test of two methods of calculating TFF evolved to mu =5 GeV
     
     Para_H = Para_Unp[0]
     x=0.0001
     _GPD_theo = GPDobserv(x,x,0.0,5.0,1)
     TFF1 = _GPD_theo.TFFNLO(Para_spe,5.0, meson = 3, flv ='All')
     print(TFF1)
     TFF2 = _GPD_theo.TFFNLO_evMom(Para_spe,5.0, meson = 3, flv ='All')
     print(TFF2)

Some notes on genearl observables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It will be numerically demanding to generate GPDs at different x to calculate TFF/CFFs or other amplitudes,
since each point would requires an inverse transform to x-space that's essentially one or two (if NLO evolutions are used) layers of integral.
But this might be the only options if the Wilson coefficients in the conformal spin space are not known.

Be cautious!
