
Quick start
===========
Introducation
-------------
Welcome to the GUMP program for the global analyses of Generalized Parton Distributions (GPDs).
The full name of GUMP is GPDs through Universal Moment Parameterization (GUMP).
This program aim to deliver the state-of-the-art tool for the global analyses of GPDs via moment-space apporach.
More resources and references are given at :ref:`Citation/Acknowledgement`.

Installation
------------
There are two ways to access this package: 
For ordinary user, ``pip install gumpgpd`` install the public version.
For the lastest developper version, download the source code from the `GitHub <https://github.com/yuxunguo/GUMP-Global-GPDs/tree/GUMP1.0>`_ page 
(make sure that the correct branch is used!), and run ``pip install -e .`` in the root folder to install in “editable” mode.

The later allows you to edit the source code to generate results not directly accessible via the given interface.
This mode is only recommended if you are familiar with the GUMP code already. 
If you need interface for any customized observables not directly accessible in the public version, contact `Yuxun Guo <mailto:youuungx@gmail.com>`_ to request.

Parameters and model setting
----------------------------

The GUMP framework is written in a form that is convenient for cutomized GPD model.
The GUMP parameterization and modelling of GPDs are given in the :ref:`Parameters module`.
Note that the model must be analytic in the complex j-space, 
so not all models can be directly implemented in the GUMP framework.

For ordinary user, the GUMP parameterization will be loaded by default. 
The parameters are obtained through a global analysis process and they are stored in ``gumpgpd.Minimizer``
and can be retrieved via:

.. code-block:: py
     :name: parameters input

     import gumpgpd.Minimizer as gM
     
     para_unp  = gM.Para_Unp_off_forward
     para_pol  = gM.Para_Pol_off_forward
     para_comb = gM.Para_Comb_off_forward

The above three high-dimensional numpy arrays stand for the best-fit parameters for the unpolarized (vector) and polarized (axial-vector) GPDs and their combination, respectively.

Calculation observables with existing interface
------------------------------------------------

With the above model we can in principle calcualte anythings that the :ref:`Observables module` allow to do. 
In the following, we present some simple examples that calculate the obserbales of interestes.

Leading-order Compton Form factors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For instance, we can use :meth:`Observables.GPDobserv.CFF` to calculate the Compton form factors (CFFs) 
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

The same for the LO Transition form factors (TFFs) with :meth:`Observables.GPDobserv.TFF`. 
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
Three functions can do this :meth:`Observables.GPDobserv.TFF` with ``p_order =2`` is equivalent to :meth:`Observables.GPDobserv.TFFNLO` .
Whereas the :meth:`Observables.GPDobserv.TFFNLO_evMom` uses the eolved moment method that provides a cross-check. 

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
