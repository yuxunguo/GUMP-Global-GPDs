
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

In most case, you won't need to modify them, unless you want to tune the parameters.

Calculation observables with existing interface
------------------------------------------------

With the above model we can in principle calcualte anythings that the :ref:`Observables module` allow to do. 
We start with observables where interface are provided as part of the package, which are also presented 
in the Example folder of `GitHub <https://github.com/yuxunguo/GUMP-Global-GPDs/tree/GUMP1.0>`_.

Samples of GPDs and Generalized Form factors (GFFs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the ``gumpgpd`` package, we provide integrated interface to calculate grid of GPDs and Generalized Form factors (GFFs).
To do so, we need the ``para_comb = gM.Para_Comb_off_forward``. An example is presented below:

.. code-block:: py
     :name: GPD and all

     from gumpgpd.Minimizer import *

     if __name__ == '__main__':
     
          PDF_pred  = PDF_theo(PDF_data,    Para = Para_Comb_off_forward) #  PDF calculation
          tPDF_pred = tPDF_theo(tPDF_data,  Para = Para_Comb_off_forward) #  tPDF calculation
          GPD_pred  = GPD_theo(GPD_data,    Para = Para_Comb_off_forward) #  GPD calculation
          GFF_pred  = GFF_theo(GFF_data,    Para = Para_Comb_off_forward) #  GFF calculation

We emphasize that it is important to wrap them in ``if __name__ == '__main__':``
because we have used ``multiprocessing.Pool()`` to parallelize the calculations.

The input ``PDF_data``, ``GPD_data``, ``tPDF_data``, ``GFF_data`` are the data we used in the global analysis.
For customized calcualtions, generate your own dataframe following this example:

.. code-block:: py
     :name: data example

     xarr = np.linspace(0.1, 0.6, 50)    
     tarr = np.linspace(-0.5, 0., 2)    

     # Create all combinations using meshgrid and flatten
     X, T = np.meshgrid(xarr, tarr)
     x_list = X.flatten()
     t_list = T.flatten()

     # Create a DataFrame
     GPDs = pd.DataFrame({
               'x': x_list,
               'xi':0.3,
               't': t_list,
               'Q': 3.0,
               'spe': 0,
               'flv': 'NS'
               })

     result = GPD_theo(GPDs, Para=Para_Comb_off_forward)

The only requirements is that these dataframe must contain the needed columns which are:

.. code-block:: py
     :name: data format

     PDF_data_names = ['x', 't', 'Q', 'spe', 'flv']
     tPDF_data_names = ['x', 't', 'Q', 'spe', 'flv'] # The same as PDF
     GPD_data_names = ['x', 'xi', 't', 'Q', 'spe', 'flv']
     GFF_data_names = ['j', 't', 'Q', 'spe', 'flv']


We note that in the gumpgpd framework, ``'x', 'xi', 't'`` denote the standard GPD variable, 
``'Q'`` stands for the factorization scale, ``'spe'`` takes 0,1,2,3 for :math:`H,E,\tilde{H},\tilde{E}` GPDs
and ``'flv'= 'u','d','g','NS','S'`` for up quark, down quark, gluon, u-d, and u+d.
Again, those code are collected in the Example folder of `GitHub <https://github.com/yuxunguo/GUMP-Global-GPDs/tree/GUMP1.0>`_.

Calculation of experimental cross-sections and asymmetries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Calculation observables without existing interface
--------------------------------------------------

There are also ways to calcuate observables that are not directly accessible with the above integrated dataframe interface.
They require you to directly call the corresponding function for the calculations. Here are some examples:

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

Other possible calculations
--------------------------------------------------

The current framework also allows one to calculate things not implemented yet.
Nevertheless, it's not recommended unless you are familiar with their moment-space implementation as well as the GUMP framework.
Contact `Yuxun Guo <mailto:youuungx@gmail.com>`_ to request.
