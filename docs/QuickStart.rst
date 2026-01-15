
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
(make sure that the correct branch is used!), and run ``pip install -e .`` in the root folder to install in editable mode.

The later mode allows you to edit the source code to generate results not directly accessible via the integrated interface.
This mode is only recommended if you are familiar with the GUMP code already. 
If you need interface for any customized observables not directly accessible in the public version, contact `Yuxun Guo <mailto:youuungx@gmail.com>`_ to request.

Parameters and model setting
----------------------------

The GUMP framework is written in a form that is convenient for cutomized GPD model.
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

**In most case, you won't need to modify them, unless you want to tune the parameters.**

In case when you need to modify the parameterization and modelling of GPDs directly,
they are provided in the :ref:`Parameters module`.
Note that the GPD models must be analytic in the complex j-space, 
so not all models can be directly implemented in the GUMP framework.

Calculation observables with existing interface
------------------------------------------------

With the above model we can in principle calcualte anythings that the :ref:`Observables module` allows. 
We start with observables with integrated interface, which are also presented 
in the Example folder of `GitHub <https://github.com/yuxunguo/GUMP-Global-GPDs/tree/GUMP1.0>`_.

Samples of GPDs and Generalized Form factors (GFFs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the ``gumpgpd`` package, we provide integrated interface to calculate grid of GPDs and their reductions:
Parton Distributions Functions (PDFs), t-dependent PDFs (tPDFs) and Generalized Form factors (GFFs).
To do so, we need the ``gM.Para_Comb_off_forward``. An example is presented below:

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

The above workflow can be easily generalized to calculate experimental observables.

.. code-block:: py
     :name: cross-sections and all

     from gumpgpd.Minimizer import *

     if __name__ == '__main__':
     
          DVCS_pred_xBtQ       = DVCSxsec_theo(DVCSxsec_data,
                                                  Para_Unp = Para_Unp_off_forward, 
                                                  Para_Pol = Para_Pol_off_forward, 
                                                  P_order = 2)
          DVCS_HERA_pred_xBtQ  = DVCSxsecHERA_theo(DVCSxsec_HERA_data, 
                                                  Para_Unp = Para_Unp_off_forward, 
                                                  Para_Pol = Para_Pol_off_forward,
                                                  P_order = 2)
          DVCS_Asym_pred_xBtQ  = DVCSAsym_theo(DVCSAsym_data, 
                                                  Para_Unp = Para_Unp_off_forward, 
                                                  Para_Pol = Para_Pol_off_forward, 
                                                  P_order = 2)
          DVrhoPH1_pred_xBtQ   = DVMPxsec_theo(DVrhoPH1xsecL_data, 
                                                  Para_Unp = Para_Unp_off_forward, 
                                                  xsec_norm = 1, meson = 1, p_order = 2)

Besides the parameters, we have P_order controlling the perturbative order: 1 for leading-order and 2 for next-to-leading order. 
(beyond not implemented yet). Meson = 1 for rho meson =3 for J/psi meson (Others not implemented yet). 
xsec_norm is an extra normalization typically not needed.

An extra care that needs to be taken care of is that, since the Compton/Transition form factors only depends on (xB,t,Q),
it would be preferrable to calculate cross-sections in groups of (xB,t,Q). 
The packge provide a ``group_by_unique()`` tool to do this with the following example:

.. code-block:: py
     :name: cross-sections group

     from gumpgpd.Minimizer import group_by_unique

     DVCSxsec_data = pd.read_csv(_DataFilePath_, header = None, names = ['y', 'xB', 't', 'Q', 'phi', 'f', 'delta f', 'pol'] , dtype = {'y': float, 'xB': float, 't': float, 'Q': float, 'phi': float, 'f': float, 'delta f': float, 'pol': str})
     DVCSxsec_group_data = group_by_unique(DVCSxsec_data)

Note that ``'f','delta f'`` is not complusory, while ``'pol'`` is. 
It must take the form of ``'PbPt'`` where ``'Pb'='U' or 'L'`` for beam polarization and 
``'Pt'='U', 'L', 'Tin', 'Tout'`` for target polarizations.
So a typical ``'pol'`` would be, e.g., ``'UTin'`` for unpolarized beam and transversely polarized target. 
Definition of the two different transversely polarizated target can be found here: `DVCS <https://inspirehep.net/literature/1925449>`_

Calculation observables without existing interface
--------------------------------------------------

There are also ways to calcuate observables that are not directly accessible with the above integrated dataframe interface.
They require you to directly call the corresponding function for the calculations. Here are some examples:

.. warning::   
   Working in progress.


Other possible calculations
--------------------------------------------------

The current framework also allows one to calculate things not implemented yet.
Nevertheless, it's not recommended unless you are familiar with their moment-space implementation as well as the GUMP framework.
Contact `Yuxun Guo <mailto:youuungx@gmail.com>`_ to request.
