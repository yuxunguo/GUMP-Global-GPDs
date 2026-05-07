
Quick Start
===========

GUMP (GPDs through Universal Moment Parameterization) is a moment-space
framework for global GPD analysis.

For publication references and acknowledgements, see :doc:`Citation`.

Install
-------

Use either the released package or an editable source install.

Install the published package:

.. code-block:: bash

     pip install gumpgpd

Install the latest development version in editable mode:

.. code-block:: bash

     git clone https://github.com/yuxunguo/GUMP-Global-GPDs.git
     cd GUMP-Global-GPDs
     pip install -e .

Editable mode is recommended only if you plan to modify source code or add
custom observables.

Load Default Parameters
-----------------------

Default best-fit parameter sets are available from :mod:`gumpgpd.Minimizer`:

.. code-block:: python

     import gumpgpd.Minimizer as gM

     para_unp = gM.Para_Unp_off_forward
     para_pol = gM.Para_Pol_off_forward
     para_comb = gM.Para_Comb_off_forward

These arrays correspond to unpolarized, polarized, and combined fit
parameters, respectively.

If you want to build a custom parameterization, see :doc:`ParametersIntro`.

Compute Built-In Observables
----------------------------

With default parameters, you can evaluate PDFs, tPDFs, GPDs, and GFFs:

.. code-block:: python

     from gumpgpd.Minimizer import *

     if __name__ == '__main__':
           PDF_pred = PDF_theo(PDF_data, Para=Para_Comb_off_forward)
           tPDF_pred = tPDF_theo(tPDF_data, Para=Para_Comb_off_forward)
           GPD_pred = GPD_theo(GPD_data, Para=Para_Comb_off_forward)
           GFF_pred = GFF_theo(GFF_data, Para=Para_Comb_off_forward)

The ``if __name__ == '__main__':`` guard is required on some platforms because
these routines use multiprocessing.

Required DataFrame Columns
--------------------------

For custom calculations, provide DataFrames with required columns:

.. code-block:: python

     PDF_data_names = ['x', 't', 'Q', 'spe', 'flv']
     tPDF_data_names = ['x', 't', 'Q', 'spe', 'flv']
     GPD_data_names = ['x', 'xi', 't', 'Q', 'spe', 'flv']
     GFF_data_names = ['j', 't', 'Q', 'spe', 'flv']

Column conventions:

- ``x``, ``xi``, ``t`` are standard GPD kinematics.
- ``Q`` is the factorization scale.
- ``spe``: ``0,1,2,3`` for :math:`H, E, \tilde{H}, \tilde{E}`.
- ``flv``: ``'u'``, ``'d'``, ``'g'``, ``'NS'``, ``'S'``.

Compute Experimental Observables
--------------------------------

Built-in wrappers are also provided for DVCS and DVMP observables:

.. code-block:: python

     from gumpgpd.Minimizer import *

     if __name__ == '__main__':
           DVCS_pred = DVCSxsec_theo(
                 DVCSxsec_data,
                 Para_Unp=Para_Unp_off_forward,
                 Para_Pol=Para_Pol_off_forward,
                 P_order=2,
           )
           DVCS_HERA_pred = DVCSxsecHERA_theo(
                 DVCSxsec_HERA_data,
                 Para_Unp=Para_Unp_off_forward,
                 Para_Pol=Para_Pol_off_forward,
                 P_order=2,
           )
           DVCS_asym_pred = DVCSAsym_theo(
                 DVCSAsym_data,
                 Para_Unp=Para_Unp_off_forward,
                 Para_Pol=Para_Pol_off_forward,
                 P_order=2,
           )

Notes:

- ``P_order=1`` for LO, ``P_order=2`` for NLO.
- For DVMP, ``meson=1`` corresponds to :math:`\rho`, and ``meson=3`` to
  :math:`J/\psi`.
- ``xsec_norm`` is an additional normalization factor (usually left at
  default/reference values).

Group by Unique Kinematics
--------------------------

Cross-sections are most efficiently evaluated in groups of identical
:math:`(x_B, t, Q)` since CFF/TFF inputs are shared within each group.
Use :func:`gumpgpd.Minimizer.group_by_unique`:

.. code-block:: python

     from gumpgpd.Minimizer import group_by_unique

     grouped = group_by_unique(DVCSxsec_data)

For DVCS/DVMP tabular inputs, ``pol`` is required and should follow beam-target
notation:

- beam: ``U`` or ``L``
- target: ``U``, ``L``, ``Tin``, ``Tout``

Example label: ``UTin``.

Next Steps
----------

- For model tensors and moment construction, read :doc:`ParametersIntro`.
- For kinematic-point observables (GPD/CFF/TFF), read :doc:`ObservablesIntro`.
- For evolution details, read :doc:`EvolutionIntro`.
- For global fitting workflow, read :doc:`MinimizerIntro`.
