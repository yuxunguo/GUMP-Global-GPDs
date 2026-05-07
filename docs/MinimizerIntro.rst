Minimizer Quick Guide
=====================

This page introduces :ref:`minimizer module`, which provides the global-fit
cost function and iMinuit-based minimization driver used in GUMP.

The module is analysis-specific (data selection, cuts, and fit strategy are
hard-wired), but it is the central entry point for reproducing the published
fit workflow.

What this module does
---------------------

At a high level, :mod:`Minimizer`:

- loads and preprocesses all fit datasets (PDF/tPDF, GFF, GPD, DVCS, DVMP),
- computes theory predictions from parameter vectors,
- builds per-point residual costs,
- adds sum-rule penalty terms,
- minimizes the total :math:`\chi^2` with iMinuit.

Core APIs
---------

Primary fit interfaces:

- :func:`Minimizer.cost_off_forward_withH_withHt`:
  full global cost function.
- :func:`Minimizer.off_forward_fit_withH_withHt`:
  iMinuit setup + MIGRAD + HESSE driver.

Important theory wrappers used inside the cost function:

- PDF/GFF/GPD sector:
  :func:`Minimizer.PDF_theo`,
  :func:`Minimizer.GFF_theo`,
  :func:`Minimizer.GPD_theo`
- DVCS/DVMP sector:
  :func:`Minimizer.DVCSxsec_theo`,
  :func:`Minimizer.DVCSAsym_theo`,
  :func:`Minimizer.DVCSxsecHERA_theo`,
  :func:`Minimizer.DVMPxsec_theo`
- Amplitude-level helpers:
  :func:`Minimizer.CFF_theo`,
  :func:`Minimizer.TFF_theo`

Parallel execution is handled through a shared process pool:

- :func:`Minimizer.get_pool`
- :func:`Minimizer.close_pool`
- :func:`Minimizer.group_by_unique`

Fit parameters
--------------

The iMinuit cost signature is flat (one argument per parameter), while the
implementation groups parameters into three blocks:

- ``Paralst_Unp_Names``: unpolarized parameters,
- ``Paralst_Pol_Names``: polarized parameters,
- ``Paralst_Aux_Names``: auxiliary parameters (currently ``jpsinorm``).

Internally these are validated by :func:`Minimizer.validate_params` and then
converted with :func:`gumpgpd.Parameters.ParaManager_Unp` and
:func:`gumpgpd.Parameters.ParaManager_Pol`.

Cost-function structure
-----------------------

The total objective is:

.. math::

   \chi^2_{\mathrm{total}} = \chi^2_{\mathrm{exp}} + \chi^2_{\mathrm{PDF/GFF/GPD}} + \chi^2_{\mathrm{penalty}}.

where:

- :math:`\chi^2_{\mathrm{exp}}` comes from DVCS/DVMP observables,
- :math:`\chi^2_{\mathrm{PDF/GFF/GPD}}` comes from PDF-like constraints,
- :math:`\chi^2_{\mathrm{penalty}}` enforces sum-rule constraints with
  configurable tolerances.

In export mode (``config.Export_Mode = True``), the cost function can return
grouped prediction DataFrames and write CSV outputs via
:func:`Minimizer.Export_Frame_Append`.

Running a fit
-------------

Use :func:`Minimizer.off_forward_fit_withH_withHt` with initial parameter
arrays in the exact order of ``Paralst_Unp_Names``, ``Paralst_Pol_Names``, and
``Paralst_Aux_Names``.

The routine:

1. builds a :class:`iminuit.Minuit` object,
2. applies parameter bounds/fixes,
3. runs ``migrad`` then ``hesse``,
4. writes a fit summary under ``GUMP_Output``.

Note: keep ``config.Export_Mode = False`` during minimization. Use export mode
only for post-fit prediction dumping.

Reference
---------

Minimization backend: `iminuit documentation <https://pypi.org/project/iminuit/>`_.