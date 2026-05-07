DVMP Quick Guide
================

This page briefly introduces :ref:`dvmp_xsec_module` for deeply virtual meson
production (DVMP) observables.

The module provides:

- the ratio :math:`R = \sigma_L/\sigma_T` for exclusive meson production,
- fitted :math:`R` parameters from combined H1+ZEUS :math:`\rho^0` data,
- longitudinal and total differential cross-sections
  :math:`d\sigma_L/dt` and :math:`d\sigma/dt`.

DVMP Main Callables
-------------------

- :func:`DVMP_xsec.R`: parametrization of :math:`R(Q)`.
- :func:`DVMP_xsec.R_rho_fit`: iMinuit fit of :math:`(a, p)` in :func:`DVMP_xsec.R`.
- :func:`DVMP_xsec.R_fitted`: fitted :math:`R(Q)` with propagated uncertainty.
- :func:`DVMP_xsec.epsilon`: virtual-photon polarization parameter
  :math:`\varepsilon`.
- :func:`DVMP_xsec.dsigmaL_DVMP_dt`: longitudinal cross-section
  :math:`d\sigma_L/dt`.
- :func:`DVMP_xsec.dsigma_DVMP_dt`: total cross-section
  :math:`d\sigma/dt = (d\sigma_L/dt)\,(\varepsilon + 1/R)`.

Meson Code Convention
---------------------

The ``meson`` argument follows the same integer convention across the module:

- ``1``: :math:`\rho^0`
- ``2``: :math:`\phi`
- ``3``: :math:`J/\psi`

In :func:`DVMP_xsec.MassCorr`, the heavy-meson correction uses
:math:`M_{J/\psi}` for ``meson=3`` and ``0`` otherwise.

DVMP Common Inputs
------------------

The DVMP cross-section functions use:

- ``y`` (float): inelasticity (beam energy-loss fraction).
- ``xB`` (float): Bjorken :math:`x_B`.
- ``t`` (float): momentum transfer squared (GeV^2).
- ``Q`` (float): photon virtuality (GeV).
- ``HTFF`` (complex): helicity-conserving TFF :math:`\mathcal{H}`.
- ``ETFF`` (complex): helicity-flip TFF :math:`\mathcal{E}`.

For :func:`DVMP_xsec.dsigma_DVMP_dt`, two additional fit parameters are used:

- ``a`` (float): fit parameter :math:`a` entering :func:`DVMP_xsec.R`.
- ``p`` (float): fit parameter :math:`p` entering :func:`DVMP_xsec.R`.

References
----------

- :func:`DVMP_xsec.R` and :func:`DVMP_xsec.epsilon` follow
  `arXiv:1112.2597 <https://arxiv.org/abs/1112.2597>`_.
- :func:`DVMP_xsec.dsigmaL_DVMP_dt` and :func:`DVMP_xsec.dsigma_DVMP_dt`
  follow `arXiv:2409.17231 <https://arxiv.org/abs/2409.17231>`_.