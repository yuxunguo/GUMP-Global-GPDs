DVCS Quick Guide
================

This page briefly introduces :ref:`dvcs_xsec_module` for deeply virtual Compton
scattering (DVCS) cross-section calculations. The formalism follows
`Guo et al. (2022) <https://inspirehep.net/literature/1925449>`_.

The implementation is generated from the Mathematica master expressions and
numerically cross-checked.

DVCS Main Callables
-------------------

The module provides three core differential cross-sections:

- :func:`DVCS_xsec.dsigma_BH` for the Bethe-Heitler (BH) contribution.
- :func:`DVCS_xsec.dsigma_DVCS` for the pure DVCS contribution.
- :func:`DVCS_xsec.dsigma_INT` for the BH-DVCS interference contribution.

Helpers:

- :func:`DVCS_xsec.dsigma_DVCS_TOT` for
  :math:`d\sigma_{\mathrm{BH}} + d\sigma_{\mathrm{DVCS}} + d\sigma_{\mathrm{INT}}`.
- :func:`DVCS_xsec.Asymmetry_DVCS_TOT` for spin asymmetries,
  :math:`A = d\sigma(\mathrm{pol}) / d\sigma(\mathrm{UU})`.
- :func:`DVCS_xsec.dsigma_DVCS_HERA` for the virtual-photon-proton cross-section
  integrated over :math:`\phi` (HERA convention).

DVCS Common Inputs
------------------

The differential functions use the following common inputs:

- ``y`` (float): beam energy-loss variable.
- ``xB`` (float): Bjorken :math:`x_B`.
- ``t`` (float): momentum transfer squared.
- ``Q`` (float): photon virtuality.
- ``phi`` (float): azimuthal angle.
- ``pol`` (str): polarization configuration label.
- ``HCFF`` (complex): Compton form factor :math:`H`.
- ``ECFF`` (complex): Compton form factor :math:`E`.
- ``HtCFF`` (complex): Compton form factor :math:`\widetilde{H}`.
- ``EtCFF`` (complex): Compton form factor :math:`\widetilde{E}`.

Polarization Convention
-----------------------

The polarization label uses beam-target form :math:`P_B P_T`, where
:math:`P_B \in \{U, L\}` and
:math:`P_T \in \{U, L, T_{\mathrm{in}}, T_{\mathrm{out}}\}`.

Supported labels are:

- ``UU``
- ``LU``
- ``UL``
- ``LL``
- ``UTin``
- ``LTin``
- ``UTout``
- ``LTout``

Example: ``pol = 'UTout'``.

HERA Observable
---------------

:func:`DVCS_xsec.dsigma_DVCS_HERA` integrates over :math:`\phi`, so ``phi`` is
not an input argument for this function.