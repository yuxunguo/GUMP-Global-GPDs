DVCS Quickguide
===============

Here we briefly introduce the :ref:`dvcs_xsec_module`.
This can be used as a stand-alone module for calculating the deeply virtual Compton scattering (DVCS) cross-section 
where the formulae are given in `our publication <https://inspirehep.net/literature/1925449>`_.

The codes are generated from the Mathematica master code, and numerically checked.

We have three main callables: :func:`DVCS_xsec.dsigma_DVCS`, :func:`DVCS_xsec.dsigma_INT()` and :func:`DVCS_xsec.dsigma_BH()`,
that calcualte the pure DVCS cross-sections, the interference cross-sections, and the Bethe-Heitler (BH) cross-sections, respectively.

Each function has the following arguments:
 *   y (float): Beam energy lost parameter
 *   xB (float): x_bjorken
 *   t (float): _description_
 *   Q (float): momentum transfer squared
 *   phi (float): azimuthal angel
 *   pol (string): polarization configuration
 *   HCFF (complex): Compton form factor H 
 *   ECFF (complex): Compton form factor E
 *   HtCFF (complex): Compton form factor Ht 
 *   EtCFF (complex): Compton form factor Et

The pol should be a string in the form of :math:`P_BP_T`
such that :math:`P_B = \{U,L\}` and :math:`P_T=\{U,L,T_{\rm{in}},T_{\rm{out}}\}` for the beam and target polarization,
e.g. ``pol = 'UTout'``.

Besides the three main callables, there is also :func:`DVCS_xsec.dsigma_TOT()` that takes the same input and returns the sum of the three functions above.
And the :func:`DVCS_xsec.dsigma_DVCS_HERA()` that return the virtual-photon-proton cross-sections and integrate over :math:`\phi`, as measured by HERA. The input :math:`\phi` will not be needed in this case.