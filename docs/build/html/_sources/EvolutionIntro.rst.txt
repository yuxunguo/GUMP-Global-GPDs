Evolution Quick Guide
=====================

This page introduces :ref:`evolution module`, which can be used as a
stand-alone module for conformal-moment evolution and Wilson-coefficient
construction in DVCS/DVMP analyses.

Core parts:

- evolution operators,
- Wilson coefficients,
- evolved conformal moments.

Evolution Operators
-------------------

Anomalous dimensions and evolution basis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main anomalous-dimension callables are:

- :func:`Evolution.non_singlet_LO`
- :func:`Evolution.non_singlet_NLO`
- :func:`Evolution.singlet_LO`
- :func:`Evolution.singlet_NLO`

They return LO/NLO anomalous dimensions in the evolution basis for:

- vector channel ``p=1`` and axial-vector channel ``p=-1``;
- parity label ``prty=+1`` (singlet/gluon and non-singlet plus) or
    ``prty=-1`` (valence and non-singlet minus).

The flavor basis is
:math:`(q_{\mathrm{Val}}, q^{(+)}_{du}, q^{(-)}_{du}, \Sigma, G)`.

.. math::

    q_{\mathrm{Val}} = \sum_{i=u,d}(q_i-\bar q_i),

.. math::

    q^{(\pm)}_{ij} = (q_i \pm \bar q_i) - (q_j \pm \bar q_j),

.. math::

    \Sigma = \sum_{i=u,d}(q_i+\bar q_i).

Only :math:`\Sigma` mixes with the gluon under evolution.

Evolution-operator structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each conformal spin :math:`j`:

- non-singlet evolution is scalar,
- singlet-gluon evolution is a :math:`2\times2` matrix.

At NLO, off-diagonal couplings in :math:`j` appear
(:math:`j-k=2,4,\ldots`), so practical implementations are organized in
diagonal and off-diagonal pieces. See :func:`Evolution.Moment_Evo_NLO` for
the combined construction used in this codebase.

Wilson Coefficients
-------------------

The module provides Wilson coefficients for both processes and perturbative
orders:

- DVCS:
    :func:`Evolution.WilsonCoef_DVCS_LO`,
    :func:`Evolution.WilsonCoef_DVCS_NLO`
- DVMP:
    :func:`Evolution.WilsonCoef_DVMP_LO`,
    :func:`Evolution.WilsonCoef_DVMP_NLO`

For evolved-Wilson-coefficient workflows, use:

- :func:`Evolution.DVCS_WCoef_Evo_NLO`
- :func:`Evolution.DVMP_WCoef_Evo_NLO`

and then combine with moments via:

- :func:`Evolution.CFF_Evo_NLO_evWC` (DVCS),
- :func:`Evolution.TFF_Evo_NLO_evWC` (DVMP).

Moment Evolution
----------------

For direct evolution of conformal moments, the key APIs are:

- :func:`Evolution.Moment_Evo_LO`
- :func:`Evolution.Moment_Evo_LO_NSp1`
- :func:`Evolution.Moment_Evo_NLO`

LO utilities that already combine moments with LO Wilson coefficients are also
available:

- :func:`Evolution.CFF_Evo_LO`
- :func:`Evolution.TFF_Evo_LO`

In practice:

1. choose the channel and basis moments,
2. evolve to the target scale,
3. combine with the corresponding DVCS/DVMP Wilson coefficients,
4. use the result as Mellin-Barnes integrand input for CFF/TFF/GPD observables.

For implementation conventions, see function docstrings in
:mod:`gumpgpd.Evolution`.
