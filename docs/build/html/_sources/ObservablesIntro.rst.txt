Observables Quick Guide
=======================

This page introduces :ref:`observables module`, the main interface for turning
parameterized conformal moments into physical observables with QCD evolution.

All high-level calculations are implemented in :class:`Observables.GPDobserv`.

Internally, this module combines:

- parameter moments from :mod:`Parameters`,
- LO/NLO evolution kernels from :mod:`Evolution`,
- inverse Mellin / Mellin-Barnes contour integrals.

Core class and initialization
-----------------------------

Create an observable object at fixed kinematics:

- :class:`Observables.GPDobserv`

Constructor arguments are:

- ``x``: parton momentum fraction,
- ``xi``: skewness,
- ``t``: momentum transfer squared,
- ``Q``: hard scale,
- ``p``: parity channel (``+1`` vector-like, ``-1`` axial-vector-like).

Flavor utilities
----------------

Module-level helpers provide flavor indexing/projections:

- :func:`Observables.flv_to_indx`
- :func:`Observables.flvs_to_indx`
- :func:`Observables.Flv_Intp`
- :func:`Observables.flvmask`

These are used internally to project evolved quantities onto
``u``, ``d``, ``g``, singlet/non-singlet, or all-flavor combinations.

Main observable methods
-----------------------

The most-used :class:`Observables.GPDobserv` methods are:

- :meth:`Observables.GPDobserv.tPDF` for :math:`f(x,t)`.
- :meth:`Observables.GPDobserv.GPD` for :math:`F(x,\xi,t)`.
- :meth:`Observables.GPDobserv.GFFj0` for generalized form factors.
- :meth:`Observables.GPDobserv.CFF` and :meth:`Observables.GPDobserv.CFFNLO`
  for DVCS Compton form factors.
- :meth:`Observables.GPDobserv.TFF` and :meth:`Observables.GPDobserv.TFFNLO`
  for DVMP transition form factors.

Alternative NLO implementations are also available:

- :meth:`Observables.GPDobserv.GPDNLO_evMom`
- :meth:`Observables.GPDobserv.CFFNLO_evMom`
- :meth:`Observables.GPDobserv.TFFNLO_evMom`

LO and NLO behavior
-------------------

For methods that accept ``p_order``:

- ``p_order=1`` selects LO evolution,
- ``p_order=2`` selects NLO evolution.

NLO paths combine this module with evolution backends in :mod:`Evolution`
(both evolved-moment and evolved-Wilson-coefficient strategies).

Input parameter structure
-------------------------

Methods expect parameter arrays produced by :mod:`Parameters` managers,
including forward and skewness-correction blocks
(:math:`\xi^0`, :math:`\xi^2`, :math:`\xi^4`).

In short:

1. build parameter tensors with :mod:`Parameters`,
2. evolve moments with :mod:`Evolution` (internally dispatched),
3. reconstruct observables via inverse Mellin or Mellin-Barnes integrals.

Practical note
--------------

This module is the recommended front-end when you need predictions at a given
kinematic point for fitting or plotting workflows.