Parameters Quick Guide
======================

This page introduces :ref:`Parameters module`, which converts fit parameters
into conformal moments used by evolution and observable calculations.

Workflow: ``flat fit parameters -> structured tensors -> conformal moments``.

Parameter Managers
------------------

The two main manager functions are:

- :func:`Parameters.ParaManager_Unp` for vector-like GPD sector (:math:`H, E`),
- :func:`Parameters.ParaManager_Pol` for axial-vector-like sector (:math:`\tilde{H}, \tilde{E}`).

They map flat parameter lists into a standardized tensor layout used
throughout the code.

Manager output has shape:

.. math::

   (2, 3, 5, n_{\rm ansatz}, 6)

Axis meaning:

1. GPD type (for example :math:`H/E` or :math:`\tilde{H}/\tilde{E}`),
2. skewness expansion block (:math:`\xi^0`, :math:`\xi^2`, :math:`\xi^4`, ...),
3. flavor basis components,
4. ansatz-term index,
5. parameter index inside each ansatz term (normalization/intercepts/powers/slopes).

Input-Output Mapping
~~~~~~~~~~~~~~~~~~~~

The input to each manager is a 1-D parameter list from the fitter. The output
is a structured tensor where each element is a physically meaningful ansatz
coefficient block.

At the single-term level, each ansatz row is:

.. math::

   [N,\alpha,\beta,\alpha',b_{\rm exp},m^{-2}]

and controls the conformal moment behavior as follows:

- :math:`N` (``norm``): overall amplitude.
- :math:`\alpha` (``alpha``): small-:math:`x` Regge intercept.
- :math:`\beta` (``beta``): large-:math:`x` falloff power.
- :math:`\alpha'` (``alphap``): :math:`t`-dependence in the Regge shift.
- :math:`b_{\rm exp}` (``bexp``): exponential :math:`t` slope.
- :math:`m^{-2}` (``invm2``): dipole residual mass scale term.

For skewness blocks:

- axis-1 slice ``0`` is the base :math:`\xi^0` block,
- slices ``1`` and ``2`` are :math:`\xi^2` and :math:`\xi^4` blocks,
  generated from the base block by rescaling only the normalization entry via
  the corresponding :math:`R_{\xi^2}` and :math:`R_{\xi^4}` prefactors.

For species coupling:

- in :func:`Parameters.ParaManager_Unp`, sea/gluon ``E`` sectors are linked to
  the corresponding ``H`` sectors by ratio parameters
  (:math:`R_{E,\bar u}`, :math:`R_{E,\bar d}`, :math:`R_{E,g}`),
- in :func:`Parameters.ParaManager_Pol`, ``\tilde E`` sea/gluon sectors are
  linked to ``\tilde H`` through ``R_Et_Sea``.

So, the relation is: a flat fitter vector first selects physics coefficients,
then manager logic injects symmetry/ratio constraints, and finally produces the
full tensor consumed by moment evolution and observable reconstruction.

Implementation details from source code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The managers do more than reshaping flat lists:

1. Flavor-wise ansatz blocks are assembled explicitly for each species.
2. :math:`\xi^2` and :math:`\xi^4` blocks are built by rescaling only the
   normalization entry via ``np.einsum(..., [R,1,1,1,1,1])``.
3. A placeholder ansatz row ``[0,0,1,0,0,0]`` is used to keep tensor shapes
   regular when some sectors use fewer active terms.

For :func:`Parameters.ParaManager_Unp`:

- ``E`` sea/gluon components are tied to ``H`` via multiplicative ratios
  (:math:`R_{E,\bar u}`, :math:`R_{E,\bar d}`, :math:`R_{E,g}`).

For :func:`Parameters.ParaManager_Pol`:

- ``\tilde E`` sea/gluon components are tied to ``\tilde H`` via ``R_Et_Sea``.
- Several higher-order skewness prefactors are currently fixed to zero in-code
  (for example gluon :math:`\xi^2/\xi^4` prefactors), effectively turning off
  those terms in the default polarized setup.

This unified structure is the direct input for moment builders.

Moment Construction
-------------------

Two helper functions build conformal moments from parameter sets:

- :func:`Parameters.ConfMoment`
- :func:`Parameters.Moment_Sum`

:func:`Parameters.ConfMoment` computes conformal moments for each ansatz term,
while :func:`Parameters.Moment_Sum` sums over ansatz terms to obtain the total
moment per flavor component.

Both functions support scalar or vectorized conformal spin input :math:`j`
(for example :math:`j \in \mathbb{C}` or an array :math:`(N,)`).

Code-level moment formula
~~~~~~~~~~~~~~~~~~~~~~~~~

The implementation follows a KM-style ansatz with exponential :math:`t`-slope
and dipole residual factor:

.. math::

  F(j,t)=\frac{N}{B(2-\alpha,1+\beta)}
  B\!\left(j+1-\alpha-\alpha' t,1+\beta\right)
  e^{b_{\rm exp}t}(1-t\,m^{-2})^{-2}

where each ansatz term uses the parameter tuple
``[norm, alpha, beta, alphap, bexp, invm2]``.

Numerics and shape behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :func:`Parameters.beta_loggamma` evaluates Beta functions through
  ``loggamma`` for complex-number stability.
- :func:`Parameters.ConfMoment` reshapes ``j`` and ``t`` to broadcast against
  parameter tensors; scalar and batched kinematics are both supported.
- :func:`Parameters.Moment_Sum` performs the ansatz reduction with
  ``axis=-1`` after :func:`Parameters.ConfMoment` is evaluated.

Typical shape flow
~~~~~~~~~~~~~~~~~~

For a common use case with vectorized :math:`j`:

1. manager output (per species block): ``(3, 5, n_ansatz, 6)``
2. :func:`Parameters.ConfMoment` output: ``(N, 5, n_ansatz)``
3. :func:`Parameters.Moment_Sum` output: ``(N, 5)``

These ``(N, 5)`` moments are the direct inputs expected by evolution and
observable reconstruction routines.

How this connects to the rest of the package
--------------------------------------------

Outputs from this module are consumed by:

- :mod:`Evolution` for LO/NLO scale evolution in moment space,
- :mod:`Observables` for inverse Mellin and Mellin-Barnes reconstruction of
  tPDFs, GPDs, CFFs, and TFFs.