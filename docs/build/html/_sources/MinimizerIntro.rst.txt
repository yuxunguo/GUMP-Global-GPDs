Minimizer Quickguide
====================

Here we briefly introduce the :ref:`minimizer module`. This part might be the one of the least reusable submodules in the package,
since it just minimize the reduced :math:`\chi^2` with givens set of observables. 
It highly depends on the choice of the data set and how the fitting is done.

In short, this submodules contains several pairs of cost and fit functions,
where the cost functions calculate the :math:`\chi^2` and the fit function tries to minimize it for a given set of parameters.

We utilize the ``iminuit`` package as the minimizer, find its `documentation <https://pypi.org/project/iminuit/>`_ on how the fit can be done/modified.
Any changes to the fit should be straightforward to made, though the code appears rather clumsy,
because we keep passing all the parameters and fixing the unused ones.