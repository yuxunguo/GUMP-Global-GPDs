Parameters Quickguide
=====================

Here we briefly introduce the :ref:`Parameters module`, the details can be found therein.
This submodules provide the GPDs in the moment space in terms of given parameter sets,
which can be devided into two parts: the parameters managers and the GPD moments.

Parameter Managers
------------------
This part consists of two function ``Parameters.ParaManager_Unp()`` and ``Parameters.ParaManager_Pol()``.
They are highly alike but defined separately for the vector-like GPDs :math:`H,E` and the axial-vector-like GPDs :math:`\tilde{H},\tilde{E}`.
Here we take the vector-like GPDs as an example.

The function ``Parameters.ParaManager_Unp()`` convert a list of all the phenomenological parameters into a set of parameters
in the standard shape of (2,3,5,n1,n2)

Each rows means:
 *   #1 = [0,1] corresponds to [H, E]
 *   #2 = [0,1,2,…] corresponds to [xi^0 terms, xi^2 terms, xi^4 terms, …]
 *   #3 = [0,1,2,3,4] corresponds to [u - ubar, ubar, d - dbar, dbar, g]
 *   #4 = [0,1,…,init_NumofAnsatz-1] corresponds to different set of parameters
 *   #5 = [0,1,2,3,…] correspond to [norm, alpha, beta, alphap,…] as a set of parameters

This standard shape will be handled by the remaining functions into the GPD moments.

GPD Moment Calculation
----------------------

There are two more function that convert the output from the previous subsection into GPD moments.

The function ``Parameters.ConfMoment()`` take input in the form of (5,n1,n2), 
which correspond to the last three dimension in the output of ``Parameters.ParaManager_Unp()`` and ``Parameters.ParaManager_Pol()``.
The last dimension contains a set of parameters that parameterize the moment of GPDs.
Therefore, the output moment will be in the shape of (5,n1)

The last function ``Parameters.Moment_Sum()`` essentially sums over the last dimension of ``Parameters.ConfMoment()``
--- it takes shape (5,n1) as introduced above and output shape (5,) by summing over the last dimension.

The two functions in this subsection also take vector input of j in shape (N,),
and thus output will be (N,5,n1) and (N,5), respectively.