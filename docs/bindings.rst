.. _external-bindings:

=================
External bindings
=================

This sections describes the Gpufit bindings to other programming languages. The bindings (to Python) aim to
emulate the :ref:`c-interface` as closely as possible.

Most high level languages feature multidimensional numeric arrays. In the bindings implemented for Python,
we adopt the convention that the input data should be organized as a 2D array, with one dimension corresponding to the
number of data points per fit, and the other corresponding to the number of fits. Internally, in memory, these arrays should
always be ordered such that the data values for each fit are kept together. In this manner, the data in memory is ordered in the
same way that is expected by the Gpufit C interface, and there is no need to copy or otherwise re-organize the data
before passing it to the GPU. The same convention is used for the weights, the initial model parameters, and the output parameters.

Unlike the C interface, the external bindings do not require the number of fits and the number of data points per fit to be 
specified explicitly. Instead, these numbers are inferred from the dimensions of the 2D input arrays.

Optional parameters with default values
---------------------------------------

The external bindings make some input parameters optional. The optional parameters are shown here. They are kept the same
for all bindings.

:tolerance:
    default value 1e-4
:max_n_iterations:
    default value 25 iterations
:estimator_id:
    the default estimator is LSE as defined in constants.h_
:parameters_to_fit:
    by default all parameters are fit

For instructions on how to specify these parameters explicitly, see the sections below.

Python
------

The Gpufit binding for Python is a project named pyGpufit. This project contains a Python package named pygpufit, which
contains a module gpufit, and this module implements a method called fit. Calling this method is equivalent to
calling the C interface function :code:`gpufit()` of the Gpufit library. The package expects the input data to be
stored as NumPy array. NumPy follows row-major order by default.

Installation
++++++++++++

Wheel files for Python 2.X and 3.X on Windows 32/64 bit are included in the binary package. NumPy is required.

Install the wheel file with.

.. code-block:: bash

    pip install --no-index --find-links=LocalPathToWheelFile pyGpufit

Python Interface
++++++++++++++++

fit
...

The signature of the fit method (equivalent to calling the C interface function :code:`gpufit()`) is

.. code-block:: python

    def fit(data, weights, model_id:ModelID, initial_parameters, tolerance:float=None, max_number_iterations:int=None, parameters_to_fit=None, estimator_id:EstimatorID=None, user_info=None):

Optional parameters are passed in as None. The numbers of points, fits and parameters is deduced from the dimensions of
the input data and initial parameters arrays.

*Input parameters*

:data: Data
    2D NumPy array of shape (number_fits, number_points) and data type np.float32
:weights: Weights
    2D NumPy array of shape (number_fits, number_points) and data type np.float32 (same as data)

    :special: None indicates that no weights are available
:tolerance: Fit tolerance

    :type: float
    :special: If None, the default value will be used.
:max_number_iterations: Maximal number of iterations

    :type: int
    :special: If None, the default value will be used.
:estimator_id: estimator ID

    :type: EstimatorID which is an Enum in the same module and defined analogously to constants.h_.
    :special: If None, the default value is used.
:model_id: model ID

    :type: ModelID which is an Enum in the same module and defined analogously to constants.h_.
:initial_parameters: Initial parameters
    2D NumPy array of shape (number_fits, number_parameter)

    :array data type: np.float32
:parameters_to_fit: parameters to fit
    1D NumPy array of length number_parameter
    A zero indicates that this parameter should not be fitted, everything else means it should be fitted.

    :array data type: np.int32
    :special: If None, the default value is used.
:user_info: user info
    1D NumPy array of arbitrary type. The length in bytes is deduced automatically.

    :special: If None, no user_info is assumed.

*Output parameters*

:parameters: Fitted parameters for each fit
    2D NumPy array of shape (number_fits, number_parameter) and data type np.float32
:states: Fit result states for each fit
    1D NumPy array of length number_parameter of data type np.int32
    As defined in constants.h_:
:chi_squares: :math:`\chi^2` values for each fit
    1D NumPy array of length number_parameter of data type np.float32
:n_iterations: Number of iterations done for each fit
    1D NumPy array of length number_parameter of data type np.int32
:time: Execution time of call to fit
    In seconds.

Errors are raised if checks on parameters fail or if the execution of fit failed.

fit_constrained
...............

The :code:`fit_constrained` method is very similar to the :code:`fit` method with the additional possibility to
specify parameter constraints.

The signature of the :code:`fit_constrained` method (equivalent to calling the C interface function :code:`gpufit_constrained()`) is

.. code-block:: python

    def fit_constrained(data, weights, model_id:ModelID, initial_parameters, constraints=None, constraint_types=None, tolerance:float=None, max_number_iterations:int=None, parameters_to_fit=None, estimator_id:EstimatorID=None, user_info=None):

*Constraint input parameters*

:constraints: Constraint bound intervals for every parameter and every fit.
    2D NumPy array of shape (number_fits, 2*number_parameter) and data type np.float32
:contraint_types: Constraint types for every parameter
    1D NumPy array of length number_parameter
    Valid values are defined in gf.ConstraintType

get_last_error
..............

The signature of the get_last_error method (equivalent to calling the C interface function *gpufit_get_last_error*) is

.. code-block:: python

    def get_last_error():

Returns a string representing the error message of the last occurred error.

cuda_available
..............

The signature of the cuda_available method (equivalent to calling the C interface function *gpufit_cuda_available*) is

.. code-block:: python

    def cuda_available():

Returns True if CUDA is available and False otherwise.

get_cuda_version
................

The signature of the get_cuda_version method (equivalent to calling the C interface function *gpufit_get_cuda_version*) is

.. code-block:: python

    def get_cuda_version():

*Output parameters*

:runtime version: Tuple of (Major version, Minor version)
:driver version: Tuple of (Major version, Minor version)

An error is raised if the execution failed (i.e. because CUDA is not available).

Python Examples
+++++++++++++++

2D Gaussian peak example
........................

An example can be found at `Python Gauss2D example`_. It is equivalent to :ref:`c-example-2d-gaussian`.

The essential imports are:

.. code-block:: python

    import numpy as np
    import pygpufit.gpufit as gf


First we test for availability of CUDA as well as CUDA driver and runtime versions.

.. code-block:: python

    # cuda available checks
    print('CUDA available: {}'.format(gf.cuda_available()))
    print('CUDA versions runtime: {}, driver: {}'.format(*gf.get_cuda_version()))

The true parameters describing an example 2D Gaussian peak functions are:

.. code-block:: python

    # true parameters
    true_parameters = np.array((10, 5.5, 5.5, 3, 10), dtype=np.float32)

A 2D grid of x and y positions can conveniently be generated using the np.meshgrid function:

.. code-block:: python

    # generate x and y values
    g = np.arange(size_x)
    yi, xi = np.meshgrid(g, g, indexing='ij')
    xi = xi.astype(np.float32)
    yi = yi.astype(np.float32)

Using these positions and the true parameter values a model function can be calculated as

.. code-block:: python

    def generate_gauss_2d(p, xi, yi):
        """
        Generates a 2D Gaussian peak.
        http://gpufit.readthedocs.io/en/latest/api.html#gauss-2d

        :param p: Parameters (amplitude, x,y center position, width, offset)
        :param xi: x positions
        :param yi: y positions
        :return: The Gaussian 2D peak.
        """

        arg = -(np.square(xi - p[1]) + np.square(yi - p[2])) / (2*p[3]*p[3])
        y = p[0] * np.exp(arg) + p[4]

        return y

The model function can be repeated and noise can be added using the np.tile and np.random.poisson functions.

.. code-block:: python

    # generate data
    data = generate_gauss_2d(true_parameters, xi, yi)
    data = np.reshape(data, (1, number_points))
    data = np.tile(data, (number_fits, 1))

    # add Poisson noise
    data = np.random.poisson(data)
    data = data.astype(np.float32)

The model and estimator IDs can be set as

.. code-block:: python

    # estimator ID
    estimator_id = gf.EstimatorID.MLE

    # model ID
    model_id = gf.ModelID.GAUSS_2D

When all input parameters are set we can call the C interface of Gpufit.

.. code-block:: python

    # run Gpufit
    parameters, states, chi_squares, number_iterations, execution_time = gf.fit(data, None, model_id, initial_parameters, tolerance, max_number_iterations, None, estimator_id, None)

And finally statistics about the results of the fits can be displayed where the mean and standard deviation of the
fitted parameters are limited to those fits that converged.

.. code-block:: python

    # print fit results

    # get fit states
    converged = states == 0
    number_converged = np.sum(converged)
    print('ratio converged         {:6.2f} %'.format(number_converged / number_fits * 100))
    print('ratio max it. exceeded  {:6.2f} %'.format(np.sum(states == 1) / number_fits * 100))
    print('ratio singular hessian  {:6.2f} %'.format(np.sum(states == 2) / number_fits * 100))
    print('ratio neg curvature MLE {:6.2f} %'.format(np.sum(states == 3) / number_fits * 100))
    print('ratio gpu not read      {:6.2f} %'.format(np.sum(states == 4) / number_fits * 100))

    # mean, std of fitted parameters
    converged_parameters = parameters[converged, :]
    converged_parameters_mean = np.mean(converged_parameters, axis=0)
    converged_parameters_std = np.std(converged_parameters, axis=0)

    for i in range(number_parameters):
        print('p{} true {:6.2f} mean {:6.2f} std {:6.2f}'.format(i, true_parameters[i], converged_parameters_mean[i], converged_parameters_std[i]))

    # print summary
    print('model ID: {}'.format(model_id))
    print('number of fits: {}'.format(number_fits))
    print('fit size: {} x {}'.format(size_x, size_x))
    print('mean chi_square: {:.2f}'.format(np.mean(chi_squares[converged])))
    print('iterations: {:.2f}'.format(np.mean(number_iterations[converged])))
    print('time: {:.2f} s'.format(execution_time))


2D Gaussian peak constrained fit example
........................................

An example for a constrained fit can be found at `Python Gauss2D constrained fit example`_. It differs from the previous
example only in that constraints are specified additionally (as 2D array of lower and upper bounds on parameters for every
fit) as well as constraint types (for all parameters including fixed parameters) that can take a value of ConstraintType (FREE, LOWER, UPPER or LOWER_UPPER)
in order to either do not enforce the constraints for a parameter or enforce them only at the lower or upper or both bounds.

The following code block demonstrates how the sigma of a 2D Gaussian peak can be constrained to the interval [2.9, 3.1] and the background and ampltiude to non-negative values.

.. code-block:: python

    # set constraints
    constraints = np.zeros((number_fits, 2*number_parameters), dtype=np.float32)
    constraints[:, 6] = 2.9
    constraints[:, 7] = 3.1
    constraint_types = np.array([gf.ConstraintType.LOWER, gf.ConstraintType.FREE, gf.ConstraintType.FREE, gf.ConstraintType.LOWER_UPPER, gf.ConstraintType.LOWER], dtype=np.int32)

    # run constrained Gpufit
    parameters, states, chi_squares, number_iterations, execution_time = gf.fit_constrained(data, None, model_id,
                                                                                initial_parameters, constraints, constraint_types,
                                                                                tolerance, max_number_iterations, None,
                                                                                estimator_id, None)