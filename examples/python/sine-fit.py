"""
Sine fit example

Requires pyGpufit and Numpy. Matplotlib.pyplot if one wants to plot results.
"""
import numpy as np
import pygpufit.gpufit as gf


def test_sine(n_fits: float = 1e6, points: int = 300) -> float:
    n_fits = int(n_fits)
    p = np.random.rand(n_fits, 4).astype(np.float32)
    # Set parameters in reasonable ranges
    p[:, 0] = p[:, 0]  # amplitude
    p[:, 1] = p[:, 1] * 2 + 2  # frequency
    p[:, 2] = p[:, 2] * np.pi - np.pi  # phase
    p[:, 3] = p[:, 3] - .5  # offset

    x = np.random.rand(n_fits, points).astype(np.float32) * 2 * np.pi
    y = p[:, 0:1] * np.sin(p[:, 1:2] * x + p[:, 2:3]) + p[:, 3:]

    # Retrieve estimate for initial parameters form data
    init_params = np.zeros((len(y), 4), dtype=np.float32)
    minimum = y[:].min(axis=1)
    maximum = y[:].max(axis=1)
    # Frequency of intenisty modulation
    frequency = 3  # One needs a good estimate for the frequency!
    offset = (maximum + minimum) / 2.0
    amplitude = (maximum - minimum) / 2.0
    phase = 0

    # Set Initial Fit Parameters
    init_params[:, 0] = amplitude
    init_params[:, 1] = frequency
    init_params[:, 2] = phase
    init_params[:, 3] = offset

    fit_p, _, _, number_iterations, execution_time = gf.fit(
        data=y,
        weights=None,
        model_id=gf.ModelID.SINE,
        initial_parameters=init_params,
        tolerance=1e-8,
        user_info=x,
        max_number_iterations=100000
        )

    max_error = np.max(abs(p - fit_p))
    print(number_iterations)
    print(f"{n_fits:4d} sine fits in {execution_time:2.2f} ms. Max abs error = {max_error:2.2e}")
    return max_error


iters = 1
n_fits = 10
max_errors = np.zeros(iters, dtype=float)
for i in range(iters):
    print(f"Test run {i+1:3d}")
    max_errors[i] = test_sine(n_fits)

print(f"Overall max error after : {np.max(max_errors):2.2e}")
