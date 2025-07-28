#ifndef GPUFIT_SINE_CUH_INCLUDED
#define GPUFIT_SINE_CUH_INCLUDED

/* Description of the calculate_sine function
* ===================================================
*
* This function calculates the values of one-dimensional sine model functions
* and their partial derivatives with respect to the model parameters. 
*
* No independent variables are passed to this model function.  Hence, the
* X coordinate of the first data value is assumed to be 0.0.  For
* a fit size of N data points, the X coordinates of the data are
* simply the corresponding array index values of the data array, starting from
* zero.
*
* Parameters:
*
* parameters: An input vector of model parameters.
*             p[0]: amplitude
*             p[1]: frequency
*             p[2]: phase
*             p[3]: offset
*
* n_fits: The number of fits.
*
* n_points: The number of data points per fit.
*
* value: An output vector of model function values.
*
* derivative: An output vector of model function partial derivatives.
*
* point_index: The data point index.
*
* fit_index: The fit index.
*
* chunk_index: The chunk index. Used for indexing of user_info.
*
* user_info: An input vector containing user information.
*
* user_info_size: The size of user_info in bytes.
*
* Calling the calculate_sine function
* =======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_sine(
    REAL const * parameters,
    int const n_fits,
    int const n_points,
    REAL * value,
    REAL * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{
    // Points
    REAL x = point_index;
    // parameters
    REAL const * p = parameters;
    // value 
    value[point_index] = p[0] * sin(p[1] * x + p[2]) + p[3];


    // derivative
    REAL * current_derivative = derivative + point_index;
    // partial derivatives
    current_derivative[0 * n_points] = sin(p[1] * x + p[2]) + p[3];
    current_derivative[1 * n_points] = p[0] * cos(p[1] * x + p[2]) * x;
    current_derivative[2 * n_points] = p[0] * cos(p[1] * x + p[2]);
    current_derivative[3 * n_points] = 1;
}

#endif
