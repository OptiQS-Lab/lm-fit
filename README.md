# OptiQS lm fit

This project is licensed under the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0).
  
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)


Levenberg Marquardt curve fitting (lm-fit) in 
- CPU
- CUDA

forked from [github.com/gpufit/Gpufit](https://github.com/gpufit/Gpufit) described in [Sci Rep 7, 15722 (2017)](https://www.nature.com/articles/s41598-017-15313-9).

## Binary distribution

The latest Gpufit binary release, supporting Windows 32-bit and 64-bit machines, can be found on the [release page](https://github.com/gpufit/Gpufit/releases).

## Documentation

Documentation for the lm-fit library is build autmatically using GitHub Pages <https://optiqs-lab.github.io/lm-fit/> using the markdown files provided in [docs](docs) directory.
## Building Gpufit from source code

Instructions for building Gpufit are found in the documentation: [Building from source code](https://github.com/gpufit/Gpufit/blob/master/docs/installation.rst).

## Using the Gpufit binary distribution

Instructions for using the binary distribution may be found in the documentation.  The binary package contains:

- The Gpufit SDK, which consists of the 32-bit and 64-bit DLL files, and 
  the Gpufit header file which contains the function definitions.  The Gpufit
  SDK is intended to be used when calling Gpufit from an external application
  written in e.g. C code.
- Gpufit Performance test: A simple console application comparing the execution speed of curve fitting on the GPU and CPU.  This program also serves as a test to ensure the correct functioning of Gpufit.
- Python version >3.12 bindings (compiled as wheel files) and
  Python examples.


## Examples

There are various examples that demonstrate the capabilities and usage of Gpufit. They can be found at the following locations:

- /examples/c++ - C++ examples for Gpufit
- /examples/c++/gpufit_cpufit - C++ examples that use Gpufit and Cpufit
- /examples/python - Python examples for Gpufit including spline fit examples (also requires [Gpuspline](https://github.com/gpufit/Gpuspline))

## Authors

Gpufit was originally created by Mark Bates, Adrian Przybylski, Björn Thiel, and Jan Keller-Findeisen at the Max Planck Institute for Biophysical Chemistry, in Göttingen, Germany (see [GitHub Project account](https://github.com/gpufit))

This fork is maintained by the [Optical Quantum Systems](https://www.kip.uni-heidelberg.de/optiqs?lang=en) research group at the Kirchhoff-Institute for Physics in Heidelberg.

### How to cite Gpufit

If you use Gpufit in your research, please cite their publication describing the software.  A paper describing the software was published in Scientific Reports.  The open-access manuscript is available from the Scientific Reports website, [here](https://www.nature.com/articles/s41598-017-15313-9).

  *  Gpufit: An open-source toolkit for GPU-accelerated curve fitting  
     Adrian Przybylski, Björn Thiel, Jan Keller-Findeisen, Bernd Stock, and Mark Bates  
     Scientific Reports, vol. 7, 15722 (2017); doi: https://doi.org/10.1038/s41598-017-15313-9 