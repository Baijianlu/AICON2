# AICON2

AICON2 is a program aims to achieve fast and accurate estimation of transport properties, such as electrical conductivity and thermal conductivity. The program is able to calculate electronic transport properties based on a [generalized Kane band model and perturbation theory in the framework of relaxation time approximation]( https://doi.org/10.1002/pssb.2220430102) and calculate lattice thermal conductivity based on the [Debye-Callaway model](https://link.aps.org/doi/10.1103/PhysRev.113.1046). We have optimized the original formula in order to achieve highly efficient numerical calculation. All the key input parameters can be calculated using first-principles methods. For more information, check out our [article1](https://doi.org/10.1016/j.cpc.2019.107074) and [article2](https://doi.org/10.1016/j.cpc.2021.108027).

AICON2 has a DOI:10.1016/j.cpc.2021.108027, you can cite this code like this:

    Tao Fan, Artem R. Oganov, AICON2: A program for calculating transport properties quickly and accurately, Computer Physics Communications, 2021, 108027. 

## prerequisites
AICON2 is a Python module and requires Python version 3.5 or higher. The dependent Python libraries include [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/), [spglib](https://atztogo.github.io/spglib/), [pymatgen](http://pymatgen.org/index.html). If you want to use automatic workflow management tools, [atomate](https://atomate.org/) and [FireWorks](https://materialsproject.github.io/fireworks/) should also be installed.  All of them can be easily obtained from the [Python Package Index](https://pypi.python.org/pypi) (PyPI), using tools such as pip. They may also be bundled with Python distributions aimed at scientists, like [Anaconda](https://anaconda.org/), and with a number of Linux distributions. Here we recommend to use Anaconda so that dependencies should be resolved automatically.

## Compiling and install AICON2
Install from pip:

    $ pip install AICON

Or, users installing from source must install the dependencies first and then run:

    $ python setup.py install
    
## Running the tests
Read /doc/UserManual to learn how to use this software and more information about the output files.
