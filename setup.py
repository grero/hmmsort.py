#!/usr/bin/env python
from distutils.core import setup,Extension
import numpy

setup(name='hmmsort',
    version='0.6.0',
    description='Python Npt tools',
    author='Roger Herikstad',
    author_email='roger.herikstad@gmail.com',
    packages=['hmmsort'],
    install_requires=['numpy',
                      'scipy',
                      'numba',
                      'blosc',
                      'h5py'],
    scripts=['hmmsort/hmm_learn.py',
             'hmmsort/create_spiketrains.py',
             'hmmsort/chunker.py',
             'scripts/hmmsort_pbs.py']
)
