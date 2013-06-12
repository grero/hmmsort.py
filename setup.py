#!/usr/bin/env python
from distutils.core import setup,Extension
import numpy

setup(name='hmmsort',
    version='0.5',
    description='Python Npt tools',
    author='Roger Herikstad',
    author_email='roger.herikstad@gmail.com',
    packages=['hmmsort'],
    scripts=['hmmsort/hmm_learn.py']
)
