In this folder, there are two functions, called:

- hmm_learn
- hmm_decode

These functions implement a spike sorting algorithm as described in the
paper "Spike sorting with hidden Markov models (Herbst, Gammeter, Ferrero,
Hahnloser 2008)".

hmm_learn:
Implementation of a 'n choose 1'-ring model learning algorithm, that
finds the spike templates, and the noise model. These parameters are then
used for the hmm_decode function.

hmm_decode:
The core functions are implemented in C++, that communicate with MATLAB
through a .mex function. The function viterbi_nd.cpp needs to be compiled 
(use makefile.m). Note that a compiler needs to be assigned in MATLAB. 
We worked on a Windows XP machine and installed the compiler according to 
http://gnumex.sourceforge.net/
Note that the code will in general run much faster than with the MATLAB
internal compiler.