% makefile to create executables from the .cpp code. For Windows install
% MinGW and GNUMEX, to compile with the much faster gcc compiler. 

mex -DWIN32 -DGNUMEX viterbi_nd.cpp
