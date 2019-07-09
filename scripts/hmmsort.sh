#!/bin/bash
if [ $# -lt 3 ]
then
	echo "Usage: hmmsort.sh group chunk sessionName"
	exit 0
fi
group=$1
chunk=$2
sessionName=$3

test -e hmmsort/${sessionName}_g`printf %.4d ${group}`.`printf %.4d ${chunk}`.mat ||  ( cd $PWD; $HOME/numba/bin/hmm_learn.py --sourceFile highpass/${sessionName}_highpass.`printf %.4d ${chunk}` --group ${group} --iterations 3 --version 3 --chunkSize 100000 && ( cd hmmsort; sh /opt/cluster/usr/bin/run_hmm_decode.sh /Applications/MATLAB_R2010a.app/ SourceFile ../highpass/${sessionName}_highpass.`printf %.4d ${chunk}` Group ${group} save fileName ${sessionName}_templatesg`printf %.4d ${group}`.`printf %.4d ${chunk}`.hdf5 hdf5) )
