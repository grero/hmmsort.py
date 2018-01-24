#!/bin/bash

TARGET=/opt/data2/Software/HTCondor

cp hmmsort_dag.py $TARGET/
cp dist/hmm_learn $TARGET/
rm -Rf $TARGET/hmm_decode.app
cp -r hmm_decode.app $TARGET/
cp run_hmm_decode.sh $TARGET/
cp hmmsort.cmd $TARGET/
cp hmmdecode.cmd $TARGET/
