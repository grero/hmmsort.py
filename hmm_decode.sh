#!/bin/sh
# script for execution of deployed applications
#
# Sets up the MATLAB Runtime environment for the current $ARCH and executes 
# the specified command.
#
exe_name=$0
exe_dir=`dirname "$0"`
MCRROOT=/Applications/MATLAB_R2016b.app
DYLD_LIBRARY_PATH=.:${MCRROOT}/runtime/maci64 ;
DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${MCRROOT}/bin/maci64 ;
DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${MCRROOT}/sys/os/maci64;
export DYLD_LIBRARY_PATH;
export MCR_CACHE_ROOT=/tmp/
args=
while [ $# -gt 0 ]; do
  token=$1
  args="${args} \"${token}\"" 
  shift
done
eval "\"${exe_dir}/hmm_decode.app/Contents/MacOS/hmm_decode\"" $args
exit

