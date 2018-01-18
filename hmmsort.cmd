universe = vanilla
executable = /home/roger/numba/bin/hmm_learn.py
arguments = "--sourceFile $(dir)/highpass.mat --iterations 3 --version 3 --chunkSize 100000"
transfer_executable = false
output	=	$(dir)/hmmsort.out
error	=	$(dir)/hmmsort.err
log		=	$(dir)/hmmsort.log
getenv  = true
request_memory=6G
request_disk = 15G

queue 1 
