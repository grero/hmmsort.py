universe = vanilla
executable = /Volumes/DataX/Users/grogee/Documents/programming/hmmsort/dist/hmm_learn
sortdir = hmmsort
arguments = "--sourceFile $(dir)/$(fname) --iterations 3 --version 3 --chunkSize 100000"
should_transfer_files = YES
transfer_executable = true
when_to_transfer_output = ON_EXIT
transfer_input_files = $(dir)/$(fname)
output	=	$(dir)/hmmsort/hmmsort.out
error	=	$(dir)/hmmsort/hmmsort.err
log		=	$(dir)/hmmsort/hmmsort.log
getenv  = true
request_memory=6G
request_disk = 15G
RunAsOwner = false

queue 1 
