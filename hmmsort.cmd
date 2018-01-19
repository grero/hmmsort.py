universe = vanilla
execroot = /Volumes/DataX/Users/grogee/Documents/programming/hmmsort
executable = $(execroot)/dist/hmm_learn
sortdir = hmmsort
arguments = "--sourceFile $(fname) --iterations 3 --version 3 --chunkSize 100000"
should_transfer_files = YES
transfer_executable = true
when_to_transfer_output = ON_EXIT
transfer_input_files = $(dir)/$(fname)
output	=	$(dir)/$(sortdir)/hmmsort.out
error	=	$(dir)/$(sortdir)/hmmsort.err
log		=	$(dir)/$(sortdir)/hmmsort.log
getenv  = true
request_memory=6G
request_disk = 15G
RunAsOwner = false

queue 1 
