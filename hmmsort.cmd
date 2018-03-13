universe = vanilla
execroot = /Volumes/DataX/Users/grogee/Documents/programming/hmmsort
executable = $(execroot)/hmm_learn
sortdir = hmmsort
maxsize = 8000000
outfile = spike_templates.hdf5
arguments = "--sourceFile $(fname) --iterations 3 --version 3 --chunkSize 100000 --outFile $(outfile) --max_size $(maxsize) --min_snr 4.0"
should_transfer_files = YES
transfer_executable = true
when_to_transfer_output = ON_EXIT
transfer_input_files = $(fname),$(outfile)
transfer_output_files = $(outfile)
output	=	$(sortdir)/hmmsort.out
stream_output = true
error	=	$(sortdir)/hmmsort.err
stream_error = true
log		=	$(sortdir)/hmmsort.log
getenv  = true
request_memory=6G
request_disk = 20G
RunAsOwner = false

queue 1 
