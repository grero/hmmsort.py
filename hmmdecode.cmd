universe = vanilla
execroot = /Volumes/DataX/Users/grogee/Documents/programming/hmmsort
executable = $(execroot)/run_hmm_decode.sh 
sortdir = hmmsort
tempfile = rpltemplates.hdf5
arguments = "/Applications/MATLAB_R2016b.app SourceFile $(fname) Group 1 fileName $(tempfile) save hdf5"
transfer_executable = yes 
should_transfer_files = yes
when_to_transfer_output = ON_EXIT
transfer_input_files = $(dir)/$(fname),$(dir)/$(sortdir)/$(tempfile), $(execroot)/hmm_decode.app
output	=	$(dir)/$(sortdir)/hmm_decode.out
error	=	$(dir)/$(sortdir)/hmm_decode.err
log		=	$(dir)/$(sortdir)/hmm_decode.log
getenv  = true
request_memory=2G
RunAsOwner = false
queue 1 
