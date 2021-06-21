#!/home/svu/lsihr/miniconda2/bin/python

import sys
import os
import glob
import getopt
import subprocess
import getpass

levels = ['day','session','array','channel']

def level(cwd):
     pp = cwd.split(os.sep)[-1]
     ll = ''
     if pp.isdigit():
         ll = 'day'
     else:
         numstr = [str(i) for i in xrange(10)]
         # sessioneye is a valid session direcory
         # so we want to add 'eye' to the list of valid suffixes
         # that will be removed properly
         #numstr.append('eye')
         ll = pp.strip(''.join(numstr))
     return ll


if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], '', longopts=['dry-run','jobids='])
    dopts = dict(opts)
    if len(args) == 0:
        print "Usage: hmmsort_pbs.py [ --dry-run ] [--jobisd=jid1,jid2,...] <execroot>"
        sys.exit(0)
    if os.path.isfile("sorting_done") or os.path.isfile("sorting_inprogress"):
	sys.stdout.write("In progress. Skipping...\n")
	sys.stdout.flush()
	sys.exit(0)
    return_path = dopts.get("--return_path", "")
    execroot = args[0]
    thislevel = level(os.getcwd())
    # get all highpass datafiles
    levelidx = levels.index(thislevel)
    if levelidx == len(levels)-1:
        bb = "."
        ch = 1
    else:
        # construct a pattern for finding all highpass files below this level
        bb = os.sep.join([levels[i]+"*" for i in xrange(levelidx+1,len(levels))])
        ch = None
    bb = os.sep.join([bb] + ["*highpass.mat"])
    files = glob.glob(bb)
    homedir = os.path.expanduser('~')
    # save current directory path since we will be changing directories later
    currentdir = os.getcwd();
    for i,f in enumerate(files):
        fname_learn = "learn_job%.4d.pbs" %(i,)
        fname_decode = "decode_job%.4d.pbs" %(i,)
        pp = f.split(os.sep)
        dd = os.sep.join([currentdir] +  pp[:-1])
        if os.path.isfile(os.sep.join([dd, "hmmsort.mat"])):
            continue  # skip this channel if it already contains sorted data
        fn = pp[-1]
        # change directories so that the output and error files will be created
        # in the respective channel directores and will be easier to check on
        # the status of the sorting
        os.chdir(dd)
        with open(fname_learn,"w") as fo:
            fo.write("#PBS -l nodes=1:ppn=1\n")
            # increased request for CPU hours to make sure even long jobs will be able to complete
            fo.write("#PBS -l walltime=24:00:00\n")
            fo.write("#PBS -l mem=6GB\n")
	    fo.write("#PBS -q serial\n")
	    if "--jobids" in dopts.keys():
		fo.write("#PBS -W depend:afterok:%s\n" %( dopt["--jobids"].replace(",",";")),)
            fo.write("cd %s\n" %(dd,))
            fo.write("%s/anaconda2/bin/hmm_learn.py --sourceFile %s --iterations 3 --version 3 " %(homedir,fn))
            fo.write("--chunkSize 100000 --outFile hmmsort/spike_templates.hdf5 ")
	    fo.write("--min_snr 4.0 ")
            # get current username instead of hardcoding username
            fo.write("--max_size 1000000 --tempPath /hpctmp/%s/tmp/\n" %(getpass.getuser()))

        if not "--dry-run" in dopts.keys():
	    fii = open("sorting_inprogress","w")
	    fii.close()
            jobid = subprocess.check_output(['/opt/pbs/bin/qsub', fname_learn]).strip()

        with open(fname_decode,"w") as fo:
             # request more memory as some decode jobs were being killed for
             # exceeding the default 4 GB
	    fo.write("#!/bin/bash\n")
            fo.write("#PBS -l mem=10GB\n")
            # commenting out next line as it does not seem necessary
            # and because I would like to keep the jobid for hmm_learn on the
            # 3rd line since some scripts are expecting that
            # fo.write("#PBS -l nodes=1:ppn=1\n")
            # increased request for CPU hours to make sure even long jobs will be able to complete
            fo.write("#PBS -l walltime=24:00:00\n")
	    fo.write("#PBS -q serial\n")
            if not "--dry-run" in dopts.keys():
                fo.write("#PBS -W depend=afterok:%s\n" %(jobid, ))
            fo.write("cd %s\n" %(dd,))
            fo.write("%s/run_hmm_decode.sh /app1/common/matlab/R2016a/ SourceFile %s Group 1 " %(execroot,fn))
            fo.write("fileName hmmsort/spike_templates.hdf5 save hdf5 ")
            fo.write("SaveFile hmmsort.mat hdf5path after_noise\n")
            fo.write("rm sorting_inprogress\n")
	    fo.write("touch sorting_done\n")

        if not "--dry-run" in dopts.keys():
             jobid = subprocess.check_output(['/opt/pbs/bin/qsub',fname_decode]).strip()
    sys.exit(0)
