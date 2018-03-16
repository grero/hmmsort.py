#!/usr/bin/env python2.7
import sys
import os
import glob
import getopt
import subprocess

levels = ['day','session','array','channel']

def level(cwd):
     pp = cwd.split(os.sep)[-1]
     ll = ''
     if pp.isdigit():
         ll = 'day'
     else:
         ll = pp.strip(''.join([str(i) for i in xrange(10)]))
     return ll
        

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], '', longopts=['dry-run']) 
    dopts = dict(opts)
    if len(args) == 0:
        print "Usage: hmmsort_pbs.py [ --dry-run ] <execroot>"
        sys.exit(0)
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
    for i,f in enumerate(files):
        fname_learn = "learn_job%.4d.pbs" %(i,)
        fname_decode = "decode_job%.4d.pbs" %(i,)
        pp = f.split(os.sep)
        dd = os.sep.join([os.getcwd()] +  pp[:-1])
        fn = pp[-1]
        with open(fname_learn,"w") as fo:
            fo.write("#PBS -l nodes=1:ppn=1\n")
            fo.write("#PBS -l walltime=10:00:00\n")
            fo.write("#PBS -l mem=6GB\n")
            fo.write("cd %s\n" %(dd,))
            fo.write("%s/anaconda2/bin/hmm_learn.py --sourceFile %s --iterations 3 --version 3 " %(homedir,fn))
            fo.write("--chunkSize 100000 --outFile hmmsort/spike_templates.hdf5 ")
            fo.write("--max_size 1000000 --tempPath /hpctmp/lsihr/tmp/\n")
	    fo.write("--min_snr 4.0\n")

        if not "--dry-run" in dopts.keys():
            jobid = subprocess.check_output(['/opt/pbs/bin/qsub', fname_learn]).strip()
        with open(fname_decode,"w") as fo:
            fo.write("#PBS -l nodes=1:ppn=1\n")
            fo.write("#PBS -l walltime=12:00:00\n")
            if not "--dry-run" in dopts.keys():
                fo.write("#PBS -W depend=afterok:%s\n" %(jobid, ))
            fo.write("cd %s\n" %(dd,))
            fo.write("%s/run_hmm_decode.sh /app1/common/matlab/R2016a/ SourceFile %s Group 1 " %(execroot,fn))
            fo.write("fileName hmmsort/spike_templates.hdf5 save hdf5\n")
	    fo.write("SaveFile hmmsort.mat\n")

        if not "--dry-run" in dopts.keys():
             jobid = subprocess.check_output(['/opt/pbs/bin/qsub',fname_decode]).strip()
    sys.exit(0)
