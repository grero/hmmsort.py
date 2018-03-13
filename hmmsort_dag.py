#!/usr/bin/env python2.7
import sys
import os
import glob
import getopt

#TODO: Add a pre-screning function, e.g look at firing rate of multiunit activity

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
    opts, args = getopt.getopt(sys.argv[1:], 'h', longopts=['dry-run',
                                                             'use-julia'])
    dopts = dict(opts)
    if "-h" in dopts.keys():
        print "Usage: hmmsort_dag.py [ --dry-run ] [ execroot ]"
        sys.exit(0)
    if len(args) == 0:
        execroot,q = os.path.split(os.path.realpath(__file__))
    else:
        execroot = args[0]
    if '--use-julia' in dopts.keys():
        decode_cmd = "hmmdecode_julia.cmd"
    else:
        decode_cmd = "hmmdecode.cmd"

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
    
    with open("hmmsort.dag","w") as fid:
        for (jid,f) in enumerate(files):
            pp = f.split(os.sep)  # get the channel
            fn = pp[-1]
            dd = os.sep.join(pp[:-1])
            # make sure that the output dir exists
            outdir = os.sep.join([dd, "hmmsort"])
            if not os.path.isdir(outdir):
                os.mkdir(outdir)
            if thislevel != 'channel':
                ch = int(filter(str.isdigit, pp[-2]))
            fid.write('JOB hmmlearn_%d %s/hmmsort.cmd DIR %s\n' % (jid, execroot,dd))
            fid.write('VARS hmmlearn_%d fname="%s"\n' %(jid, fn))
            fid.write('VARS hmmlearn_%d execroot="%s"\n' %(jid, execroot))
            fid.write('VARS hmmlearn_%d outfile="spike_templates.hdf5"\n' %(jid, ))
            fid.write('SCRIPT PRE hmmlearn_%d %s/hmm_learn --initOnly --outFile spike_templates.hdf5\n' % (jid, execroot,))
            fid.write('JOB hmmdecode_%d %s/%s DIR %s\n' % (jid, execroot, decode_cmd, dd))
            fid.write('VARS hmmdecode_%d fname="%s"\n' %(jid, fn))
            fid.write('VARS hmmdecode_%d tempfile="spike_templates.hdf5"\n' %(jid, ))
            fid.write('PARENT hmmlearn_%d CHILD hmmdecode_%d\n' % (jid, jid))
            fid.write('\n')

    if not '--dry-run' in dopts.keys():
        os.system('condor_submit_dag -maxpre 10 hmmsort.dag')
    sys.exit(0)
