#!/usr/bin/env python

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
        numstr = [str(i) for i in range(10)]
        # sessioneye is a valid session direcory
        # so we want to add 'eye' to the list of valid suffixes
        # that will be removed properly
        #numstr.append('eye')
        ll = pp.strip(''.join(numstr))
    return ll


if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], 'q:', longopts=['dry-run'])
    dopts = dict(opts)
    thislevel = level(os.getcwd())
    # get all highpass datafiles
    levelidx = levels.index(thislevel)
    if levelidx == len(levels)-1:
        bb = "."
        ch = 1
    else:
        # construct a pattern for finding all highpass files below this level
        bb = os.sep.join([levels[i]+"*" for i in range(levelidx+1,len(levels))])
        ch = None
    queue = dopts.get("-q","flexi")
    bb = os.sep.join([bb] + ["*highpass.mat"])
    files = glob.glob(bb)
    homedir = os.path.expanduser('~')
    # save current directory path since we will be changing directories later
    currentdir = os.getcwd();
    execdir,py = os.path.split(sys.executable)
    master_file = open("sorting.pbs","w")
    master_file.write("#!/bin/bash\n")
    for i,f in enumerate(files):
        fname_sort = "hmmsort_%.4d.pbs" %(i,)

        pp = f.split(os.sep)
        dd = os.sep.join([currentdir] +  pp[:-1])
        if os.path.isfile(os.sep.join([dd, "hmmsort.mat"])):
            continue  # skip this channel if it already contains sorted data
        if os.path.isfile(os.sep.join([dd, "sorting_inprogress"])):
            continue
        if os.path.isfile(os.sep.join([dd, "sorting_done"])):
            continue

        master_file.write("cd {}\n".format(dd))
        master_file.write("qsub {}\n".format(fname_sort))
        master_file.write("cd {}\n".format(currentdir))
        fn = pp[-1]
        # change directories so that the output and error files will be created
        # in the respective channel directores and will be easier to check on
        # the status of the sorting
        os.chdir(dd)
        with open(fname_sort,"w") as fo:
            fo.write("#!/bin/bash\n")
            if queue == "flexi":
                # commenting out next line as it does not seem necessary
                # and because I would like to keep the jobid for hmm_learn on the
                # 3rd line since some scripts are expecting that
                fo.write("#PBS -l select=8:ncpus=1:mem=3GB\n")
                fo.write("#PBS -l walltime=128:00:00\n")
            elif queue == "serial":
                fo.write("#PBS -q serial\n")
                fo.write("#PBS -l mem=30GB\n")
                fo.write("#PBS -l select=1:ncpus=1:mem=30GB")
                fo.write("#PBS -l walltime=128:00:00\n")
            elif queue == "parallel12":
                fo.write("#PBS -q parallel12\n")
                fo.write("#PBS -l mem=30GB\n")
                fo.write("#PBS -l walltime=48:00:00\n")
            else:
                fo.write("#PBS -q short\n")
                fo.write("#PBS -l select=1:ncpus=1:mem=30GB\n")
                fo.write("#PBS -l walltime=24:00:00\n")

            fo.write("#PBS -o {}\n".format(os.path.join(dd, "hmmsort.o")))
            fo.write("#PBS -e {}\n".format(os.path.join(dd, "hmmsort.e")))

            fo.write("np=$( cat  ${PBS_NODEFILE} |wc -l );\n")
            fo.write("source /etc/profile.d/modules.sh\n")
            fo.write("module load miniconda\n")
            fo.write("conda activate hmmsort\n")
            fo.write("bash\n")
            fo.write(". ~/.bashrc\n")
            fo.write("cd %s\n" %(dd,))
            fo.write("touch sorting_inprogress\n")
            fo.write("%s/hmm_learn.py --sourceFile %s --iterations 3 " %(execdir,fn))
            fo.write("--chunkSize 100000 --outFile hmmsort/spike_templates.hdf5 ")
            fo.write("--min_snr 4.0 ")
            # get current username instead of hardcoding username
            fo.write("--max_size 1000000 --tempPath /hpctmp2/%s/tmp/\n" %(getpass.getuser()))
            fo.write("touch templates_found\n")
            fo.write("%s/run_hmm_decode.sh /app1/common/matlab/R2016a/ SourceFile %s Group 1 " %(execdir,fn))
            fo.write("fileName hmmsort/spike_templates.hdf5 save hdf5 ")
            fo.write("SaveFile hmmsort.mat hdf5path after_noise\n")
            fo.write("rm sorting_inprogress\n")
            fo.write("touch sorting_done\n")

    master_file.close()
    sys.exit(0)
