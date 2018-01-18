import os
import glob

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
    thislevel = level(os.getcwd())
    # get all highpass datafiles
    levelidx = levels.index(thislevel)
    # construct a pattern for finding all highpass files below this level
    bb = os.sep.join([levels[i]+"*" for i in xrange(levelidx+1,len(levels))])
    bb = os.sep.join([bb] + ["highpass.mat"])
    files = glob.glob(bb)
    
    with open("hmmsort.dag","w") as fid:
        for f in files:
            pp = f.split(os.sep)  # get the channel
            dd = os.sep.join(pp[:-1])
            ch = int(filter(str.isdigit, pp[-2]))
            fid.write('JOB hmmlearn_%d hmmsort.cmd\n' % (ch, ))
            fid.write('VARS hmmlearn_%d fname=%s\n' %(ch, f))
            fid.write('JOB hmmdecode_%d hmmdecode.cmd\n' % (ch, ))
            fid.write('VARS hmmdecode_%d fname=%s\n' %(ch, f))
            fid.write('PARENT hmmlearn_%d CHILD hmmdecode_%d\n' % (ch, ch))
            fid.write('\n')

