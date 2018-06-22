import numpy as np
import numba
import sys
import tempfile
import time
import os

# tempPath = os.path.join('C:','Users','a0126587','Desktop','tempfile')
ps = os.sep

tempPath = 'C:'+ps+'Users'+ps+'a0126587'+ps+'Desktop'+ps+'tempfile'

# f = tempfile.NamedTemporaryFile(dir=tempPath,delete=False)
# f.write(b'Hello world!')
# print(f.name)
# f.close() # file is not immediately deleted because we
#           # used delete=False

def shortenCWD():
    """
    Creates a shortened name for the current working directory
    """
    cwd = os.getcwd()
    # split into directory strings
    cwdstrs = cwd.split(os.sep)
    # get channel
    chanstr = cwdstrs[-1]
    arraystr = cwdstrs[-2]
    sesstr = cwdstrs[-3]
    daystr = cwdstrs[-4]

    return daystr + sesstr[-2:] + arraystr[-2:] + chanstr[-3:]

if __name__ == '__main__':

    iteration = 10

    scwd = shortenCWD()
    size = 100000
    print_strs = 'Hello World'
    n = 10000
    # try_array = np.array([[1,2],[3,4]])
    print_strs = ''.join(char*n for char in print_strs)

    for bw in range(iteration):
        # sys.stdout.flush()
        # tempPath = '~/hpctmp2'
        # fid = tempfile.NamedTemporaryFile(dir=tempPath, delete=False, prefix=scwd)
        fid = tempfile.NamedTemporaryFile(dir=tempPath, delete=False, prefix=scwd)

        kk = 0
        while kk < 100:
            try:
                for i in range(1,10):
                    fid.write(bytes(print_strs, 'utf-8'))
                    time.sleep(1)
                    print('file' + str(bw) + str(i) + 'is written...')
            except ValueError:
                kk += 1
                time.sleep(10)
            else:
                break

        if kk == 100:
            print("Couldn't save the file...")



