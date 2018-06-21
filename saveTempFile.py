import numpy as np
import numba
import sys
import tempfile
import time
import os

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

    scwd = 'a'
    tempPath = os.path.join('C:','Users','a0126587','Desktop','tempfile')

    scwd = shortenCWD()

    for bw in range(iteration):
        sys.stdout.flush()
        # tempPath = '~/hpctmp2'
        fid = tempfile.NamedTemporaryFile(dir=tempPath,delete=False,prefix=scwd)

        kk = 0
        while kk < 100:
            try:
                fid.write('Hi')
                print('file' + str(i) + 'is written...')
            except ValueError:
                kk += 1
                time.sleep(10)
            else:
                break

        if kk == 100:
            print("Couldn't save the file...")



