import numpy as np
import numba
import sys
import tempfile
import time
import os

print('Processing...')

# tempPath = os.path.join('C:','Users','a0126587','Desktop','tempfile')
ps = os.sep

# tempPath = 'C:'+ps+'Users'+ps+'a0126587'+ps+'Desktop'+ps+'tempfile'
tempPath = ps+'home'+ps+'svu'+ps+'a0126587'+ps+'hpctmp2'+ps+'tmp'

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

    iteration = 1

    scwd = shortenCWD()
    size = 100000
    print_strs = 'Hello World'
    n = 1000000
    # try_array = np.array([[1,2],[3,4]])
    print_strs = ''.join(char*n for char in print_strs)

    for bw in range(iteration):
        # sys.stdout.flush()
        # tempPath = '~/hpctmp2'
        # fid = tempfile.NamedTemporaryFile(dir=tempPath, delete=False, prefix=scwd)

        # create the temp file
        kk = 0
        while kk < 100:
            try:
                fid = tempfile.NamedTemporaryFile(dir=tempPath, delete=False, prefix=scwd)
                print('file' + str(bw) + str(kk) + 'is written')
		break
            except IOError:
                kk += 1
                time.sleep(np.random.random() * 30)
                print("Could not create tempfile due to IOError. Retrying ... ")
                sys.stdout.flush()
            else:
                print("Some error occurred while creating temp file...")
                break
        if kk == 100:
            if __name__ == '__main__':
                print("Could not create temporary file after 100 tries.")
                sys.stdout.flush()
                sys.exit(11)
            else:
                raise IOError('Could not create temporary file')



        # write something in the file
        kk = 0
        while kk < 100:
            try:
                for i in range(1,300):
                    #fid.write(bytes(print_strs, 'utf-8'))
                    #fid.write(str(i))
		    fid.write(print_strs)
                    time.sleep(1)
                print('something has been written in file' + str(bw) + str(i))
		break
            except ValueError:
                kk += 1
                time.sleep(np.random.random() * 30)
                print("Could not write in tempfile due to ValueError. Retrying ... ")
                sys.stdout.flush()
            else:
                print("Some error occurred while writing temp file...")
                break

        if kk == 100:
            if __name__ == '__main__':
                print("Could not write in temporary file after 100 tries.")
                sys.stdout.flush()
                sys.exit(22)
            else:
                raise IOError('Could not write in temporary file')




