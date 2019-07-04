#!/home/svu/a0126587/anaconda2/bin/python
import argparse
import os
import sys
import h5py
import glob

def get_chunks(fname, chunksize):
    ff = h5py.File(fname,"r")
    data = ff["highpassdata"]["data"]["data"]
    N = data.shape[0]
    chunks = range(0, N, chunksize)
    if chunks[-1] != N-1:
        chunks.append(N)
    ff.close()
    return chunks

def copy_file(infile,outfile):
    ff = h5py.File(infile, "r")
    gg = h5py.File(outfile, "w")
    gg.create_group("highpassdata")
    gg.create_group("highpassdata/data")
    for k in ff["highpassdata/data"].keys():
        if k != "data":
            if type(ff["highpassdata/data"][k]) == h5py._hl.group.Group:
                gg["highpassdata/data"].create_group(k)
                for kk in ff["highpassdata/data"][k].keys():
                    gg["highpassdata/data"][k][kk] = ff["highpassdata/data"][k][kk][:]
            else:
                gg["highpassdata/data"][k] = ff["highpassdata/data"][k][:]
            
    _data = gg["highpassdata/data/"].create_dataset("data", (1000000,), 
                                                    maxshape=(None,),
                                                    dtype='i2',
                                                    chunks=(50000,))
    n = ff["highpassdata/data/data/"].shape[0]
    _data.resize(n, axis=0)
    _data[:n] = ff["highpassdata/data/data"][:]
    gg.close()
    ff.close()

def assempble_chunks(bfname):
    files = glob.glob(bfname + "_*.mat")
    files.sort()
    outfile = bfname + "_2.mat"
    #copy data from the first file
    copy_file(files[0], outfile)
    gg = h5py.File(outfile,"r+")
    odata = gg["highpassdata/data/data"]
    for f in files[1:]:
        ff = h5py.File(f,"r")
        idata = ff["highpassdata/data/data"]
        m = odata.shape[0]
        n = idata.shape[0]
        odata.resize(m+n, axis=0)
        odata[m:m+n] = idata
        gg.flush()
        ff.close()
    gg.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index',type=int, 
                       required=False,
                       help="chunk index")
    parser.add_argument('-a', '--all', 
                        action="store_true",
                        help="Create all chunks")
    parser.add_argument('-s', '--size', type=int,
                       default=50000,
                       required=True,
                       help="chunk size")
    parser.add_argument('-f', '--file', type=str,
                       required=True,
                       help="Name of file to chunk")
    args = parser.parse_args()

    chunksize = args.size
    chunkindex = args.index
    fname = args.file

    chunks = get_chunks(fname, chunksize)
    sys.stdout.write("%d\n" % (len(chunks),))
    ff = h5py.File(fname,"r")
    data = ff["highpassdata"]["data"]["data"]
    bb,ext = os.path.splitext(fname) 
    if args.all:
        cidx = xrange(len(chunks)-1)
    else:
        cidx = [chunkidx]
    for ci in cidx:
        fdata = data[chunks[ci]:chunks[ci+1]]
        outfname = bb + "_%03d" % (ci,) + ext
        if not os.path.isfile(outfname):
            gg = h5py.File(outfname, "w")
            gg.create_group("highpassdata")
            gg.create_group("highpassdata/data")
            for k in ff["highpassdata/data"].keys():
                if k != "data":
                    if type(ff["highpassdata/data"][k]) == h5py._hl.group.Group:
                        gg["highpassdata/data"].create_group(k)
                        for kk in ff["highpassdata/data"][k].keys():
                            gg["highpassdata/data"][k][kk] = ff["highpassdata/data"][k][kk][:]
                    else:
                        gg["highpassdata/data"][k] = ff["highpassdata/data"][k][:]
                    
            gg["highpassdata/data/"]["data"] = fdata
            gg.close()

    ff.close()




