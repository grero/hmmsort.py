import argparse
import os
import h5py

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index',type=int, 
                       required=True,
                       help="chunk index")
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

    ff = h5py.File(fname,"r")
    data = ff["highpassdata"]["data"]["data"]
    N = data.shape[0]
    chunks = range(0, N, chunksize)
    if chunks[-1] != N-1:
        chunks.append(N)
    
    fdata = data[chunks[chunkindex]:chunks[chunkindex+1]]
    bb,ext = os.path.splitext(fname) 
    outfname = bb + "_%03d" % (chunkindex,) + ext
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






