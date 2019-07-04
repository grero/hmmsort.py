import numpy as np

def writeWaveformsFile(spikeshapes,spikes,fname,conversion=10000):
    """writes the spike shapes as well as the time stamps in a format that the npt toolbox can understand
       input:   spikeshapes:    num_spikes X num_channels X num_samples
                spikes:         timepoints in units of 0.1 ms
                conversion      conversion factor that converts the spike times
                to second, e.g. 0.1 ms *10000 = 1s
   """

    fid = open(fname,'w')
    try:
        #write header
        hs = np.array([100],dtype=np.int32)
        hs.tofile(fid)
        num_spikes = np.array([spikeshapes.shape[0]],dtype=np.uint32)
        num_spikes.tofile(fid)
        if spikeshapes.ndim==3:
            num_channels = np.array([spikeshapes.shape[1]],dtype=np.uint8)
        else:
            num_channels = np.asarray([1],dtype=np.uint8)
        num_channels.tofile(fid)
        timepts = np.array([spikeshapes.shape[-1]],dtype=np.uint32)
        np.array([conversion,],dtype=np.uint32).tofile(fid)
        timepts.tofile(fid)

        fid.seek(hs)
        #now write spike shapes
        #since tofile always writes  C-contiguous arrays, make sure we are not doing something wrong
        if spikeshapes.flags['C_CONTIGUOUS']:
            spikeshapes.astype(np.int16).tofile(fid)
        else:
            spikeshapes.copy().astype(np.int16).tofile(fid)

        #timestamps
        #convert from seconds to microseconds
        C = 1.0e6/conversion
        spikes = (spikes*C).astype(np.uint64)
        spikes.tofile(fid)

    finally:
        fid.close()
    

def writeSyncsFile(fname,syncs,headerSize=300):
    """
    """
    fid = open(fname,'w')
    try:
        np.array([headerSize],dtype=np.int32).tofile(fid)
        #TODO: replace with name
        np.zeros((260,),dtype=np.uint8).tofile(fid)
        np.array([len(syncs)],dtype=np.int32).tofile(fid)
        fid.seek(headerSize)
        syncs.astype(np.int32).tofile(fid)
    finally:
        fid.close()
