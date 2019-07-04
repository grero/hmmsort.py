import numpy as np
import os

def readWaveformsFile(fname,readWaveforms=True,readTimestamps=True,
                  channels=None,timeFudge=0.001):
    """
    Read waveforms and timestamps from a waveforms file. the timeFudge factor
    takes the timestamps to milisecond precision
    """
    if not os.path.isfile(fname):
        IOError('File not found')
    
    fid = open(fname,'r')
    timepts = 32
    D = {}
    header = {}
    try:
        #read header
        hs = np.fromfile(fid,count=1,dtype=np.uint32).astype(np.uint64)
        ns = np.fromfile(fid,count=1,dtype=np.uint32).astype(np.uint64)
        chs = np.fromfile(fid,count=1,dtype=np.uint8).astype(np.uint64)
        conv = np.fromfile(fid,count=1,dtype=np.uint32).astype(np.uint64)
        timepts = np.fromfile(fid,count=1,dtype=np.uint32).astype(np.uint64)
        header['numSpikes'] = ns
        header['numChannels'] = chs
        header['timepts'] = timepts
        fid.seek(hs,0)
        if readWaveforms:
            if channels is None:
                    d = np.fromfile(fid,count=ns*chs*timepts,dtype=np.int16)
                    D['waveforms'] = d.reshape(ns,chs,timepts)
                    del d
            else:
                    #for
                    pass
        if readTimestamps: 
            fid.seek(hs+ns*chs*timepts*2,0)
            D['timestamps'] = np.fromfile(fid,count=ns,dtype=np.uint64)
            #precision of 1ms 
            D['timestamps']=D['timestamps']*timeFudge
    finally:
        #make sure we close the file if anthing happens
        fid.close()
    D['header']  = header
    return D


def readSyncsFile(fname):

    if not os.path.isfile(fname):
        IOError('File not found')

    fid = open(fname,'r')
    try:
        headersize = np.fromfile(fid,dtype=np.int32,count=1)
        dName = np.fromfile(fid,dtype=np.uint8,count=260)
        records = np.fromfile(fid,dtype=np.int32,count=1)
        meanF = np.fromfile(fid,dtype=np.float64,count=1)
        stdF = np.fromfile(fid,dtype=np.float64,count=1)
        #rewind file
        fid.seek(0)

        fid.seek(headersize)
        syncs = np.fromfile(fid,np.int32)
    finally:
        fid.close()

    return {'headersize':headersize,'records': records, 'meanF':meanF,'stdF':stdF,'syncs':syncs}

def readDescriptor(fname):

    lines = open(fname,'r').read().strip('\n').split('\n')
    #skip first line
    lines.pop(0)
    channels = int(lines.pop(0).split(' ')[-1])
    sampling_rate = float(lines.pop(0).split(' ')[-1])
    lines.pop(0)
    gain = float(lines.pop(0).split(' ')[-1])
    lines.pop(0) 
    ch_nr,gr_nr,status,typ = [],[],[],[]
    offset = 0
    missing_channels = []
    for l in lines:
        parts = l.split()
        if parts[-1].lower() == 'missing':
            missing_channels.append(int(parts[0]))
            offset+=1
            continue
        ch_nr.append(int(parts[0])-offset)
        gr_nr.append(int(parts[2]))
        typ.append(parts[1])
            
        status.append(parts[-1]=='Active')
    lines.close()
    return {'num_channels': channels, 'sampling_rate':sampling_rate,'gain':
            gain,'ch_nr':np.array(ch_nr),'gr_nr':np.array(gr_nr),'channel_status':np.array(status),'channel_type':typ,
            'missing_channels': missing_channels}

def readTriggers(fname):
    """reads the trigger signal from the given file name"""

    if not os.path.isfile(fname):
        IOError('File not found')

    #first, get the descriptor to find out wich channel contains the trigger
    path,fname = os.path.split(fname) 
    session_name,ext = os.path.splitext(fname)
    descr = readDescriptor('_'.join((session_name,'descriptor.txt')))

    #get the channel corresponding to presenter
    pidx =  descr['channel_type'].index('presenter')
    pch = descr['ch_nr'][pidx]
    nchs = descr['num_channels']

    #use mmap to map the file
    #first get the header size
    hs = np.fromfile(fname,dtype=np.int32,count=1)
    
    data = np.memmap(fname,offset=hs.item(),dtype=np.int16,mode='r')
    #resahep
    data = data.reshape(data.size/nchs,nchs)

    psignal = data[:,pidx]

    return psignal

def readNptData(fname):
    """
    Read data stored in the npt data format
    """
    #define data type mapping
    dataTypes = {1: np.uint8,
                 2: np.uint8,
                 3: np.int8,
                 4: np.int16,
                 5: np.int32,
                 6: np.int64,
                 7: np.uint8,
                 8: np.uint16,
                 9: np.uint32,
                 10: np.uint64,
                 11: np.float16,
                 12: np.float32,
                 13: np.double,
                 14: np.float64,
                }

    fid = open(fname,'r')
    #get header information
    header_size = np.fromfile(fid,count=1,dtype=np.int32)
    num_channels = np.fromfile(fid,count=1,dtype=np.uint8).astype(np.uint64)
    sampling_rate = np.fromfile(fid,count=1,dtype=np.uint32)
    datatype = np.fromfile(fid,count=1,dtype=np.int8).item()
    #get size of file
    fid.seek(0,2)
    fsize = fid.tell()
    npoints = (fsize-header_size)/np.array([1],
                                          dtype=dataTypes[datatype]).nbytes
    npoints/=num_channels
    fid.seek(header_size)
    data = np.fromfile(fid,count=num_channels*npoints,
                       dtype=dataTypes[datatype])

    data = data.reshape(npoints,num_channels).T
    fid.close()
    return data

