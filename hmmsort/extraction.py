"""@package PyNpt.extraction
This module containst various functions related to the extraction of neural spikes from
a broadband signal
"""
import os
import numpy as np
import scipy.signal as signal
import sys
import time
import glob
import scipy
import scipy.linalg
#imports from npt
from . import fileReaders as fr
from . import utility
from . import fileWriters as fw
from . import utility as util


def readDataFile(fname,headerOnly=False,chunk=None):
    """
    Read data from a streamer file. If given, chunk should specify a range in
    data coordinates
    """
    fsize = os.stat(fname).st_size
    fid = open(fname,'r')
    headersize = np.fromfile(fid,count=1,dtype=np.uint32).astype(np.uint64)
    samplingrate = np.fromfile(fid,count=1,dtype=np.uint32).astype(np.uint64)
    if headersize == 74:
        nchs = np.fromfile(fid,count=1,dtype=np.uint16).astype(np.uint64)
    else:
        nchs = np.fromfile(fid,count=1,dtype=np.uint8).astype(np.uint64)
        
    #get file size
    if samplingrate > 1e5:
        fid.seek(4,0)
        if headersize == 74:
            nchs = np.fromfile(fid,count=1,dtype=np.uint16).astype(np.uint64)
        else:
            nchs = np.fromfile(fid,count=1,dtype=np.uint8).astype(np.uint64)
        samplingrate = util.first(np.fromfile(fid,count=1,dtype=np.uint32).astype(np.uint64))
    fid.close()
    nchs = int(util.first(nchs))
    hs = int(util.first(headersize))
    ndatapts = (fsize-hs)//nchs//2
    #create a memory map 
    if headerOnly: 
        return {'channels':nchs, 'samplingRate': samplingrate,
                'datapts': ndatapts,
                'headerSize':headersize}
    if chunk is None:
        #create a memory map of the whole file
        print("headersize: {}".format(util.first(headersize)))
        print("typeof(headersize): {}".format(type(util.first(headersize))))
        print("fsize: {}".format(fsize))
        hs = int(util.first(headersize))
        data = np.memmap(fname, offset=hs, dtype=np.int16,
                         mode='r', shape=(int((fsize-hs)/2),))
    else:
        #chunk specifies a range of data points to read
        #make sure we are within the limits of the file
        if chunk[0] < 0 or chunk[1] > ndatapts:
            raise ValueError('invalid chunk')
        #open file
        fid = open(fname,'r')
        #seek to the correct position
        fid.seek(headersize+nchs*chunk[0]*2)
        data = np.fromfile(fid,dtype=np.int16,count=(chunk[1]-chunk[0])*nchs)

        fid.close()

    print("nchs: {}".format(nchs))
    return data.reshape(data.size//nchs, nchs).T, samplingrate

def writeDataFile(fname,data,samplingRate=30000,headerSize=90):
    nchs,npts = data.shape
    fid = open(fname,'w')
    
    try:
       np.array([headerSize],dtype=np.uint32).tofile(fid)
       np.array([samplingRate],dtype=np.uint32).tofile(fid)
       np.array([nchs],dtype=np.uint16).tofile(fid)
       fid.seek(headerSize,0)
       data.T.astype(np.int16).tofile(fid)
    finally:
        fid.close()

def highpass(data,low=500.0,high=10000,samplingRate=30000,reorder=None,**kwargs):
    """Uses a 4th order butterworth filter to highpass filter the data from
    low to high"""
    
    b,a = signal.butter(4,1.0*low/high,btype='high')
    if isinstance(reorder,str):
        reorder = np.loadtxt(reorder,dtype=np.int)-1
    if data.ndim==2:
        #filter along the second axis
        if not 'idx' in kwargs:
            idx = np.arange(data.shape[0])
        else:
            idx = kwargs['idx']
        if reorder == None:
            fdata = np.zeros((len(idx),data.shape[1]))
            for i in xrange(fdata.shape[0]):
                fdata[i,:] = signal.filtfilt(b,a,data[idx[i]])
        else:
            fdata = np.zeros((reorder.max()+1,data.shape[1]))
            for i in xrange(data.shape[0]):
                fdata[reorder[i],:] = signal.filtfilt(b,a,data[idx[i]])

 
    else:
        fdata = signal.filtfilt(b,a,data)


    return fdata

def filter(data,low=500.0,high=10000,type='high',samplingRate=30000,downSampledRate=None,
           reorder=None,returnSamplingRate=False,**kwargs):
    """Uses a 4th order butterworth filter to highpass filter the data from
    low to high"""
    if type == 'low':
        #if we are lowpass-filtering, we need to downsample first
        if downSampledRate is None:
            downSampledRate = 1000
        _downsample = np.int(np.round(samplingRate/downSampledRate))
        use_data = signal.decimate(data,q=_downsample,n=4,axis=-1)
        #if we are lowpass filtering, we want to normalize to the nyquist
        #frequnecy
        high = downSampledRate/2.0
        samplingRate = downSampledRate
    else:
        use_data = data
    b,a = signal.butter(4,1.0*low/high,btype=type)
    if isinstance(reorder,str):
        reorder = np.loadtxt(reorder,dtype=np.int)-1
    if data.ndim==2:
        #filter along the second axis
        if not 'idx' in kwargs:
            idx = np.arange(use_data.shape[0])
        else:
            idx = kwargs['idx']
        if reorder == None:
            fdata = np.zeros((len(idx),use_data.shape[1]))
            for i in xrange(fdata.shape[0]):
                fdata[i,:] = signal.filtfilt(b,a,use_data[idx[i]])
        else:
            fdata = np.zeros((reorder.max()+1,use_data.shape[1]))
            for i in xrange(use_data.shape[0]):
                fdata[reorder[i],:] = signal.filtfilt(b,a,use_data[idx[i]])

 
    else:
        fdata = signal.filtfilt(b,a,use_data)

    if returnSamplingRate:
        return fdata,samplingRate

    return fdata

def filterFile(fname,chunkSize=None,type='high',samplingRate=None,**kwargs):
    """Reads the data and then calls highpass,writing the resulting data to a
    file whose name is the original filename with _highpass appended
    If chunkSize is specified, breaks the data into chunks before analyzing.
    chunkSize should be units of seconds
    """
    header = readDataFile(fname,headerOnly=True)
    if 'descriptor' in kwargs:
        descriptor = kwargs['descriptor']
        #only use channels which are active
    else:
        #try to locate
        descriptorFile = fname.replace('.bin','_descriptor.txt')
        #descriptorFile = glob.glob('*_descriptor.txt')
        #if len(descriptorFile)==0:
        if os.path.exists(descriptorFile):
            descriptor = fr.readDescriptor(descriptorFile)
        else:
            descriptor = {'channel_status':
                          np.ones((header['channels'],),dtype=np.bool)}
            print("No descriptor found. Using default...")
    #samplingRate = descriptor
    
    if type == 'high':
        print("Highpass filtering file %s..." % (fname,))
    elif type =='low':
        print("Lowpass filtering file %s..." % (fname,))
        
    idx = np.where(descriptor['channel_status'])[0]
    nchs = np.uint64(len(idx))
    sys.stdout.flush()
    #check if there is a highpass folder
    if not os.path.isdir('%spass' %(type,)):
        os.mkdir('%spass' %(type,))
    if chunkSize is not None:
        #get the file header
        if samplingRate is None:
            samplingRate =  header['samplingRate']
        #use the sampling rate to convert from seconds to datapts
        chunksize = chunkSize*samplingRate
        #find the number of chunks
        nchunks = max(1,header['datapts']/chunksize)
        #create chunks
        chunks = np.arange(nchunks,dtype=np.uint64)*chunksize
        #add the last chunk
        chunks = np.append(chunks,header['datapts'])
        #get chunkpairs
        chunkPairs = [(chunks[i],chunks[i+1]) for i in xrange(len(chunks)-1)]
        print("\tBreaking file into %d chunks..." %( nchunks,))
        sys.stdout.flush()
        #nchunks-=1
        dt = 0
        base,pt,ext = fname.partition('.')
        base,ext = os.path.splitext(fname)
        k = 0 #variable to keep track of counting if we do skip files
        for i in xrange(nchunks):
            outfname = '%spass/%s_%spass.%.4d' % (type,base,type,i+1)
            #check if the file already exists
            if os.path.exists(outfname):
                #check if size matches what we expect it to be
                ffsize = os.stat(outfname).st_size
                xpsize = ((chunkPairs[i][1]-chunkPairs[i][0])*nchs*2 +
                 90)
                if ffsize == xpsize:
                    print("\t\tA file for this chunk already exists. Skipping")
                    sys.stdout.flush()
                    continue
                    
            if type == 'high':
                print("\t\tHighpass filtering chunk %d..." % (i+1,))
            elif type =='low':
                print("\t\tLowpass filtering chunk %d..." % (i+1,))
            sys.stdout.flush()
            t1 = time.time()
            data,sr = readDataFile(fname,chunk=chunkPairs[i])
            if type == 'low':
                #make sure we return the new sampling rate
                fdata,sr = filter(data,samplingRate=samplingRate,idx=idx,type=type,
                                  low=300, returnSamplingRate=True,
                                  **kwargs)
            else:
                fdata = filter(data,samplingRate=samplingRate,idx=idx,type=type,**kwargs)
            #if 'gain' in descriptor:
            #    fdata/=descriptor['gain']
            writeDataFile(outfname,fdata.astype(np.int16),sr)
            t2 = time.time()
            #upate time average
            dt = (dt*(k) + (t2-t1))/(k+1)
            #output an educated guess of long we have left
            print("\t\tETTG: %f seconds" %( dt*(nchunks-(i+1))))
            sys.stdout.flush()
            k+=1

    else:
        base,pt,ext = fname.partition('.')
        base,ext = os.path.splitext(fname)
        outfname = '%spass/%s_%spass.%s' % (type,base,type,ext)
        if not os.path.isfile(outfname):
            data,sr = readDataFile(fname)
            fdata = filter(data,idx=idx,type=type,**kwargs)
            if 'gain' in descriptor:
                fdata/=descriptor['gain']
            writeDataFile(outfname,fdata.astype(np.int16),sr)
        else:
            print("File %s already exists. Skipping.." % (outfname,))

def computeStd(data,stdFactor=6):
    """Computes the noise of the data using two steps;
    first the standard devation is computed; then, anything exceeding 
    stdFactor * std is removed and the std is recomputed
    """
    if data.ndim==2:
        std = data.std(1)
        std2 = np.zeros(std.shape)
        m = np.zeros(std.shape)
        for i in xrange(len(std)):
            std2[i] = data[i,np.abs(data[i,:])<stdFactor*std[i]].std()
            m[i] = data[i,np.abs(data[i,:])<stdFactor*std[i]].mean()
            
    else:
        std = data.std()
        std2 = data[np.abs(data)<stdFactor*std].std()
        m = data[np.abs(data)<stdFactor*std].mean()

    return std2*stdFactor,m

def extract(data,threshold=None,stdFactor=6,spikePointsPrior=10,spikePointsPost=22,
        highpass=False,removePositiveArtifacts=True,removalMethod=2,**kwargs):
    """extracts spikes from the signal; if threshold is not given, it will be
    calculated using the standard deviation of the data, multiplied by
    stdFactor
    spikePointsPrior and spikePointsPost specify how many points to extract
    around the spike
    By default, it is assumed that data is already highpass filtered. If not,
    specify highpass=True and supply arguments to the highpass function
    Returns spikeshapes,timestamps (in units of data points), thresholds, means
    removalMethod indicates how positive artifacts should be removed. 
        removalMethod == 1      :    At least half the channels need to positive
        artifacts
        removalMethod == 2      :    A single channels is enough
    """
    if highpass==True:
        data = highpass(data,**kwargs)
    if threshold == None:
        threshold,mean = computeStd(data,stdFactor)
    else:
        mean =None
    #find negative peaks by identifying where the derivative changes sign, from
    #negative to positive
    nPointsPerSpike = spikePointsPrior + spikePointsPost
    if data.ndim==2:
        #multiple channels
        #detect negative peaks by looking at where the derivate changes from
        #negative to positive
        ch,peaks = np.where(np.diff(np.sign(np.diff(data)))>0)
        idx = data[ch,peaks+1]<-threshold[ch]
        chidx,pidx = ch[idx],peaks[idx]+1
        #sort spikes according to channel
       # qidx = np.lexsort((chidx,pidx))
        #sort according to peak size such that we process the larger spikes
        #first
        #make sure we don't exceed the length of the data
        qidx = pidx<data.shape[1]-nPointsPerSpike
        pidx = pidx[qidx]
        chidx = chidx[qidx]
        sidx = np.argsort(data[chidx,pidx])
        #sidx = np.argsort(pidx)
        #grab spikes across all channels, largest spikes first
        #impose refractory period of +/- 3 points
        spikes = []
        timestamps = []
        ridx = np.arange(-spikePointsPrior,spikePointsPost,dtype=np.int)
        cidx = np.arange(data.shape[0])
        pidxs = pidx[sidx]
        #pidxs = list(pidxs)
        i = 0
        while i < len(pidxs):
        #for i in xrange(len(pidx)):
            #midx = np.argmax(data[cidx[:,None],])
            spikes.append(data[cidx[:,None],(pidxs[i]+ridx)[None,:]])
            timestamps.append(pidxs[i])
            #remove peaks that fall within 3 points of this peak
            pidxs = np.concatenate((pidx[:i+1],pidxs[i+1:][np.abs(pidxs[i+1:]-pidxs[i])>3]))
            #jidx = np.where(np.abs(pidxs[i]-pidxs[i+1:])<3)[0]
            #for j in jidx[::-1]:
            #    pidxs.remove(pidxs[i+1+j])
            i+=1
        spikes = np.array(spikes)
        timestamps = np.array(timestamps)
    else:
        peaks = np.where(np.diff(np.sign(np.diff(data)))>0)[0]

        #identify valid peaks;i.e. spikes that exceed the threshold

        pidx = peaks[data[peaks+1]<-threshold]+1
        timestamps = pidx
        #ignore spikes at the end
        pidx[-spikePointsPost:] = False
        #grab 32 points around the peak
        spikeIdx = pidx[:,None] + np.arange(-spikePointsPrior,spikePointsPost)
        spikes = data[spikeIdx]
    if spikes.size > 0:
        if removePositiveArtifacts:
            #sometimes there are weird,large positive going spikes; remove these
            if removalMethod == 1:
                nchs = data.shape[0]
                i = np.where(~((spikes.max(-1)>threshold[None,:]).sum(1)>nchs/2))[0]
            elif removalMethod == 2:
                if spikes.ndim==3:
                    i = np.where(~((spikes.max(-1)>2*threshold[None,:]).any(1)))[0]
                else:
                    i = np.where(~((spikes.max(-1)>2*threshold)))[0]

            #i,j,k = np.where(spikes<2*threshold.max())
            spikes = spikes[i]
            timestamps = timestamps[i]
        #check if we have any spikes left after removing positive artifacts
        if spikes.size > 0:
            idx = np.argsort(timestamps)
            spikes = spikes[idx]
            timestamps = timestamps[idx]

    return spikes,timestamps,threshold,mean


def extractSpikesForSeesion(path=None,sessionName=None,spikeshapes=None,timestamps=None,
                            thresholds=None,means=None,save=False,
                            descriptor=None,reorder=None,samplingRate=None,groups=None,**kwargs):

    """
    """
    if path is None:
        path = os.getcwd()

    if reorder is not None:
        if isinstance(reorder,str):
            reorder = np.loadtxt(reorder,dtype=np.int)
    if sessionName is None:
        #get session name
        sessionName = utility.getSessionName(path)
    print('Extracting spikes for session %s' % (sessionName,))
    sys.stdout.flush()
    #get he file header
    fname = '%s.bin' %(sessionName,)
    if not os.path.isfile(fname):
        fname = 'highpass/%s_highpass.bin' %(sessionName,)
    header = readDataFile(fname,headerOnly=True)
    if descriptor is None:
        #check for the presence of a descriptor file
        descriptor = '/'.join((path,'%s_descriptor.txt' %(sessionName,)))
        if not os.path.exists(descriptor):
            descriptor = '/'.join((path,'../%s_descriptor.txt' %(sessionName,)))
        if not os.path.exists(descriptor):
            #create a default descriptor
            descriptor = {'channel_status':
                          np.ones((header['channels'],),dtype=np.bool),
                          'gr_nr': np.arange(header['channels'],dtype=np.int)+1}
            Warning('No descriptor file found')
    elif descriptor == False:
        #proceed without descriptor
        pass
    if isinstance(descriptor,str):
        descriptor = fr.readDescriptor(descriptor)
    #get the groups
    groupNr = np.asarray(descriptor['gr_nr'])
    #we only want positive groups
    if groups is None:
        groups = np.unique(groupNr[groupNr>0])
    #get the channels status
    chStates = descriptor['channel_status']
    #only use groups for which the group status is active
    groupNr = groupNr[chStates]
    #dictionaries to hold the spike data
    if spikeshapes is None:
        spikeshapes = {}
    if timestamps is None:
        timestamps = {}
    if thresholds is None:
        thresholds = {}
    if means is None:
        means = {}
    #if any of the above structurs already contain data, we skip those
    groups = list(set(groups).difference(spikeshapes.keys()))
    #check for the presence of a highpass directory
    highpassDir = '/'.join((path,'highpass'))
    #define variables to hold time statistics
    dataReadTime = []
    extractionTime = [] 
    nspikes = {}
    if not os.path.exists(highpassDir):
        highpassDir = '/'.join((path,'../highpass'))
    if os.path.exists(highpassDir):
        highpassFiles = glob.glob('%s/%s_highpass.*' %(highpassDir,sessionName))
        #make sure the files are sorted
        if len(highpassFiles)>1:
            highpassFiles = sorted(highpassFiles, key = lambda a:
                                   int(a.split('.')[-1]))
        #check if we got something
        if highpassFiles:
            nFiles = len(highpassFiles)
            #first load each file and compute the extraction threshold
            thresholdsFile = '%s.thresholds' %(sessionName,)
            if os.path.isfile(thresholdsFile):
                print("\tPrevious thresholds file %s" %(thresholdsFile,))
                threshold = np.loadtxt(thresholdsFile).flatten()
            else:
                print("\tEstimating extraction threshold...")
                sys.stdout.flush()
                x = np.zeros((header['channels'],))
                x2 = np.zeros((header['channels'],))
                N = np.zeros((header['channels'],))
                stdFactor = kwargs.get('stdFactor',4)
                for f in highpassFiles:
                    data,sr = readDataFile(f)
                    #first compute standard devation
                    ss = data.std(1)
                    #exclude points that exceed stdFactor X std
                    qidx = np.abs(data)<stdFactor*ss[:,None]
                    x+= (data*qidx).astype(np.float).sum(1)
                    x2+= (((data*qidx).astype(np.float))**2).sum(1)
                    N += qidx.astype(np.int).sum(1)
                x = x/N
                x2 = x2/N
                threshold = stdFactor*np.sqrt(x2-x**2)
                print("\tSaving thresholds to file %s" % (thresholdsFile,))
                np.savetxt(thresholdsFile,threshold)
            print(threshold)
            print("\tExtracting using a mean threshold of %f..." % (threshold.mean(),))
            sys.stdout.flush()
            #loop through each file, extracting spikes as we go
            #TODO: this really should be parallelized 
            #do the first file first
            hf = highpassFiles[0]
            #get the data as well as the sampling rate
            print("\tLoading data from file %s" %(hf,))
            sys.stdout.flush()      
            #for timing of reads
            t1 = time.time()
            data,sr = readDataFile(hf) 
            t2 = time.time()
            dataReadTime.append(t2-t1)
            print("\tThat took %f seconds" % (t2-t1,))
            sys.stdout.flush()      
            print('\t\tExtracting spikes')
            sys.stdout.flush()      
            t1 = time.time()
            for g in groups:
                chs = np.where(groupNr==g)[0]
                print('\t\t\tExtracting spikes for group %d channels %s...' %(g,
                                                                              ' '.join((map(str,chs)))))
                sys.stdout.flush()
                spikeshapes[g],timestamps[g],thresholds[g],means[g] = extract(data[chs,:],
                                                                              threshold=threshold[chs],**kwargs)
                print("\t\t\t\tExtracted %d spikes exceeding %f" %(
                    spikeshapes[g].shape[0],thresholds[g]))
                if reorder is not None:
                    #we need to reorder the channels such that within the
                    #virtual tetrode
                    if reorder.max() > maxCh:
                        chs = reorder[chs]
                    else:
                        chs = descriptor['ch_nr'][chs]
                        chs = np.where((chs==reorder[:,None]).any(1))[0]
                    chs = np.argsort(chs)
                    spikeshapes[g] = spikeshapes[g][:,chs,:]
                    thresholds[g] = thresholds[g][chs]
                    means[g] = means[g][chs]
                    #extract the spikes for this group and file
            #set an offset such that the timepoints are offset correctly
            t2 = time.time()
            extractionTime.append(t2-t1)
            print("\t\tThat took %f seconds" % (t2-t1,))
            sys.stdout.flush()      
            offset = data.shape[1]
            for i in xrange(1,nFiles):
                hf = highpassFiles[i]

                #get the data as well as the sampling rate
                print("\tLoading data from file %s" %(hf,))
                sys.stdout.flush()
                del data
                t1 = time.time()
                data,sr = readDataFile(hf) 
                t2 = time.time()
                dataReadTime.append(t2-t1)
                print("\tThat took %f seconds" % (t2-t1,))
                sys.stdout.flush()
                print('\t\tExtracting spikes')
                sys.stdout.flush()
                t1 = time.time()
                for g in groups:
                    chs = np.where(groupNr==g)[0]
                    print('\t\t\tExtracting spikes for group %d channels %s...' %(g,
                                                                                  ' '.join((map(str,chs)))))
                    sys.stdout.flush()
                    _spikeshapes,_timepoints,_thresholds,_means = extract(data[chs,:],
                                                                          threshold=threshold[chs],**kwargs)
                    if reorder is not None:
                        if reorder.max() > maxCh:
                            chs = reorder[np.where(descriptor['gr_nr']==g)[0]]
                        else:
                            chs = descriptor['ch_nr'][descriptor['gr_nr']==g]
                            chs = np.where((chs==reorder[:,None]).any(1))[0]
                        chs = np.argsort(chs)
                        _spikeshapes = _spikeshapes[:,chs,:]
                        _means = _means[chs]
                        _thresholds = _thresholds[chs]
                    #extract the spikes for this group and file
                    #only proceed if we actually found some spikes
                    if _spikeshapes.size > 0:
                        #offset the timepooints
                        _timepoints += offset
                        #update the offset for the next iterations
                        #append data for this file
                        if spikeshapes[g].size >0:
                            spikeshapes[g] = np.append(spikeshapes[g],
                                      _spikeshapes,axis=0)
                            timestamps[g] = np.append(timestamps[g],_timepoints)
                        else:
                            spikeshapes[g] = _spikeshapes
                            timestamps[g] = _timepoints
                        print("\t\t\t\tExtracted %d spikes exceeding %f" %(
                            _spikeshapes.shape[0],_thresholds))
                        sys.stdout.flush()
                        thresholds[g] = np.vstack((thresholds[g],_thresholds))
                        means[g] = np.vstack((means[g],_means))
                #update the offset for the new iteration    
                t2 = time.time()
                extractionTime.append(t2-t1)
                print("\t\tThat took %f seconds" % (t2-t1,))
                print("\t ETA: %f seconds" %((np.mean(extractionTime) +
                                   np.mean(dataReadTime))*(nFiles-(i-2)),))
                sys.stdout.flush()
                offset += data.shape[1] 
    else:
        print('No highpass files found')
    print("Average read time per file: %f" %(np.mean(dataReadTime),))
    print("Average extraction time per file: %f" % (np.mean(extractionTime),))
    if samplingRate is None:
        samplingRate = sr
    if save:
        #save the spikes and timestamps for each of the groups
        print("Saving waveforms files")
        sys.stdout.flush()
        saveWaveformsFiles(spikeshapes,timestamps,sessionName,
                           samplingRate=samplingRate)

    return spikeshapes,timestamps,thresholds,means

def saveWaveformsFiles(spikeshapes,timestamps,sessionName,basedir=None,
                       samplingRate=29990, thresholds=None,means=None):
    """
    Save the information contained in the spikeshapes and timestamps dictionary
    """
    groups = spikeshapes.keys()
    if basedir is None:
        basedir = os.getcwd()
    for g in groups:
        if spikeshapes[g].size>0:
            waveformsFile = '%sg%.4dwaveforms.bin' %(sessionName,g)
            try:
                fw.writeWaveformsFile(spikeshapes[g],timestamps[g],'%s/%sg%.4dwaveforms.bin'
                                         %(basedir,sessionName,g),conversion=samplingRate)
            except IOError:
                fw.writeWaveformsFile(spikeshapes[g],timestamps[g],'%s/%sg%.4dwaveforms.bin'
                                         %('/tmp',sessionName,g),conversion=samplingRate)
                print("Could not save waveforms file %s to directory %s. Saved to /tmp/ instead" %(waveformsFile,basedir))
            if thresholds is not None and means is not None:
                pass


def resolveOverlaps(waveforms,cids,threshold=0.05):
    """
    Resolve overlaps in the waveforms data. cids should contain the cluster indices of the known clusters, with a value for 0 for those waveforms which are not resolved
    """

    #compute means
    uc = np.unique(cids)
    uc = uc[uc>0]
    ncells = len(uc)
    #create pair index
    p1,p2 = np.where(uc[:,None]<uc[None,:])
    means = np.zeros((len(uc),) + waveforms.shape[1:])
    for i in xrange(len(uc)):
        means[i,:,:] = waveforms[cids==uc[i]].mean(0)

    #use a toeptlitz matrix to create overlaps
    sidx = scipy.linalg.toeplitz(np.arange(waveforms.shape[-1]))
    #we only want shifts with +/- 3 pts
    sidx = np.append(sidx[:3,],sidx[-3:,:],axis=0)
    #compute all possible shifts of the mean waveforms
    c = np.arange(len(uc))
    #all unique pairwise combinations
    ci = c[:,None]<c[None,:]
    Q = np.zeros((ncells*(ncells-1)/2,4,6,6,32))
    k = 0;
    for i in xrange(ncells-1):
        for j in xrange(i+1,ncells):
            Q[k,:,:,:,:] = (means[i,:,sidx][:,None,:,:] +
                            means[j,:,sidx][None,:,:,:]).transpose((3,0,1,2))
    

    ZQ = (Q**2).sum(-1).sum(1)
    #compute the mean square error for all waveforms
    #array to hold the scores
    Z = np.zeros((waveforms.shape[0]))
    oidx = np.zeros((waveforms.shape[0],len(uc)),dtype=np.bool)
    pidx = np.arange(6)
    #only select the waveforms not currently assigned
    widx = np.where(cids==0)[0]
    for i in xrange(len(widx)):
        s = (((waveforms[widx[i],:,:][:,None,None,:]-Q[:,:,:,:])**2).sum(-1).sum(1)/ZQ).min(-1).min(-1)
        #find the best match; best pair across all shifts
        #midx = s.argmin(1)
        #ridx = s[pidx[:,None],midx[:,None]].argmin()
        midx = s.argmin()
        q = s[midx]
        #if we accept the match, set the corresponding entries in oidx to true
        Z[widx[i]] = q
        if q < threshold:
            oidx[widx[i],p1[midx]] = True
            oidx[widx[i],p2[midx]] = True

    return oidx,Z

def removeNoise(data,samplingRate=30000,windowSize=None):
    nchs,pts = data.shape
    if windowSize is None:
        windowSize = samplingRate
    nchunks = int(np.ceil(data.shape[1]/windowSize))
    chunks = np.arange(0,data.shape[1],data.shape[1]/nchunks)
    if chunks[-1] != data.shape[1]:
        chunks = np.append(chunks,[data.shape[1]])
    chunks = zip(chunks[:-1],chunks[1:])
    D = np.zeros(data.shape,dtype=data.dtype)
    for i in xrange(nchunks):
        for c in xrange(nchs):
            D[c,chunks[i][0]:chunks[i][1]] = ecorrige50(data[c,chunks[i][0]:chunks[i][1]],
                                                        samplingRate=samplingRate)
    
    return D

def ecorrige50(signal,lineFreq=50,samplingRate=30000):
    """
    Remove the line noise and harmonics of the given lineFreq
    """
    ptsPer50 = np.fix(samplingRate/lineFreq)
    lll = max(signal.shape)
    #replicate the signal 4 times (?)
    indices = np.arange(lll,dtype=np.int)[None,:].repeat(4,0).flatten()
    #for each copy of the signal, shift the indices such that we end up sliding
    matind = np.arange(lll)[None,:] + (np.arange(-50/2+1,50/2+1)*ptsPer50)[:,None] + lll
    matind = matind.astype(np.int)
    matind = indices[matind]
    meanSig = signal[matind].mean(0)
    S = signal - meanSig
    
    return S

