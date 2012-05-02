#!/bin/env python2.6

"""
Functions to post-process the output from the hmm_sorting matlab routines
"""

#TODO:Add raster plot as the third plot in the last row; this can only be done
#if we have repetitions, of course

import numpy as np
import h5py
import os
import sys
import pylab as plt
import glob
import h5py
from PyNpt import fileWriters as fw
import fileReaders as fr
from mpl_toolkits.axisartist import Subplot
import scipy.cluster.hierarchy as hcluster
import scipy.weave as weave
import scipy.spatial as spatial
import scipy.io as mio
from PyNpt import extraction
import grating_tuning as gt

def formatAxis(ax):
    try:
        ax.axis['bottom'].major_ticks.set_tick_out(True)
        ax.axis['bottom'].minor_ticks.set_tick_out(True)
        ax.axis['left'].major_ticks.set_tick_out(True)
        ax.axis['left'].minor_ticks.set_tick_out(True)
    except:
        pass
    ax.axis['right'].set_visible(False)
    ax.axis['top'].set_visible(False)

def writeWwaveformsFile(data,fname='testwaveforms',samplingRate=29990):

    #TODO:Make sure that we sort the spike times and the units equally
    keys = np.sort(data['unitTimePoints'].keys())
    timestamps = ((np.concatenate([data['unitTimePoints'][c] for c in keys],axis=0))/(samplingRate/1000.0)).astype(np.uint64)
    idx = np.argsort(timestamps)
    timestamps = timestamps[idx]
    allspikes = np.concatenate([data['unitSpikes'][c] for c in keys],axis=0)
    allspikes = allspikes[idx,:,:]
    fw.writeWaveformsFile(allspikes.transpose((0,2,1)),timestamps,'%s.bin' % (fname,))
    #get the 
    cids = np.concatenate([np.array([[k]*len(v),v]) for k,v in data['spikeIdx'].items()],axis=1).astype(np.int)
    cids.astype(np.uint64).tofile('%s.overlap' % (fname,))
    ucids = cids.copy()
    C = np.bincount(cids[1,:])
    ucids[0,C[C>1]] = -1
    ucids[0,:].tofile('%s.cut' % (fname,),sep='\n')


def sortBasedOnSession(psession):
    """Sorts the spikes in this session based on already sorted spikes from a
    previous session. This entails running the hmm_decode algorithm using
    templaes from already sorted units"""
    #get the name of the current session by looking for descriptor files
    dFile = glob.glob('*_descriptor.txt')
    if len(dFile)==0:
        print "Could not determine current session"
        return None

    thisSessionName = dFile[0].split('/')[-1].replace('_descriptor.txt','')
    descriptorFiles = glob.glob('%s/*_descriptor.txt' % psession)
    if len(descriptorFiles)==0:
        print "No descriptor found"
        return None
    descriptor = fr.readDescriptor(dFile[0])
    nchs = descriptor['channel_status'].sum()
    #get the session name from the descriptor
    sessionName = descriptorFiles[0].split('/')[-1].replace('_descriptor.txt','')
    groups = np.unique(descriptor['gr_nr'])
    #we only want positive groups
    groups = groups[groups>0]
    #check for the presence of a sort directory
    if not os.path.isdir('%s/sort' % psession):
        print "No sort results found"
        return None
    dataFile = h5py.File('%s.hdf5' %( thisSessionName,),'a')
    dataFile.create_group('templates')
    #compute covariance for the current session
    if os.path.isdir('%s/highpass' % psession):
        highpassFiles = glob.glob('highpass/%s_highpass.*'
                                  %(thisSessionName,))
    else:
        highpassFiles = glob.glob('%s_highpass.*' %(thisSessionName,))

    C = None
    if len(highpassFiles)==0:
        print "Could not find any highpass files. "
        print "Highpass filtering data..."
        extraction.highpassFile('%s.bin' %( thisSessionName,))
        highpassFiles = glob.glob('%s_highpass.*' %(thisSessionName,))
    if len(highpassFiles)==0:
        print "Could not find any highpass files. "
        dataFile.close()
        return None
    else:
        N = 0
        C = np.zeros((nchs,nchs))
        for f in highpassFiles:
            data,sr = extraction.readDataFile(f)
            n = data.shape[1]
            c = np.cov(data)
            C+=n*c
            N+=n
        C/=N
    for g in groups:
        wfFile = '%s/sort/%sg%.4dwaveforms.bin' %( psession,sessionName,g)
        if not os.path.isfile(wfFile):
            print "No waveformsfile found for group %d.." %(g,)
            continue
        #get the cutfile
        cutFile = wfFile.replace('.bin','.cut')
        if not os.path.isfile(cutFile):
            print "No cut file found for group %d..." %(g,)
            continue
        channels = np.where(descriptor['gr_nr'][descriptor['channel_status']]==g)[0]
        wfData = fr.readWaveformsFile(wfFile)['waveforms']
        cids = np.loadtxt(cutFile,comments='%',dtype=np.int)
        ucids = np.unique(cids)
        ucids = ucids[ucids>0]
        #now create the templates by taking the mean of each cluster
        templates = np.array([wfData[cids==c].mean(0) for c in ucids])
        dataFile['templates'].create_group('group%d' %g )
        dataFile['templates']['group%d' %g]['spikeForms'] = templates
        if C!=None:
            dataFile['templates']['group%d' %g]['cinv'] = np.linalg.pinv(C[channels[:,None],channels[None,:]])
        dataFile.flush()

    dataFile.close()


def processGroups(dataFilePattern=None):
    
    files = glob.glob('*_descriptor.txt')
    if len(files)==0:
        print "No descriptor file found. Skipping..."
        return
    descriptorFile = files[0]
    sessionName,_,_ = descriptorFile.partition('_') 
    #get groups
    descriptor = fr.readDescriptor(descriptorFile)
    groups = np.unique(descriptor['gr_nr'])
    groups = groups[groups>0]
    data = {}
    for g in groups:
        data[g] = processFiles('%sg%.4d.*.mat' %(sessionName,g),dataFilePattern=dataFilePattern)

    return data


def processFiles(pattern,outFile=None,dataFilePattern=None): 
    """
    Pattern could be any file pattern that glob understands.
    """
    files = glob.glob(pattern)
    #get numeric part of file names
    P = pattern.replace('*','([0-9]*)')
    F = ' '.join((files))
    nparts = glob.re.findall(P,F)
    ndigits = len(nparts[0])
    nparts = np.array(map(int,nparts))
    #sort
    fidx = np.argsort(nparts)
    #detect any gaps
    gaps = np.diff(nparts[fidx])
    gapidx = np.where(gaps>1)[0]
    gaps = gaps[gapidx]
    if len(gaps) > 0:
        mf = np.concatenate([nparts[fidx][gapidx[i]] + np.arange(1,gaps[i]) for i in xrange(len(gaps))])
        ps = pattern.replace('*','%.4d')
        print "Gaps detected. The missing files are most likely:"
        for q in mf:
            print ps % q
        query = raw_input('Accept [y/n]')
        if query == 'n':
            return {}

    data = {'dataSize':0}
    spIdxOffset = 0
    for fi in xrange(len(files)):
        f = files[fidx[fi]]
        print "Processing file %s" % (f,)
        sys.stdout.flush()
        if dataFilePattern == None:
            datafile = f.replace('.mat','') 
        else:
            datafile = '%s.%.4d' % (dataFilePattern,nparts[fidx[fi]])
        qdata = processData(f,datafile)
        for k in qdata.keys():
            if isinstance(qdata[k],dict):
                if not k in data:
                   data[k]  = {}
                for kk,vv in qdata[k].items():
                    if not kk in data[k]:
                        #data[ks][kks] = np.empty((0,)+vv.shape[1:])
                        data[k][kk] = vv
                    else:
                        if k == 'unitTimePoints':
                            data[k][kk] = np.concatenate((data[k][kk],vv+data['dataSize']),axis=0)
                        elif k == 'spikeIdx':
                            data[k][kk] = np.concatenate((data[k][kk],vv+spIdxOffset),axis=0)
                        elif k == 'uniqueIdx' or k == 'nonOverlapIdx':
                            data[k][kk] = np.concatenate((data[k][kk],vv+data[k][kk].shape[0]),axis=0)
                        else:
                            data[k][kk] = np.concatenate((data[k][kk],vv),axis=0)
            else:
                if k == 'dataSize':
                    if fidx[fi] in gapidx:
                        #we have a gap; i.e. the next file is missing. In that
                        #case, we have to offset by more
                        data[k]+=gaps[gapidx==fidx[fi]]*qdata[k]
                    else:
                        data[k]+=qdata[k]
                else:
                    if not k in data:
                        data[k] = np.empty((0,)+qdata[k].shape[1:])
                    data[k] = np.concatenate((data[k],qdata[k]),axis=0)
        spIdxOffset += qdata['allSpikes'].shape[0]

    return data

def processFilesHDF5(pattern,outFile=None,dataFilePattern=None): 
    """
    Pattern could be any file pattern that glob understands.
    """
    files = glob.glob(pattern)
    fidx = np.arange(len(files))
    nparts = 1+np.arange(len(files)) 
    gapidx = []
    if pattern.find('*')>-1:
    #get numeric part of file names
        P = pattern.replace('*','([0-9]*)')
        F = ' '.join((files))
        nparts = glob.re.findall(P,F)
        ndigits = len(nparts[0])
        nparts = np.array(map(int,nparts))
        #sort
        fidx = np.argsort(nparts)
        #detect any gaps
        gaps = np.diff(nparts[fidx])
        gapidx = np.where(gaps>1)[0]
        gaps = gaps[gapidx]
        if len(gaps) > 0:
            mf = np.concatenate([nparts[fidx][gapidx[i]] + np.arange(1,gaps[i]) for i in xrange(len(gaps))])
            ps = pattern.replace('*','%.4d')
            print "Gaps detected. The missing files are most likely:"
            for q in mf:
                print ps % q
            query = raw_input('Accept [y/n]')
            if query == 'n':
                return {}

    data = h5py.File(outFile,'a')
    data.create_dataset('dataSize',data=np.zeros((1,),dtype=np.int))
    try:
        spIdxOffset = 0
        for fi in xrange(len(files)):
            f = files[fidx[fi]]
            print "Processing file %s" % (f,)
            sys.stdout.flush()
            if dataFilePattern == None:
                datafile = f.replace('.mat','') 
            else:
                datafile = '%s.%.4d' % (dataFilePattern,nparts[fidx[fi]])
            qdata = processData(f,datafile)
            for k in qdata.keys():
                ks = str(k)
                if isinstance(qdata[k],dict):
                    if not ks in data:
                        #create an empty group
                       #data[k]  = {}
                       g = data.create_group(ks)
                    for kk,vv in qdata[k].items():
                        kks = str(kk)
                        if not kks in data[ks]:
                            #data[ks][kks] = np.empty((0,)+vv.shape[1:])
                            chunks = tuple([min(10,max(1,s)) for s in vv.shape])
                            if vv.size>0:
                                data[ks].create_dataset(kks,data=vv,chunks = chunks,compression=2,fletcher32=True,shuffle=True,maxshape=(None,)+vv.shape[1:]) 
                            else:
                                data[ks].create_dataset(kks,shape=vv.shape,chunks = chunks,compression=2,fletcher32=True,shuffle=True,maxshape=(None,))

                            #data[k][kk] = vv
                        else:
                            S = data[ks][kks].shape
                            Sv = vv.shape
                            if Sv[0]==0:
                                continue
                            #resize the dataset; only along the first dimension
                            newsize = (S[0] + Sv[0],) + S[1:]
                            data[ks][kks].resize(newsize)
                            if k == 'unitTimePoints':
                                data[ks][kks][S[0]:S[0]+Sv[0]] = vv+data['dataSize']
                            elif k == 'spikeIdx':
                                data[ks][kks][S[0]:S[0]+Sv[0]] = vv+spIdxOffset
                            elif k == 'uniqueIdx' or k == 'nonOverlapIdx':
                                data[ks][kks][S[0]:S[0]+Sv[0]] = vv+S[0]
                            else:
                                data[ks][kks][S[0]:S[0]+Sv[0]] = vv 
                else:
                    if k == 'spikeForms':
                        if not ks in data:
                            data.create_dataset('spikeForms',data=qdata[k])
                    elif k == 'channels':
                        if not ks in data:
                            data.create_dataset('channels',data=qdata[k])
                    elif k == 'dataSize':
                        if fidx[fi] in gapidx:
                            #we have a gap; i.e. the next file is missing. In that
                            #case, we have to offset by more
                            data[ks][0]+=data[ks][0]+gaps[gapidx==fidx[fi]]*qdata[k]
                        else:
                            data[ks][0]+=qdata[k]
                    else:
                        if not ks in data:
                            chunks = tuple([max(min(10,s),1) for s in qdata[k].shape])
                            if qdata[k].shape[0]>0:
                                data.create_dataset(ks,data=qdata[k],chunks=chunks,compression=2,fletcher32=True,shuffle=True,maxshape=(None,)+qdata[k].shape[1:])
                            else:
                                data.create_dataset(ks,shape=qdata[k].shape,chunks=chunks,compression=2,fletcher32=True,shuffle=True,maxshape=(None,) + qdata[k].shape[1:])
                        else:
                            S = data[ks].shape
                            Sv = qdata[k].shape                            
                            if Sv[0] > 0:
                                newsize = (S[0]+Sv[0],) + S[1:]
                                data[ks].resize(newsize)
                                data[ks][S[0]:S[0]+Sv[0]] =qdata[k]
            data.flush()
            spIdxOffset += qdata['allSpikes'].shape[0]
    finally:
        data.close()
    data = h5py.File(outFile,'a')

    return data

"""
def processFilesHDF5(pattern,outfile=None): 
    files = glob.glob(pattern)
    if outfile == None:
        base,pt,ext = pattern.partition('.')
        base.replace('*','')
        outfile  = '.'.join((base,'hdf5'))
        outfile = os.path.expanduser('~/Documents/research/data/%s' % outfile)
    data = h5py.File(outfile,'w')
    try:
        data['dataSize'] = np.array([0]) 
        spIdxOffset = 0
        for f in files:
            datafile = f.replace('.mat','') 
            qdata = processData(f,datafile)
            for k in qdata.keys():
                ks = str(k)
                if isinstance(qdata[k],dict):
                    if not ks in data:
                        g = data.create_group(ks)
                    for kk,vv in qdata[k].items():
                        kks = str(kk)
                        if not kks in data[ks]:
                            #data[ks][kks] = np.empty((0,)+vv.shape[1:])
                            data[ks][kks] = vv
                        else:
                            if k == 'unitTimePoints':
                                data[ks][kks] = np.concatenate((data[ks][kks],vv+data['dataSize']),axis=0)
                            elif k == 'spikeIdx':
                                data[ks][kks] = np.concatenate((data[ks][kks],vv+spIdxOffset),axis=0)
                            else:
                                data[ks][kks] = np.concatenate((data[ks][kks],vv),axis=0)
                else:
                    if k == 'dataSize':
                        data[ks]+=qdata[k]
                    else:
                        if not ks in data:
                            data[ks] = np.empty((0,)+qdata[k].shape[1:])
                        data[ks] = np.concatenate((data[ks],qdata[k]),axis=0)
            spIdxOffset += qdata['allSpikes'].shape[0]
            data.flush()
    finally:
        data.close()

    data = h5py.File(outfile,'r')
    return data
"""

def processData(fname,dataFile=None):

    if not os.path.isfile(fname):
        print "Sorry, the file %s could not be found" % fname
        return
    mustClose = False
    try:
        if h5py.is_hdf5(fname):
            sortData = h5py.File(fname,'r')
            mustClose = True
        else:
            sortData = {}
            d = mio.loadmat(fname,mdict=sortData)

        
        #load the state sequence for all neurons
        seq = sortData['mlseq'][:].astype(np.int)
        if seq.shape[0] > seq.shape[1]:
            seq = seq.T
        #load the spike forms for each neuron
        spikeForms = sortData['spikeForms'][:]
        if spikeForms.shape[0] != seq.shape[0]:
            spikeForms = spikeForms.transpose((2,1,0))
        
        #create a signal with overlaps on each channel
        #S = spikeForms[np.arange(spikeForms.shape[0])[:,None,None],np.arange(spikeForms.shape[1])[None,:,None],seq[:,None,:]].sum(0)
        #find where at least one neuron is active
        idx = np.where((seq>0).any(0))[0]

        noverlapPts = spikeForms.shape[0]*spikeForms.shape[-1]
        #find the total number of states (including overlaps) involved in each
        #spike
        #get the start and the end points of each compound spike
        #make sure to exclude spikes that start towards the end of the
        #file;these will not have an ending point
        #spikeStart = spikeStart[spikeStart<seq.shape[1]-spikeForms.shape[-1]]
        #spikeEnd = np.where((seq==spikeForms.shape[-1]-1).any(0)*((seq==spikeForms.shape[-1]-1)+(seq==0)).all(0))[0]
        #make sure we are only using spikes that end
        #spikeStart = spikeStart[spikeStart < spikeEnd[-1]]
        #pidx = np.array([spikeStart,spikeEnd]).T.flatten()
        #create a spike matrix
        #spMatrix = np.zeros((len(cidx),spikeForms.shape[1],noverlapPts)).transpose((1,0,2))
        #spIdxMatrix = np.zeros((len(cidx),noverlapPts),dtype=np.int)

        #create an index that will place each overlap spike in the correct position
        #in the spike matrix
        #k,l = np.where(np.arange(noverlapPts)[None,:]<cidx[:,None])

        #k and l now gives the spike index and the timepoint index into the spMatrix
        #spMatrix[:,k,l] = S[:,idx]
        #TODO: we should extact the spikes here, i.e. use the +/- 3pt refractory
        #period around each spikes
        #spIdxMatrix[k,l] = idx.astype(np.int)
        #spMatrix = spMatrix.transpose((1,0,2))
        #find the minimum point on each channel and shift each spike such that the
        #minimum point occurs at 1/3
        #spidx = spMatrix.min(1).argmin(1)
        #tpts = int(1/3.0*spikeForms.shape[-1])
        #W = np.array([np.roll(spMatrix[u],tpts-spidx[u],axis=-1) for u in xrange(len(spidx))])
       
        #create a dictionary of spike indices assigned to each unit
        minpts = spikeForms.min(1).argmin(1)
        #get the index of each minimum point for each neuron
        i,j = np.where(seq==minpts[:,None])
        i = i[j<seq.shape[1]-22]
        j = j[j<seq.shape[1]-22]
        #find the true overlaps, i.e. spikes that differ by less than 3 points
        """
        d = j[:,None]-j
        k,l = ((d<=3)*(d>0))
        """
        #tids = hcluster.fclusterdata(j[:,None],3,criterion='distance',metric='cityblock')-1
        #get an index into the original timestamp array for the unique points
        #u,q = np.unique(tids)
        #find wich compound spike each single unit spike contributes to
        #cspike,spidx = np.where(((j[:,None] >= spikeStart)*(j[:,None] <= spikeEnd)))
        #cspikes,spidx,uspidx = np.unique(spidx,return_index=True,return_inverse=True)
        #qidx = np.digitize(j,pidx)
        units = dict([(u,j[i==u]) for u in np.unique(i)]) 
        #spikeIdx = dict([(u,spidx[uspidx[i==u]]) for u in np.unique(i)]) 
        spikeIdx = dict([(u,np.where(i==u)[0]) for u in np.unique(i)]) 
        spikes = {}
        channels = None
        data = None
        dataSize = 0
        if dataFile == None:
            if 'data' in sortData:
                data = sortData['data'][:]
            elif 'dataFile' in sortData:
                dataFile = sortData['dataFile']
        if data == None:
            if dataFile != None:
                data,sr = extraction.readDataFile(dataFile)
                #reordering
                if os.path.isfile('reorder.txt'):
                    reorder = np.loadtxt('reorder.txt',dtype=np.int)-1
                    if reorder.max()>=data.shape[0]:
                        rdata=np.zeros((reorder.max()+1,data.shape[1]))
                        rdata[reorder,:] = data
                        data = rdata
                        del rdata
                data = data.T
                dataSize = data.shape[0]
                if 'Channels' in sortData:
                    #subtract 1 because we are dealing with matlab base-1 indexing
                    channels = sortData['Channels'][:].flatten().astype(np.int)-1
                    data = data[:,channels]
                else:
                    channels = np.arange(data.shape[1])
            else:
                print "Sorry, no data found. Exiting..."
                if mustClose:
                    sortData.close()
                return {'unitTimePoints': units,'spikeIdx':spikeIdx}
        
        keys = np.array(units.keys())
        uniqueIdx = {}
        nonoverlapIdx = {}
        for c in keys: 
            idx = units[c]
            idx = idx[idx<data.shape[0]-22][:,None]+ np.arange(-10,22)[None,:]
            spikes[c] = data[idx,:]
            otherkeys = keys[keys!=c] 
            if len(otherkeys>0):
                uniqueIdx[c] = np.where(np.array([pdist_threshold(units[c],units[c1],3) for c1 in otherkeys]).prod(0))[0]
                nonoverlapIdx[c] = np.where(np.array([pdist_threshold(units[c],units[c1],32) for c1 in otherkeys]).prod(0))[0]
            else:
                uniqueIdx[c] = np.arange(len(units[c]))
                nonoverlapIdx[c] = np.arange(len(units[c]))

        #get the unique spikes
        if len(units.keys())>0:
            allSpikeIdx = np.concatenate(units.values(),axis=0)
            allSpikeIdx = (allSpikeIdx[allSpikeIdx<data.shape[0]-22][:,None]+np.arange(-10,22))[:,None,:].repeat(data.shape[1],1)
            allSpikes = data[allSpikeIdx,np.arange(data.shape[1])[None,:,None]].transpose((0,2,1))
        else:
            allSpikes = np.empty((0,spikeForms.shape[1],32)).transpose((0,2,1))
        del data

    finally:
        if mustClose:
            sortData.close()
    
    
    return {'unitTimePoints': units,'unitSpikes':spikes,'allSpikes':allSpikes,'spikeIdx':spikeIdx,'dataSize':dataSize,'spikeForms':spikeForms,'channels':channels,'uniqueIdx':uniqueIdx,'nonOverlapIdx': nonoverlapIdx}

def pdist_threshold(a1,a2,thresh):
    n1 = len(a1)
    n2 = len(a2)
    idx = np.ones((n1,),dtype=np.uint8)
    code = """
    unsigned int i,j;
    double d;
    for(i=0;i<n1;i++)
    {
        for(j=0;j<n2;j++)
        {
            d = a1[i]-a2[j];
            d = sqrt(d*d);
            if( d < thresh )
            {
                idx[i] = 0;
                break;
            }
        }
    }
    """
    err = weave.inline(code,['a1','a2','n1','n2','thresh','idx'])

    return idx.astype(np.bool)

def pdist_threshold2(a1,a2,thresh):
    n1 = len(a1)
    n2 = len(a2)
    idx = np.ones((n1,),dtype=np.uint8)
    code1 = """
    unsigned int i,j,c;
    double d;
    c = 0;
    //first pass; count the number of items
    for(i=0;i<n1;i++)
    {
        for(j=0;j<n2;j++)
        {
            d = a1[i]-a2[j];
            d = sqrt(d*d);
            if( d < thresh )
            {
                c+=1;
            }
        }
    }
    return_val = c;
    """
    c = weave.inline(code1,['a1','a2','n1','n2','thresh'])
    print c 
    dist = np.zeros((c,))
    code = """
    unsigned int i,j,k;
    double d;
    k = 0;
    for(i=0;i<n1;i++)
    {
        for(j=0;j<n2;j++)
        {
            d = a1[i]-a2[j];
            if( sqrt(d*d) < thresh )
            {
                dist[k] = d;
                k++;
            }

        }
    }
    """
    err = weave.inline(code,['a1','a2','n1','n2','thresh','dist'])

    return dist 

def plotSpikes(qdata,save=False,fname='hmmSorting.pdf',tuning=False,figsize=(10,6)):

    allSpikes = qdata['allSpikes'] 
    unitSpikes = qdata['unitSpikes']
    spikeIdx = qdata['spikeIdx']
    spikeIdx = qdata['unitTimePoints']
    units = qdata['unitTimePoints']
    spikeForms = qdata['spikeForms']
    channels = qdata['channels']
    uniqueIdx = qdata['uniqueIdx']
    samplingRate = qdata.get('samplingRate',30000.0)
    """
    mustClose = False
    if isinstance(dataFile,str):
        dataFile = h5py.File(dataFile,'r')
        mustClose = True
    data = dataFile['data'][:]
    """
    keys = np.array(units.keys())
    x = np.arange(32)[None,:] + 42*np.arange(spikeForms.shape[1])[:,None]
    xt = np.linspace(0,31,spikeForms.shape[-1])[None,:] + 42*np.arange(spikeForms.shape[1])[:,None]
    xch = 10 + 42*np.arange(len(channels))
    for c in units.keys():
        ymin,ymax = (5000,-5000)
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(hspace=0.3)
        print "Unit: %s " %(str(c),)
        print "\t Plotting waveforms..."
        sys.stdout.flush()
        #allspikes = data[units[c][:,None]+np.arange(-10,22)[None,:],:]
        #allspikes = allSpikes[spikeIdx[c]]
        allspikes = qdata['unitSpikes'][c]
        otherunits = keys[keys!=c]
        #nonOverlapIdx = np.prod(np.array([~np.lib.arraysetops.in1d(spikeIdx[c],spikeIdx[c1]) for c1 in otherunits]),axis=0).astype(np.bool)
        #nonOverlapIdx = np.prod(np.array([pdist_threshold(spikeIdx[c],spikeIdx[c1],3) for c1 in otherunits]),axis=0).astype(np.bool)
        #nonOverlapIdx = uniqueIdx[c]
        nonOverlapIdx = qdata['nonOverlapIdx'][c]
        overlapIdx = np.lib.arraysetops.setdiff1d(np.arange(qdata['unitTimePoints'][c].shape[0]),nonOverlapIdx)
        #allspikes = allSpikes[np.lib.arraysetops.union1d(nonOverlapIdx,overlapIdx)]
        ax = Subplot(fig,2,3,1)
        fig.add_axes(ax)
        formatAxis(ax)
        #plt.plot(x.T,sp,'b')
        m = allspikes[:].mean(0)
        s = allspikes[:].std(0)
        plt.plot(x.T,m,'k',lw=1.5)
        #find the minimum point for this template
        ich = spikeForms[int(c)].min(1).argmin()
        ix = spikeForms[int(c)][ich,:].argmin()
        #plt.plot(x.T,spikeForms[int(c)][:,ix-10:ix+22].T,'r')
        plt.plot(x.T,np.roll(spikeForms[int(c)],10-ix,axis=1)[:,:32].T,'r')
        for i in xrange(x.shape[0]):
            plt.fill_between(x[i],m[:,i]-s[:,i],m[:,i]+s[:,i],color='b',alpha=0.5)
        yl = ax.get_ylim()
        ymin = min(ymin,yl[0])
        ymax = max(ymax,yl[1])
        ax.set_title('All spikes (%d)' % (allspikes.shape[0],))

        ax = Subplot(fig,2,3,2)
        fig.add_axes(ax)
        formatAxis(ax)
        if len(nonOverlapIdx)>0:
            m =  allspikes[:][nonOverlapIdx,:,:].mean(0)
            s =  allspikes[:][nonOverlapIdx,:,:].std(0)
            plt.plot(x.T,m,'k',lw=1.5)
            for i in xrange(x.shape[0]):
                plt.fill_between(x[i],m[:,i]-s[:,i],m[:,i]+s[:,i],color='b',alpha=0.5)
        #plt.plot(x.T,spikeForms[int(c)][:,ix-10:ix+22].T,'r')
        plt.plot(x.T,np.roll(spikeForms[int(c)],10-ix,axis=1)[:,:32].T,'r')
        yl = ax.get_ylim()
        ymin = min(ymin,yl[0])
        ymax = max(ymax,yl[1])
        #for sp in allspikes[nonOverlapIdx,:,:]:
        #    plt.plot(x.T,sp,'r')

        ax.set_title('Non-overlap spikes (%d)' %(nonOverlapIdx.shape[0],))
        ax = Subplot(fig,2,3,3)
        fig.add_axes(ax)
        formatAxis(ax)
        if len(overlapIdx)>0:
            m =  allspikes[:][overlapIdx,:,:].mean(0)
            s =  allspikes[:][overlapIdx,:,:].std(0)
            plt.plot(x.T,m,'k',lw=1.5)
            for i in xrange(x.shape[0]):
                plt.fill_between(x[i],m[:,i]-s[:,i],m[:,i]+s[:,i],color='b',alpha=0.5)
        #plt.plot(x.T,spikeForms[int(c)][:,ix-10:ix+22].T,'r')
        plt.plot(x.T,np.roll(spikeForms[int(c)],10-ix,axis=1)[:,:32].T,'r')
        yl = ax.get_ylim()
        ymin = min(ymin,yl[0])
        ymax = max(ymax,yl[1])
        #for sp in allspikes[~nonOverlapIdx,:,:]:
        #    plt.plot(x.T,sp,'g')
        ax.set_title('Overlap spikes (%d)' % ((overlapIdx).shape[0],))
        for a in fig.axes:
            a.set_ylim((ymin,ymax))
            a.set_xticks(xch)
            a.set_xticklabels(map(str,channels))
            a.set_xlabel('Channels')
        for a in fig.axes[1:]:
            a.set_yticklabels([])
        fig.axes[0].set_ylabel('Amplitude')
        """
        isi distribution
        """
        print "\t ISI distribution..."
        sys.stdout.flush()
        timepoints = qdata['unitTimePoints'][c][:]/(samplingRate/1000)
        if len(timepoints)<2:
            print "Too few spikes. Aborting..."
            continue 
        isi = np.log(np.diff(timepoints))
        n,b = np.histogram(isi,100)
        ax = Subplot(fig,2,3,4)
        fig.add_axes(ax)
        formatAxis(ax)
        ax.plot(b[:-1],n,'k')
        yl = ax.get_ylim()
        ax.vlines(0.0,0,yl[1],'r',lw=1.5)
        ax.set_xlabel('ISI [ms]')
        #get xticklabels
        xl,xh = int(np.round((b[0]-0.5)*2))/2,int(np.round((b[-1]+0.5)*2))/2
        xl = -0.5
        dx = np.round(10.0*(xh-xl)/5.0)/10
        xt_ = np.arange(xl,xh+1,dx)
        ax.set_xticks(xt_)
        ax.set_xticklabels(map(lambda s: r'$10^{%.1f}$' % (s,),xt_))

        """
        auto-correlogram
        """
        print "\t auto-correllogram..."
        sys.stdout.flush()
        if not 'autoCorr' in qdata:
            if isinstance(qdata,dict):
                qdata['autoCorr'] = {}
            else:
                qdata.create_group('autoCorr')
        if not c in qdata['autoCorr']:
            C = pdist_threshold2(timepoints,timepoints,50)
            qdata['autoCorr'][c] = C
            if not isinstance(qdata,dict):
                qdata.flush()
        else:
            C = qdata['autoCorr'][c][:]
        n,b = np.histogram(C[C!=0],np.arange(-50,50))
        ax = Subplot(fig,2,3,5)
        fig.add_axes(ax)
        formatAxis(ax)
        ax.plot(b[:-1],n,'k')
        ax.fill_betweenx([0,n.max()],-1.0,1.0,color='r',alpha=0.3)
        ax.set_xlabel('Lag [ms]')
        if tuning:
            print "\tPlotting tuning..."
            sys.stdout.flush()
            #attempt to get tuning for the current session, based on PWD
            stimCounts,isiCounts,angles,spikedata = gt.getTuning(sptrain=timepoints)        
            #reshape to number of orientations X number of reps, collapsing
            #across everything else
            #angles = np.append(angles,[angles[0]])
            C = stimCounts['0'].transpose((1,0,2,3))
            C = C.reshape(C.shape[0],C.size/C.shape[0])
            ax = plt.subplot(2,3,6,polar=True) 
            ax.errorbar(angles*np.pi/180,C.mean(1),C.std(1))

        if save:
            if not os.path.isabs(fname):
                fn = os.path.expanduser('~/Documents/research/figures/SpikeSorting/hmm/%s' % (fname.replace('.pdf','Unit%s.pdf' %(str(c),)),))
            else:
                fn = fname.replace('.pdf','Unit%s.pdf' %(str(c),))

            fig.savefig(fn,bbox='tight')

    if not save:
        plt.draw()
    """
    if mustClose:
        dataFile.close()
    """


def plotXcorr(qdata,save=False,fname='hmmSortingUnits.pdf'):

    unitTimePoints = qdata['unitTimePoints']
    samplingRate = qdata.get('samplingRate',30000.0)
    fig = plt.figure(figsize=(10,10) )
    units = unitTimePoints.keys()
    nunits = len(units)
    i = 1
    if not 'XCorr' in qdata:
        if isinstance(qdata,dict):
            qdata['XCorr'] = {}
        else:
            qdata.create_group('XCorr')
    for k1 in xrange(len(units)-1) :
        if not units[k1] in qdata['XCorr']:
            qdata['XCorr'].create_group(units[k1])
        for k2 in xrange(k1+1,len(units)):
            if not units[k2] in qdata['XCorr'][units[k1]]:
                T1 = unitTimePoints[units[k1]][:]/(samplingRate/1000)
                T2 = unitTimePoints[units[k2]][:]/(samplingRate/1000)
                #compute differences less than 50 ms
                C = pdist_threshold2(T1,T2,50)
                qdata['XCorr'][units[k1]].create_dataset(units[k2],data=C,compression=2,fletcher32=True,shuffle=True)
            else:
                C = qdata['XCorr'][units[k1]][units[k2]][:]
            n,b = np.histogram(C,np.arange(-50,50),normed=True)
            ax = Subplot(fig,nunits-1,nunits,k1*nunits+k2) 
            fig.add_axes(ax)
            formatAxis(ax)
            ax.plot(b[:-1],n,'k')
            ax.fill_betweenx([0,n.max()],-1.0,1.0,color='r',alpha=0.3)
            if not (k1 == len(units)-2 and k2 == len(units)-1):
                ax.set_xticklabels('')
                ax.set_yticklabels('')
    if save:
        fig.savefig(os.path.expanduser('~/Documents/research/figures/SpikeSorting/hmm/%s' %( fname,)),bbox='tight') 
    else:
        plt.draw()

def verifySpikes(data,group=1,dataFilePattern=None):
    """
    Overlay the spikes found with the highpass data files
    """
    #TODO: only shows the first file for now
    dataFiles = glob.glob(dataFilePattern)
    descriptorFile,_,_ = dataFiles[0].partition('.')
    descriptorFile = '%s.txt' % (descriptorFile.replace('highpass','descriptor'),)
    #get the descriptor
    descriptor = fr.readDescriptor(descriptorFile)
    #get the relevant channels for this group
    nchannels = sum(descriptor['gr_nr']>0)
    channels = np.where(descriptor['gr_nr']==group)[0]
    hdata = np.memmap(dataFiles[0],mode='r',offset=73,dtype=np.int16) 
    hdata = hdata.reshape(hdata.size/nchannels,nchannels)[:,channels]
    #calculate offset 
    offset = np.append([0],np.cumsum(hdata.max(0))[:-1])
    x = np.arange(hdata.shape[0])
                       
    plt.plot(x[:,None].repeat(len(channels),1),hdata+offset[None,:].repeat(hdata.shape[0],0))
    yl = plt.ylim()
    timepoints = data['unitTimePoints']
    for k in timepoints.keys():
        T = timepoints[k]
        plt.vlines(T[(T>=x[0])*(T<x[-1])],yl[0],yl[1])

def isolationDistance(c,fdata,cids,ncids=None):
    
    #get the number of points in this cluster
    npoints = sum(cids==c)
    #get the cluster mean
    m = fdata[cids==c,:].mean(0)
    #get the largest distance from the mean for points in this cluster
    dm = np.sqrt(((fdata[cids==c,:] - m)**2).sum(1))
    #get points not in this cluster
    if ncids==None:
        ncids = cids!=c
    #get the distance from this cluster to mean to all the points
    D = np.sqrt(((fdata[ncids,:] - m[None,:])**2).sum(1))

    #isolation distance is the distance to the n'th point not in this cluster,
    #where n is the number of points in this cluster
    #normalize by the largest distance from the mean to a point int hs cluster
    Ds = np.sort(D)
    if len(Ds)>npoints:
        return Ds[npoints-1]/dm
    else:
        return Ds[-1]/dm

def plotISIDistributions(sessions,groups=None,sessionTypes=None,samplingRate=30000.0,save=False,fname=None,figsize=(10,6)):
    """Plots the isi distributions for all the cells in the given sessions"""
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.05,right=.95)
    ISIs = {}
    ncells = 0
    if sessionTypes != None:
        sessionTypes = dict(zip(sessions,sessionTypes))
    for g in groups:
        for s in sessions:
            dataFile = h5py.File(os.path.expanduser('~/Documents/research/data/spikesorting/hmm/p=1e-20/%sg%.4d.hdf5' % (s,g)),'r')
            try:
                for c in dataFile['unitTimePoints'].keys():
                    isi = np.log(np.diff(dataFile['unitTimePoints'][c][:]/(samplingRate/1000)))
                    cn = 'g%dc%d' % (g,int(c))
                    if cn in ISIs:
                        ISIs[cn]['%s' %(s,)] = isi
                    else:
                        ISIs[cn] = {'%s' %(s,): isi}
            finally:
                dataFile.close()
        
    i = 1
    ncells = len(ISIs.keys())
    for c in ISIs.keys():
        ax = Subplot(fig,1,ncells,i)
        formatAxis(ax)
        fig.add_axes(ax)
        ax.set_title(c)
        for k,v in ISIs[c].items():
            if sessionTypes != None:
                L = sessionTypes[k]
            else:
                L = k
            n,b = np.histogram(v,bins=20,normed=True)
            plt.plot(b[:-1],n,label=L)
        i+=1
        ax.set_xlabel('ISI [ms]')
        xl,xh = int(np.round((b[0]-0.5)*2))/2,int(np.round((b[-1]+0.5)*2))/2
        xl = -0.5
        dx = np.round(10.0*(xh-xl)/4.0)/10
        xt_ = np.arange(xl,xh+1,dx)
        ax.set_xticks(xt_)
        ax.set_xticklabels(map(lambda s: r'$10^{%.1f}$' % (s,),xt_))

    fig.axes[-1].legend()
    if save:
        if fname == None:
            fname = os.path.expanduser('~/Documents/research/figures/isi_comparison.pdf')
        fig.savefig(fname,bbox='tight')

def plotSpikeCountDistributions(sessions,groups=None,sessionTypes=None,samplingRate=30000.0,save=False,windowSize=40,fname=None,figsize=(10,6)):
    """Plots the isi distributions for all the cells in the given sessions"""
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.05,right=.95)
    spikeCounts = {}
    ncells = 0
    nz = {}
    if sessionTypes != None:
        sessionTypes = dict(zip(sessions,sessionTypes))
    for g in groups:
        for s in sessions:
            dataFile = h5py.File(os.path.expanduser('~/Documents/research/data/spikesorting/hmm/p=1e-20/%sg%.4d.hdf5' % (s,g)),'r')
            try:
                for c in dataFile['unitTimePoints'].keys():
                    sptrain = dataFile['unitTimePoints'][c][:]/(samplingRate/1000)
                    nbins = sptrain.max()/windowSize
                    bins,bs = np.linspace(0,sptrain.max(),nbins,retstep=True)
                    sc,bins = np.histogram(sptrain,bins)
                    cn = 'g%dc%d' % (g,int(c))
                    sl = s
                    if sessionTypes != None:
                        sl = sessionTypes[s]
                    if cn in spikeCounts:
                        spikeCounts[cn]['%s' %(s,)] = {'counts': sc, 'bins':bins}
                        nz[cn]['%s' %( sl,)] = 1.0*sum(sc==0)/len(sc)
                    else:
                        spikeCounts[cn] = {'%s' %(s,):  {'counts':sc,'bins':bins}}
                        nz[cn] = {'%s' %(sl,): 1.0*sum(sc==0)/len(sc)}
            finally:
                dataFile.close()
        
    i = 1
    ncells = len(spikeCounts.keys())
    nsessions = len(sessions)
    colors = ['b','r','g','y','c','m']
    for c in spikeCounts.keys():
        ax = Subplot(fig,1,ncells,i)
        formatAxis(ax)
        fig.add_axes(ax)
        ax.set_title(c)
        j = 0
        for k,v in spikeCounts[c].items():
            #n,b = np.histogram(v['counts'],bins=20,normed=True)
            #plt.plot(b[:-1],n,label=k)
            if sessionTypes != None:
                L = sessionTypes[k]
            else:
                L = k
            n = np.bincount(v['counts'])
            b = np.unique(v['counts'])
            b = np.arange(b[0],len(n))
            ax.bar(b+j*0.2,1.0*n/n.sum(),align='center',width=0.3,fc=colors[j],label=L)
            j+=1
        i+=1
        ax.set_xlabel('Spike counts')
    fig.axes[-1].legend()
    if save:
        if fname == None:
            fname = os.path.expanduser('~/Documents/research/figures/isi_comparison.pdf')
        fig.savefig(fname,bbox='tight')

    return nz

def findBest3DProjection(fdata,cids):
    
    npoints,ndims = fdata.shape
    clusters = np.unique(cids)
    clusters = clusters[clusters>=0].astype(np.int)
    #for all clusters, get the remaining points
    otherPoints = [cids!=c for c in clusters]
    maxIsoDist = []
    bestDims = []
    
    #loop through all combinations of 3
    for i in xrange(ndims-2):
        for j in xrange(i+1,ndims-1):
            for k in xrange(j+1,ndims):
                D = np.zeros((len(clusters),))
                for c in clusters:
                    D[c] = isolationDistance(c,fdata[:,[i,j,k]],cids,otherPoints[c])
                dm = D.max()
                bestDims.append((i,j,k))
                maxIsoDist.append(dm)
   
    bestDimIdx = np.argmax(maxIsoDist)
    return bestDims[bestDimIdx]


if __name__ == '__main__':

    data = processGroups()
    
    #plot all groups
    for g in data.keys():
        plotSpikes(qdata,save=True)
