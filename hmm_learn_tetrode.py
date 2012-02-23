#!/opt/cluster/usr/bin/python2.6
##!/usr/bin/env python2.6

import numpy as np
import scipy.weave as weave
import scipy.stats as stats
import sys
import tempfile
import os
import h5py
import glob
import traceback
import fileReaders as fr
import scipy.interpolate as interpolate
from PyNpt import extraction

#use PDF backend if we are running a script
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('PDF')
    import matplotlib.pylab as plt
else:
    import pylab as plt

np.seterr(all='warn')
#only raise an error if we are dividing by zero; this usually means we made a
#mistake somewhere
np.seterr(divide='raise')
if os.path.isdir('/Volumes/Chimera/tmp'):
    tempfile.tempdir = '/Volumes/Chimera/tmp'

def forward(g,P,spklength,N,winlength,p):

    code = """
    unsigned int t,j;
    double S,tiny,T;
    tiny = exp(-700.0);
    for(t=1;t<winlength;t++)
    {
       for(j=0;j<M;j++) 
       {
            g[j*winlength+t] = g[q[j]*winlength+t-1];
       }
       S = 0; 
       for(j=0;j<_np;j++)
       {
            S+=p[j];
       }
       T = 0;
       for(j=0;j<_np-1;j++)
       {
            T+=(g[(1+j*(spklength-1))*winlength+t]);
        }
       g[t] = T+g[t]-g[t-1]*S;
       for(j=0;j<_np-1;j++)
        {
            g[(1+j*(spklength-1))*winlength+t] = g[t-1]*p[j];
        }
       S = 0;
       for(j=0;j<M;j++)
       {
            g[j*winlength+t]=g[j*winlength+t]*P[j*winlength+t];
            S+=g[j*winlength+t];
       }
       for(j=0;j<M;j++)
       {
            g[j*winlength+t]/=(S+tiny);
       }

    }
    """
    winlength = g.shape[1]
    M = g.shape[0]
    _np = len(p)
    q = np.concatenate(([N*(spklength-1)-1],np.arange(N*(spklength-1))),axis=0)
    err = weave.inline(code,['p','_np','q','g','winlength','P','spklength','M'])
    return g 

def learnTemplatesFromFile(dataFile,group=1,save=True,outfile=None,chunksize=1.5e6,version=2,nFileChunks=None,fileChunkId=None,divideByGain=False,**kwargs):

    if not os.path.isfile(dataFile):
        print "File at path %s could not be found " % (dataFile,)
        return [],[]
    
    #channels is the 3rd argument
    fid = open(dataFile,'r')
    header_size = np.fromfile(fid,dtype=np.uint32,count=1)
    channels = np.fromfile(fid,dtype=np.uint8,count=1).astype(np.uint32)
    sampling_rate = np.fromfile(fid,dtype=np.uint32,count=1).astype(np.float)
    #check whether the sampling rate is out of whack; if so, we try changing
    #the order
    if sampling_rate > 1e5:
        #rewind to read the information again
        fid.seek(4,0)
        sampling_rate = np.fromfile(fid,dtype=np.uint32,count=1).astype(np.float)
        channels = np.fromfile(fid,dtype=np.uint8,count=1).astype(np.uint32)
    sampling_rate = min(30000.0,sampling_rate)
    
    fid.close()

    data = np.memmap(dataFile,mode='r',dtype=np.int16,offset=header_size)
    data = data.reshape(data.size/channels,channels)
    descriptorFile = '%s_descriptor.txt' % (dataFile[:dataFile.rfind('_')],)
    #get group information form the descriptor
    descriptor = fr.readDescriptor(descriptorFile)
    channels = np.where(descriptor['gr_nr']==group)[0]
    cdata = data[:,channels]
    if divideByGain and 'gain' in descriptor:
        cdata = cdata/np.float(descriptor['gain'])
    if nFileChunks!=None and fileChunkId!=None:
        #we should only process parts of the file
        fileChunkSize = np.ceil(1.0*cdata.shape[0]/nFileChunks)
    
        cdata = cdata[fileChunkId*fileChunkSize:(fileChunkId+1)*fileChunkSize,:]
    if save:
        if outfile == None:
            name,ext = os.path.splitext(dataFile)
            name.replace('_highpass','')
            if not os.path.isdir('hmmsort'):
                os.mkdir('hmmsort')
            if fileChunkId == None:
                outfile = 'hmmsort/%sg%.4d%s.hdf5' % (name,group,ext)
            else:
                outfile = 'hmmsort/%sg%.4d%s.%d.hdf5' % (name,group,ext,fileChunkId)
        try:
            outf = h5py.File(outfile,'a')
        except IOError:
            #file exists; what do we do?
            print 'An error occurred trying to open the file %s...' %(outfile,)
    if version == 1:
        #compute the covariance matrix of the full data
        cinv = np.linalg.pinv(np.cov(cdata.T))
        #divide file into two chunks
        nchunks = int(np.ceil(1.0*cdata.shape[0]/chunksize))
        spkforms = []
        p = []
        for i in xrange(nchunks):
            print "Processing chunk %d of %d..." % (i+1,nchunks)
            sys.stdout.flush()
            try:
                sp,pp,_ = learnTemplates(cdata[i*chunksize:(i+1)*chunksize,:],samplingRate = sampling_rate,**kwargs)

                spkforms.extend(sp) 
                p.extend(pp)
                if save and len(sp)>0:
                    try:
                        outf.create_group('chunk%d' %(i,))
                        outf['chunk%d' %(i,)]['spikeForms'] = sp
                        outf['chunk%d' %(i,)]['p'] = pp 
                    except:
                        pass
                    finally:
                        outf.flush()
            except:
                continue
        spkforms = np.array(spkforms)
        p = np.array(p)
        #spkforms2,p2,_ = learnTemplates(cdata[:data.shape[0]/2,:],samplingRate = sampling_rate,**kwargs)
       
        #combine spkforms from both chunks
        if spkforms.shape[0]>=2:
            spkforms,p = combineSpikes(spkforms,p,cinv,data.shape[0])
    else:
        if save:
            outf.close()
        else:
            outfile = False
        spikeForms,cinv = learnTemplates(cdata,samplingRate=sampling_rate,chunksize=chunksize,version=2,saveToFile=outfile,**kwargs)
    if spikeForms != None and spikeForms['second_learning']['after_sparse']['spikeForms'].shape[0]>=1:
        if save:
            #reopen to save the last result
            outf.close()
            outf = h5py.File(outfile,'a')
            try:
                outf['spikeForms'] = spikeForms['second_learning']['after_sparse']['spikeForms']
                outf['p'] = spikeForms['second_learning']['after_sparse']['p']
                outf['cinv'] = cinv
                outf.flush()
            except:
                pass
    else:
        print "No spikeforms found"

    #make sure we close the file
    if save:
        outf.close()


    return spikeForms,cinv

def learnTemplates(data,splitp=None,debug=True,save=False,samplingRate=None,version=2,saveToFile=False,redo=False,**kwargs):
    """ 
    Learns templates from the data using the Baum-Welch algorithm.
        Inputs:
            data    :   npoints X nchannels
            splitp  :   scalar  :   minimum firing rate accepted for a neuron in Hz
            samplingRate    :   scalar  :   sampling rate of the data in Hz
            debug   :   boolean :   if true, spikeforms are plotted as they are discovered and refined
            save    :   boolean :   if true, the debug plots are saved 
        Outputs:
                spkform : the learned templates
                p   :   the estimated probability of firing for each template
    """
    if debug:
        plt.figure(figsize=(6,10))
    if samplingRate == None:
        samplingRate = 30000.0
    if splitp == None:
        #set the minimum firing rate at 0.5 Hz
        splitp = 0.5/samplingRate
    else:
        splitp = splitp/samplingRate
    if save:
        #open a file to save the spkforms to
        pass
    #version 2 uses chunking more aggressively, and does not make heavy use of memory maps
    if version==2:
        learnf = learndbw1v2
    else:
        learnf = learndbw1
    if saveToFile:
        try:
            outFile = h5py.File(saveToFile,'a')
            if 'all' in outFile and not redo:
                print "Data already exists and redo was not requested. Skipping..."
                return None,None
        except IOError:
            print "Could not open file %s..." %(saveToFile,)
            saveToFile = False
    spikeForms = {}
    data,spkform,p,cinv = learnf(data,iterations=1,debug=debug,**kwargs)
    spikeForms['all'] = {'spikeForms': spkform,'p': p}
    if saveToFile:
        if not 'all' in outFile:
            outFile.create_group('all')
        outFile['all']['spikeForms'] = spkform
        outFile['all']['p'] = p
        outFile.flush()
    spkform,p,idx = removeStn(spkform,p,cinv,data,kwargs.get('small_thresh',1))
    spikeForms['after_noise'] = {'spikeForms': spkform,'p': p}
    if saveToFile and len(p)>0:
        if not 'after_noise' in outFile:
            outFile.create_group('after_noise')
        outFile['after_noise']['spikeForms'] = spkform
        outFile['after_noise']['p'] = p
        outFile.flush()
    if len(spkform)==0:
        print "No spikeforms remain after removing those compatible with noise"
        return spikeForms,cinv
    spkform,p = removeSparse(spkform,p,splitp)
    spikeForms['after_sparse'] = {'spikeForms': spkform,'p': p}
    if saveToFile and len(p)>0:
        if not 'after_sparse' in outFile:
            outFile.create_group('after_sparse')
        outFile['after_sparse']['spikeForms'] = spkform
        outFile['after_sparse']['p'] = p
        outFile.flush()
    if len(spkform)==0:
        print "No spikeforms remain after removing templates that fire too sparsely"
        return spikeForms,cinv
    if debug:
        plt.gca().clear()
        x = np.arange(spkform.shape[-1]) + (spkform.shape[-1]+10)*np.arange(spkform.shape[1])[:,None]
        plt.gcf().clear()
        for i in xrange(spkform.shape[0]):
            plt.subplot(spkform.shape[0],1,i+1)
            plt.plot(x.T,spkform[i].T)
        if save:
            plt.savefig(os.path.expanduser('~/Documents/research/figures/SpikeSorting/hmm/learn_example_init.pdf'),bbox='tight')
        else:
            plt.draw()
    #combine templates
    spkform,p = combineSpikes(spkform,p,cinv,data.shape[0])
    spikeForms['after_combine'] = {'spikeForms':spkform,'p':p}
    if saveToFile and len(p)>0:
        if not 'after_combine' in outFile:
            outFile.create_group('after_combine')
        outFile['after_combine']['spikeForms'] = spkform
        outFile['after_combine']['p'] = p
        outFile.flush()
    if debug:
        plt.gcf().clear()
        for i in xrange(spkform.shape[0]):
            plt.subplot(spkform.shape[0],1,i+1)
            plt.plot(x.T,spkform[i].T)
        if save:
            plt.savefig(os.path.expanduser('~/Documents/research/figures/SpikeSorting/hmm/learn_example_combined.pdf'),bbox='tight')
        else:
            plt.draw()
        
    if len(spkform)>0:
        #learn some more
        data,spkform,p,cinv = learnf(data,spkform,iterations=2,cinv=cinv,p=p,**kwargs)
        spikeForms['second_learning'] = {'a': {'spikeForms':spkform,'p':p}}
        if saveToFile and len(p)>0:
            if not 'second_learning' in outFile:
                outFile.create_group('second_learning')
            outFile['second_learning']['spikeForms'] = spkform
            outFile['second_learning']['p'] = p
            outFile.flush()

        #remove sparse waveforms
        if len(spkform)>0:
            spkform,p = removeSparse(spkform,p,splitp)
            spikeForms['second_learning']['after_sparse'] = {'spikeForms':spkform,'p':p}
            if saveToFile and len(p)>0:
                if not 'after_sparse' in outFile['second_learning']:
                    outFile['second_learning'].create_group('after_sparse')
                outFile['second_learning']['after_sparse']['spikeForms'] = spkform
                outFile['second_learning']['after_sparse']['p'] = p
                outFile.flush()
        if debug:
            plt.gcf().clear()
            for i in xrange(spkform.shape[0]):
                plt.subplot(spkform.shape[0],1,i+1)
                plt.plot(x.T,spkform[i].T)
            if save:
                plt.savefig(os.path.expanduser('~/Documents/research/figures/SpikeSorting/hmm/learn_example_final.pdf'),bbox='tight')
            else:
                plt.draw()

        #remove spikes that are too small
        if len(spkform)>0:
            spkform,p,idx = removeStn(spkform,p,cinv,data,kwargs.get('small_thresh',1))
            if saveToFile and len(p)>0:
                if not 'after_noise' in outFile['second_learning']:
                    outFile['second_learning'].create_group('after_noise')
                outFile['second_learning']['after_noise']['spikeForms'] = spkform
                outFile['second_learning']['after_noise']['p'] = p
                outFile.flush()


            print "Included because of sigma: "
            s = ['%d ' %(i,) for i in idx]
            print s
        if saveToFile:
            outFile.close()
            
    return spikeForms,cinv
    return spkform,p,cinv


def learndbw1(data,spkform=None,iterations=10,cinv=None,p=None,splitp=None,dosplit=True,states=60,**kwargs):
    """
    The function runs the baum-welch algorithm on the specified data. Data shauld have dimensions of datapts X channels
    """
    if spkform == None:
        neurons = 8
        levels = 4
        amp = np.random.random(size=(neurons,levels,))
        amp = amp/amp.max(1)[:,None]
        spkform = np.concatenate((np.zeros((levels,12)),np.sin(np.linspace(0,3*np.pi,states-42))[None,:].repeat(levels,0),np.zeros((levels,30))),axis=1)[None,:,:]*amp[:,:,None]*(np.median(np.abs(data),axis=0)/0.6745)[None,:,None]
        #spkform = spkform[None,:,:].repeat(8,0)
    else:
        neurons,levels,spklength = spkform.shape
    x = np.arange(spkform.shape[-1]) + (spkform.shape[-1]+10)*np.arange(spkform.shape[1])[:,None]
    for j in xrange(neurons):
        plt.subplot(neurons,1,j+1)
        plt.plot(x.T,spkform[j].T)
    plt.draw()
    N = len(spkform)
    if cinv == None:
        c = np.linalg.pinv(np.cov(data.T))
    else:
        c = cinv
    if p == None:
        p = 1.e-8*np.ones((N,))
    else:
        if len(p) < len(spkform):
            p = p.repeat(N,0)

    if splitp == None:
        splitp = 3.0/40000

    """
    if p.shape[0] > p.shape[1]:
        p = p.T
    """
    p_reset = p
    
    winlength,dim = data.shape
    spklength = spkform.shape[-1]
    #W = np.zeros((dim,N*(spklength-1)+1))
    W = spkform[:,:,1:].transpose((1,0,2)).reshape(dim,N*(spklength-1))
    W = np.concatenate((np.zeros((dim,1)),W),axis=1)
    """
    for i in xrange(N):
        W[:,1+(spklength-1)*i:(i+1)*(spklength-1)] = spkform[i,:,1:]
   """ 
    g = np.memmap(tempfile.TemporaryFile(),dtype=np.float,shape=(N*(spklength-1)+1,winlength),mode='w+')
    #fit = np.memmap(tempfile.TemporaryFile(),dtype=np.float,shape=(N*(spklength-1)+1,winlength),mode='w+')
    #this is an index vector
    q = np.concatenate(([N*(spklength-1)],np.arange(N*(spklength-1))),axis=0)
    tiny = np.exp(-700)

    for bw in xrange(iterations):
        p = p_reset
        #g = np.zeros((N*(spklength-1)+1,winlength))
        for i in xrange(g.shape[0]):
            g[i,:] = np.zeros((g.shape[1],))
        #fit = np.zeros(g.shape)
        b = np.zeros((g.shape[0],))
        g[0,0] = 1
        b[0] = 1
        #compute probabilities  
        #note that we are looping over number of states here; the most we are
        #creating is 2Xdata.nbytes
        """
        for i in xrange(W.shape[1]):
            X = W[:,i][:,None] - data.T.astype(np.float)
            fit[i,:] = np.exp(-0.5*(X*np.dot(c,X)).sum(0))
            #remove X
            del X
        """
        #X = W[:,:,None]-data.T[:,None,:]
        #fit = np.exp(-0.5*(X*np.dot(c,X.transpose((1,0,2))).sum(0)).sum(0))
        #F = np.array(fit.tolist()).reshape(fit.shape)
        #G = forward(g,F,spklength,N,winlength,p)
        #g = np.zeros((N*(spklength-1)+1,winlength))
        #g[0,0] = 1
        #forward
        print "Running forward algorithm..."
        sys.stdout.flush()
        for t in xrange(1,winlength):
            x = W-data[t,:][:,None]
            f = np.exp(-0.5*(x*np.dot(c,x)).sum(0))
            g[:,t] = g[q,t-1]
            g[0,t] = g[1:2+(N-1)*(spklength-1):(spklength-1),t].sum() + g[0,t] - g[0,t-1]*p.sum()
            g[1:2+(N-1)*(spklength-1):(spklength-1),t] = g[0,t-1]*p
            #g[:,t] = g[:,t]*fit[:,t]
            g[:,t] = g[:,t]*f[:]
            g[:,t] = g[:,t]/(g[:,t].sum()+tiny)

        #backward
        print "Running backward algorithm..."
        sys.stdout.flush()
        for t in xrange(winlength-2,-1,-1):
            x = W-data[t+1,:][:,None]
            f = np.exp(-0.5*(x*np.dot(c,x)).sum(0))
            #b = b*fit[:,t+1]
            b = b*f[:]
            b[q] = b
            b[0] = (1-p.sum())*b[-1] + np.dot(p,b[:(N-1)*(spklength-1)+1:(spklength-1)].T)
            b[(spklength-1):1+(N-1)*(spklength-1):(spklength-1)] = b[-1]
            b = b/(b.sum()+tiny)
            g[:,t] = g[:,t]*b
        
        g = g/(g.sum(0)+tiny)[None,:]
        #TODO: This stop could be quite memory intensive
        #W = np.dot(data.T,g.T)/g.sum(1)[None,:]
        W = np.zeros(W.shape)
        for i in xrange(data.shape[0]):
            W+=data[i,:][:,None]*g[:,i]
        W = W/g.sum(1)[None,:]
        W[:,0] = 0
        p = g[1::(spklength-1),:].sum(1).T/winlength
        cinv = np.linalg.pinv(np.cov((data-np.dot(g.T,W.T)).T))
       
        maxamp = np.zeros((len(spkform),))
        for j in xrange(len(spkform)):
            spkform[j] = np.concatenate((W[:,0][:,None],W[:,j*(spklength-1)+1:(j+1)*(spklength-1)+1]),axis=1)
            maxamp[j] = (spkform[j]*(np.dot(cinv,spkform[j]))).sum(0).max(0)

        nspikes = p*winlength

        print "Spikes found per template: " 
        print ' '.join((map(lambda s: '%.2f' %s,nspikes)))
        sys.stdout.flush()
        if dosplit:
            for i in xrange(len(spkform)):
                if p[i] < splitp:
                    try:
                        j = np.where((p>=np.median(p))*(p > splitp*4)*(maxamp>10))[0]
                        j = j[np.random.random_integers(size=(1,1,len(j)))]
                        W[:,i*(spklength-1)+1:(i+1)*(spklength-1)] = W[:,j*(spklength-1)+1:j*(spklength-1)]*.98
                        p[i] = p[j]/2
                        p[j] =p[j]/2
                        print "Waveformsupdate: %d <- %d" % (i,j)
                        sys.stdout.flush()
                        break
                    except:
                        print "Clustersplitting failed"
                        sys.stdout.flush()

        for j in xrange(len(spkform)):
            plt.subplot(neurons,1,j+1)
            plt.gca().clear()
            plt.plot(x.T,spkform[j].T)
        plt.draw()    

    #del g,fit 
    del g 
    for j in xrange(len(spkform)):
        spkform[j,:,:-1] = np.concatenate((W[:,1][:,None], W[:,j*(spklength-1)+1:(j+1)*(spklength-1)]),axis=1)
    
    return data,spkform,p,cinv

def learndbw1v2(data,spkform=None,iterations=10,cinv=None,p=None,splitp=None,dosplit=True,states=60,chunksize=10000,debug=False,**kwargs):
    """
    This function runs the baum-welch algorithm on the specified data, learning spiking templates. The input data should have dimensions of datapts X channels. This code runs on the data in chunks, offloading data to disk when not in use. This allows it to analyse arbitrarily long sequences of data.
    """

    if spkform == None:
        neurons = 8
        levels = 4
        amp = np.random.random(size=(neurons,levels,))
        amp = amp/amp.max(1)[:,None]
        spkform = np.concatenate((np.zeros((levels,12)),np.sin(np.linspace(0,3*np.pi,states-42))[None,:].repeat(levels,0),np.zeros((levels,30))),axis=1)[None,:,:]*amp[:,:,None]*(np.median(np.abs(data),axis=0)/0.6745)[None,:,None]
        #spkform = spkform[None,:,:].repeat(8,0)
    else:
        neurons,levels,spklength = spkform.shape
    x = np.arange(spkform.shape[-1]) + (spkform.shape[-1]+10)*np.arange(spkform.shape[1])[:,None]
    if debug:
        for j in xrange(neurons):
            plt.subplot(neurons,1,j+1)
            plt.plot(x.T,spkform[j].T)
        plt.draw()
    N = len(spkform)
    if cinv == None:
        c = np.linalg.pinv(np.cov(data.T))
    else:
        c = cinv
    if p == None:
        p = 1.e-8*np.ones((N,))
    else:
        if len(p) < len(spkform):
            p = p.repeat(N,0)

    if splitp == None:
        splitp = 3.0/40000

    """
    if p.shape[0] > p.shape[1]:
        p = p.T
    """
    p_reset = p
    
    winlength,dim = data.shape
    spklength = spkform.shape[-1]
    #W = np.zeros((dim,N*(spklength-1)+1))
    W = spkform[:,:,1:].transpose((1,0,2)).reshape(dim,N*(spklength-1))
    W = np.concatenate((np.zeros((dim,1)),W),axis=1)
    """
    for i in xrange(N):
        W[:,1+(spklength-1)*i:(i+1)*(spklength-1)] = spkform[i,:,1:]
   """ 
    #g = np.memmap(tempfile.TemporaryFile(),dtype=np.float,shape=(N*(spklength-1)+1,winlength),mode='w+')
    #fit = np.memmap(tempfile.TemporaryFile(),dtype=np.float,shape=(N*(spklength-1)+1,winlength),mode='w+')
    #this is an index vector
    q = np.concatenate(([N*(spklength-1)],np.arange(N*(spklength-1))),axis=0)
    tiny = np.exp(-700)
    nchunks = int(np.ceil(1.0*data.shape[0]/chunksize))
    chunks = np.append(np.arange(0,data.shape[0],chunksize),[data.shape[0]])
    chunksizes = np.diff(chunks)
    nchunks = len(chunksizes)
    for bw in xrange(iterations):
        fid = tempfile.TemporaryFile()
        p = p_reset
        #g = np.zeros((N*(spklength-1)+1,winlength))
        g = np.zeros((N*(spklength-1)+1,chunksize))
        #fit = np.zeros(g.shape)
        b = np.zeros((g.shape[0],))
        g[0,0] = 1
        b[0] = 1
        #compute probabilities  
        #note that we are looping over number of states here; the most we are
        #creating is 2Xdata.nbytes
        """
        for i in xrange(W.shape[1]):
            X = W[:,i][:,None] - data.T.astype(np.float)
            fit[i,:] = np.exp(-0.5*(X*np.dot(c,X)).sum(0))
            #remove X
            del X
        """
        #X = W[:,:,None]-data.T[:,None,:]
        #fit = np.exp(-0.5*(X*np.dot(c,X.transpose((1,0,2))).sum(0)).sum(0))
        #F = np.array(fit.tolist()).reshape(fit.shape)
        #G = forward(g,F,spklength,N,winlength,p)
        #g = np.zeros((N*(spklength-1)+1,winlength))
        #g[0,0] = 1
        #forward
        print "Running forward algorithm..."
        sys.stdout.flush()
        #do this in chunks
        for i in xrange(nchunks):
            print "Analyzing chunk %d of %d" % (i+1,nchunks) 
            for t in xrange(1,chunksizes[i]):
                a = chunks[i]+t
                y = W-data[a,:][:,None]
                f = np.exp(-0.5*(y*np.dot(c,y)).sum(0))+tiny
                g[:,t] = g[q,t-1]
                g[0,t] = g[1:2+(N-1)*(spklength-1):(spklength-1),t].sum() + g[0,t] - g[0,t-1]*p.sum()
                g[1:2+(N-1)*(spklength-1):(spklength-1),t] = g[0,t-1]*p
                #g[:,t] = g[:,t]*fit[:,t]
                g[:,t] = g[:,t]*f[:] + tiny
                g[:,t] = g[:,t]/(g[:,t].sum()+tiny)
            #store to file and reset for the next chunk
            g=g[:,:chunksizes[i]]
            try:
                g.tofile(fid)
            except ValueError:
                """
                for some reason, sometimes we get a value error here. If that
                happens, just report the exception and let sge know an error
                occured
                """
                traceback.print_exec(file=sys.stdout)
                sys.exit(99)

            g[:,0] = g[:,-1]

        #backward
        print "Running backward algorithm..."
        sys.stdout.flush()
        G = np.zeros((g.shape[0],)) 
        for i in xrange(nchunks-1,-1,-1):
            print "Analyzing chunk %d of %d" % (i+1,nchunks) 
            a = chunks[i]*(N*(spklength-1)+1)
            fid.seek(a*g[0,0].nbytes,0)
            g = np.fromfile(fid,dtype=np.float,count=(N*(spklength-1)+1)*chunksizes[i])
            g = g.reshape(N*(spklength-1)+1,chunksizes[i])
            for t in xrange(chunksizes[i]-2,-1,-1):
                a = chunks[i]+t+1
                y = W-data[a,:][:,None]
                f = np.exp(-0.5*(y*np.dot(c,y)).sum(0)) + tiny
                #b = b*fit[:,t+1]
                b = b*f[:] + tiny
                b[q] = b
                b[0] = (1-p.sum())*b[-1] + np.dot(p,b[:(N-1)*(spklength-1)+1:(spklength-1)].T)
                b[(spklength-1):1+(N-1)*(spklength-1):(spklength-1)] = b[-1]
                b = b/(b.sum()+tiny)
                g[:,t] = g[:,t]*b + tiny
            g = g/(g.sum(0)+tiny)
            G+=g.sum(1)
            #rewind file
            fid.seek(-(N*(spklength-1)+1)*chunksizes[i]*g[0,0].nbytes,1)
            g.tofile(fid)
        
        #g = g/(g.sum(0)+tiny)[None,:]
        #TODO: This stop could be quite memory intensive
        #W = np.dot(data.T,g.T)/g.sum(1)[None,:]
        W = np.zeros(W.shape)
        fid.seek(0)
        for i in xrange(nchunks):
            g = np.fromfile(fid,dtype=np.float,count=(N*(spklength-1)+1)*chunksizes[i]).reshape(N*(spklength-1)+1,chunksizes[i])
            W+=np.dot(data[chunks[i]:chunks[i+1],:].T,g.T)
        W = W/G[None,:]
        W[:,0] = 0
        p = np.zeros((N,))
        fid.seek(0)
        D = np.memmap(tempfile.TemporaryFile(),dtype=np.float,shape=data.shape,mode='w+')
        for i in xrange(nchunks):
            g = np.fromfile(fid,dtype=np.float,count=(N*(spklength-1)+1)*chunksizes[i])
            g = g.reshape(N*(spklength-1)+1,chunksizes[i])
            p+= g[1::(spklength-1),:].sum(1)
            fid.seek(g.nbytes,0)
            D[chunks[i]:chunks[i+1],:] = (W[:,:,None]*g[None,:,:]).sum(1).T
        fid.close() 
        p=p/winlength
        
        cinv = np.linalg.pinv(np.cov((data-D).T))
       
        maxamp = np.zeros((len(spkform),))
        for j in xrange(len(spkform)):
            spkform[j] = np.concatenate((W[:,0][:,None],W[:,j*(spklength-1)+1:(j+1)*(spklength-1)+1]),axis=1)
            maxamp[j] = (spkform[j]*(np.dot(cinv,spkform[j]))).sum(0).max(0)

        nspikes = p*winlength

        print "Spikes found per template: " 
        print ' '.join((map(lambda s: '%.2f' %s,nspikes)))
        sys.stdout.flush()
        if dosplit:
            for i in xrange(len(spkform)):
                if p[i] < splitp:
                    try:
                        j = np.where((p>=np.median(p))*(p > splitp*4)*(maxamp>10))[0]
                        j = j[np.random.random_integers(size=(1,1,len(j)))]
                        W[:,i*(spklength-1)+1:(i+1)*(spklength-1)] = W[:,j*(spklength-1)+1:j*(spklength-1)]*.98
                        p[i] = p[j]/2
                        p[j] =p[j]/2
                        print "Waveformsupdate: %d <- %d" % (i,j)
                        sys.stdout.flush()
                        break
                    except:
                        print "Clustersplitting failed"
                        sys.stdout.flush()

        if debug:
            for j in xrange(len(spkform)):
                plt.subplot(len(spkform),1,j+1)
                plt.gca().clear()
                plt.plot(x.T,spkform[j].T)
            plt.draw()    

    #del g,fit 
    del g 
    for j in xrange(len(spkform)):
        spkform[j,:,:-1] = np.concatenate((W[:,1][:,None], W[:,j*(spklength-1)+1:(j+1)*(spklength-1)]),axis=1)
    
    return data,spkform,p,cinv

def combineSpikes(spkform_old,pp,cinv,winlen,tolerance=4):

    winlen = winlen/tolerance
    alpha = 0.001
    maxp = 12.0
    spks,dim,spklen = spkform_old.shape
    j =-1 
    k = spks-1
    p = np.zeros((spks,))
    forms = np.zeros(spkform_old.shape)
    ind = np.zeros((spks,))
    for i in xrange(spks):
        t = np.trace(np.dot(np.dot(cinv,spkform_old[i]),spkform_old[i].T))
        if t < 3*spklen:
            forms[k] = spkform_old[i]
            p[k] = pp[i]
            ind[k] = i
            k-=1
        else:
            j+=1
            forms[j] = spkform_old[i]
            p[j] = pp[j]
            ind[j] = i

    spkform_old = forms
    spkform = spkform_old
    pp = p
    excl = np.array([j+1,spks-1]) 
    spks = j+1
    numspikes = j+1

    #combine larger spikes
    for r in xrange(numspikes):
        spklennew = spklen*10+10
        splineform = np.zeros((dim,spklennew,spks))
        splineform_test = np.zeros((dim,spklennew,spks))
        for i in xrange(spks):
            for j in xrange(dim):
                S = interpolate.spline(np.arange(spklen),spkform[i,j,:],np.linspace(0,spklen-1,spklen*10))
                splineform[j,:,i] = np.concatenate((np.zeros((10,)),S),axis=0)
                splineform_test[j,:,i] = np.concatenate((np.zeros((10,)),np.ones((spklen*10-10,)),np.zeros((10,))),axis=0)

        #calculate max similarity with respect to the first template and shift by
        #calculated index
        shift = np.ones((spks,))
        for i in xrange(1,spks):
            difference_old = np.inf
            for j in xrange(spklennew):
                shifted_template = np.concatenate((splineform[:,j:,i],splineform[:,:j,i]),axis=1)
                difference = np.trace(np.dot(splineform[:,:,0]-shifted_template,np.dot(cinv,splineform[:,:,0]-shifted_template).T))
                if difference < difference_old:
                    difference_old = difference
                    shift[i] = j

            splineform[:,:,i] = np.concatenate((splineform[:,shift[i]:,i],splineform[:,:shift[i],i]),axis=1)
            splineform_test[:,:,i] = np.concatenate((splineform_test[:,shift[i]:,i],splineform_test[:,:shift[i],i]),axis=1)
            
        index,docombine,value = combineTest(spks,splineform,splineform_test,cinv,winlen,p,maxp,alpha)

        while docombine:
            r+=1
            splineform[:,:,index[0]] = (p[index[0]]*splineform[:,:,index[0]] + p[index[1]]*splineform[:,:,index[1]])/p[index].sum()
            p[index[0]] = p[index].sum()
            s = splineform.shape
            s = s[:-1] + (s[-1]-1,)
            splineform_new = np.zeros(s)
            s = splineform_test.shape
            s = s[:-1] + (s[-1]-1,)
            splineform_test_new = np.zeros(s)
            p_new = np.zeros((spks-1,))
            shift_new = np.zeros((spks-1,))
            k =-1 
            for count in xrange(spks):
                if count != index[1]:
                    k+=1
                    splineform_new[:,:,k] = splineform[:,:,count]
                    splineform_test_new[:,:,k] = splineform_test[:,:,count]
                    p_new[k] = p[count]
                    shift_new[k] = shift[count]

            shift = shift_new
            splineform = splineform_new
            p = p_new
            splineform_test = splineform_test_new
            spks-=1
            index,docombine,value = combineTest(spks,splineform,splineform_test,cinv,winlen,p,maxp,alpha)

        print "Could not combine any more: value %.3f" %(value,)
        sys.stdout.flush()

        for i in xrange(1,spks):
            if shift[i]!=1:
                splineform[:,:,i] = np.concatenate((splineform[:,-shift[i]+1:,i],splineform[:,:-shift[i]+1,i]),axis=1)


        #downsample
        spkform = np.zeros((spks,)+spkform.shape[1:])
        for i in xrange(spks):
            S = np.zeros((splineform.shape[0],(splineform.shape[1]-1)/10-1))
            for k in xrange(dim):
                S[k] = interpolate.spline(np.arange(splineform.shape[1]),splineform[k,:,i],np.arange(21,splineform.shape[1]+1,10)-10)
            spkform[i] = np.concatenate((np.zeros((splineform.shape[0],1)),S),axis=1)

            
        #rotate order of the templates
        ptemp =np.zeros((spks,))
        forms = np.zeros(spkform.shape)
        for i in xrange(spks):
            forms[i] = spkform[i % spks]
            ptemp[i] = p[i % spks]

        spkform = forms
        p = ptemp

    #add excluded templates
    i = spks-1
    #make space for the small, excluded templates
    #plus 1 because excl are proper indices
    spkform = np.resize(spkform,(spks+excl[1]-excl[0]+1,) + spkform.shape[1:])
    p = np.resize(p,(spks+excl[1]-excl[0]+1,))
    spkform[spks:] = spkform_old[excl[0]:]
    p[spks:]  = pp[excl[0]:]
    """
    for j in xrange(excl[0],excl[1]):
        i+=1
        spkform[i] = spkform_old[j]
        p[i] = pp[j]
    """

    return spkform,p 

def combineTest(spks,splineform,splineform_test,cinv,winlen,p,maxp,alpha):

    if spks == 1:
        index = 1
        docombine = 0
        value = 0
        return index,docombine,value
    
    splineform = splineform[:,splineform_test[0,:,:].sum(-1)==spks,:]
    h = np.zeros((spks,spks))
    pvalt = np.zeros((spks,spks))
    teststat = np.zeros((spks,spks))

    for i in xrange(spks-1):
        for j in xrange(i+1,spks):
            diffnorm = np.zeros((splineform.shape[0],splineform.shape[1]/10+1))
            for k in xrange(splineform.shape[0]):
                diffnorm[k,:] = interpolate.spline(np.arange(splineform.shape[1]),splineform[k,:,i]-splineform[k,:,j],np.arange(0,splineform.shape[1]+1,10))
            teststat[i,j] = np.trace(np.dot(diffnorm.T,np.dot(cinv,diffnorm)))/(1.0/(min(winlen*p[i],maxp))+1.0/min(winlen*p[j],maxp))
            if teststat[i,j] < diffnorm.size:
                pvalt[i,j] = 1
            else:
                #compute variance
                v = np.concatenate(([np.sqrt(teststat[i,j])/2,-np.sqrt(teststat[i,j])/2],np.zeros((diffnorm.size-2,))),axis=0).var()
                h[i,j],pvalt[i,j] = (diffnorm.size-1)*v,stats.chisqprob((diffnorm.size-1)*v,diffnorm.size-1)

            #symmetry
            pvalt[j,i] = pvalt[i,j]
            

    teststat += teststat.T + np.eye(teststat.shape[0])*teststat.max(0).max()

    b = np.argmin(teststat,1)
    a = teststat[np.arange(len(b)),b]
    c = np.argmin(a)
    a = a[c]

    index = np.array([c,b[c]])
    maximum = pvalt[index[0],index[1]]
    value = teststat[index[0],index[1]]/diffnorm.size

    #check confidence level
    if maximum > alpha:
        print "combine spikes, p-value: %.3f" % (maximum,)
        print "combine spikes, test-value: %.3f" %(teststat[index[0],index[1]],)
        print "overlapping area: %.3f" %(np.floor(splineform.shape[1]/10.0),)
        sys.stdout.flush()
        docombine = 1
    else:
        docombine = 0

    return index,docombine,value


def removeSparse(spkform,p,splitp):
    """
    removes templates that fire too sparsely
    """
    idx = p>splitp
    spkform = spkform[idx]
    p = p[idx]
    return spkform,p

def removeStn(spkform,p,cinv,data=None,small_thresh=1,nsamples=1000):
    """
    Remove templates that do not exceed the twice the energy of an average noise
    patch
    """
    if data == None:
        limit = spkform.shape[-1]*3
    else:
        tmp = spkform.shape[-1]
        test = np.zeros((nsamples,))
        #pick some random patches
        idx = np.random.random_integers(0,data.shape[0]-spkform.shape[-1],size=(nsamples,))
        for i in xrange(nsamples):
            x = data[idx[i]:idx[i]+spkform.shape[-1],:].T
            test[i] = (x*np.dot(cinv,x)).sum()
        limit = np.median(test)
    
    j = -1
    ind = []
    woe = np.zeros((spkform.shape[0],))
    pp = [] 
    new_spkform = []

    for i in xrange(spkform.shape[0]):
        woe[i] = (spkform[i]*np.dot(cinv,spkform[i])).sum()
        if woe[i] >= limit*small_thresh:
            j+=1
            new_spkform.append(spkform[i])
            pp.append(p[i])
            ind.append(i)

    return np.array(new_spkform),np.array(pp),np.array(ind)


if __name__ == '__main__':
    
    import getopt

    opts,args = getopt.getopt(sys.argv[1:],'',longopts=['sourceFile=','group=','minFiringRate=','outFile=','combine','chunkSize=','version=','debug','fileChunkSize=','redo'])

    opts = dict(opts)

    dataFileName = opts.get('--sourceFile')
    outFileName = opts.get('--outFile')
    group = int(opts.get('--group','1'))
    splitp = np.float(opts.get('--minFiringRate','0.5'))
    chunkSize = min(np.float(opts.get('--chunkSize','50000')),1.0e6)
    version = int(opts.get('--version','2'))
    debug = opts.has_key('--debug')
    redo = opts.has_key('--redo')
    if '--combine' in opts:
#get all the data file, read the spkforms from each, then combine them 
        #get the descriptor, if any
        descriptorFile = glob.glob('*_descriptor.txt')
        if len(descriptorFile)==0:
            print "No descriptpr file found. Exiting.."
            sys.exit(3)
        descriptorFile = descriptorFile[0]
        #get the base from the descriptor file
        base = descriptorFile[:descriptorFile.index('_descriptor')]
        #get the sort files
        sortFiles = glob.glob('hmmsort/%s_highpassg%.4d.*.hdf5'% (base,group))
        if dataFileName == None:
            #try to guess from the group 
            pass
        else:
            files = opts.get('--sourceFile','').split(',')
        dataFileName = files[0]
        spkforms = []
        p = []
        useFiles = []
        for f in sortFiles:
            try:
                dataFile = h5py.File(f,'r')
                spkforms.extend(dataFile['all']['spikeForms'][:])
                p.extend(dataFile['all']['p'][:])
                dataFile.close()
                useFiles.append(f)
            except:
                continue
        files = useFiles   
        spkforms = np.array(spkforms) 
        p = np.array(p)
#get descriptor information
        #base = dataFileName[:dataFileName.rfind('_')]
        #descriptorFile = '%s_descriptor.txt' % (dataFileName[:dataFileName.rfind('_')],)
        #if not os.path.isfile(descriptorFile):
#sometimes the descriptor is located one level up
        #    descriptorFile = '../%s' % (descriptorFile,)
        descriptor = fr.readDescriptor(descriptorFile)
        channels = np.where(descriptor['gr_nr']==group)[0]
        nchs = sum(descriptor['gr_nr']>0)
        """
#here it becomes tricky; if the combined data file has already been
#reordered, we need to get the channels in the reordering scheme
            reorder = np.loadtxt('reorder.txt',dtype=np.uint16)
            channels = np.where(np.lib.arraysetops.in1d(reorder,channels))[0]
#compute covariance matrix on the full dataset


#spkforms,p = combineSpikes(spkforms,p,cinv,winlen)
        """
#gather all files to compute covariance matrix
        """
        if descriptorFile[:2] == '..':
            files = glob.glob('../*_highpass.[0-9]*')
        else:
            files = glob.glob('*_highpass.[0-9]*')
        """
        dataFiles = glob.glob('%s_highpass.[0-9]*' % (base,))
        sizes =  [os.stat(f).st_size for f in dataFiles]
        total_size = ((np.array(sizes)-73)/2/nchs).sum()
        alldata = np.memmap('/tmp/%s.all' %(base,),dtype=np.int16,shape=(len(channels),total_size),mode='w+')
        offset = 0
        for f in dataFiles:
            print "Loading data from file %s..." %(f,)
            sys.stdout.flush()
            data,sr = extraction.readDataFile(f)
            
            #data = np.memmap(f,mode='r',dtype=np.int16,offset=73,shape=((os.stat(f).st_size-73)/2/nchs,nchs))
            alldata[:,offset:offset+data.shape[0]] = data[:,channels].T
            offset+=data.shape[0]
        
        print "Computing inverse covariance matrix..."
        sys.stdout.flush()
        cinv = np.linalg.pinv(np.cov(alldata))
        winlen = alldata.shape[1]
        dataFile = h5py.File('%sg%.4d.hdf5' %(base,group),'a')
        try:
            dataFile['cinv'] = cinv
            dataFile.create_group('spikeFormsAll')
            dataFile['spikeFormsAll']['spikeForms'] = spkforms
            dataFile['spikeFormsAll']['p'] = p 
            dataFile.flush()
        except:
#field already exists
            pass
#remove small waveforms
        print "Removing small templates..."
        sys.stdout.flush()
        q = len(p)
        spkforms,p,idx = removeStn(spkforms,p,cinv,alldata.T)
        print "Removed %d templates.." %( q-len(p),)
        if len(spkforms)>0:
            dataFile.create_group('spikeFormsLarge')
            dataFile['spikeFormsLarge']['spikeForms'] = spkforms
            dataFile['spikeFormsLarge']['p'] = p

        if len(spkforms)>1:
            print "Combining templates..."
            sys.stdout.flush()
            spkforms,p = combineSpikes(spkforms,p,cinv,winlen)
            print "Left with %d templates .." %(len(p),)
        if len(spkforms)>0:
            dataFile['spikeForms'] = spkforms
            dataFile['p'] = p
            dataFile.close()

    else:
        #check for the presence of an SGE_TASK_ID
        tid = None
        nchunks = None
        if 'SGE_TASK_ID' in os.environ:
            #signifies that we should use split the file
            tfirst = int(os.environ.get('SGE_TASK_FIRST',0))
            tlast = int(os.environ.get('SGE_TASK_LAST',0))
            nchunks = tlast-tfirst+1
            tid = int(os.environ['SGE_TASK_ID'])-1
            print "Analyzing file %s in %d chunks. Analyzing chunk %d...." %(dataFileName,nchunks,tid+1)
            sys.stdout.flush()
            
        try:
            spikeForms,cinv = learnTemplatesFromFile(dataFileName,group,splitp = splitp,outfile=outFileName,chunksize=chunkSize,version=version,debug=debug,nFileChunks=nchunks,fileChunkId=tid)
        except IOError:
            print "Could not read/write to file"
            sys.exit(99)

