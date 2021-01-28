#!~/anaconda2/bin/python
"""@package hmm_learn
This package contains a set of functions to learn spike templates from highpass
data
"""

import numpy as np
import scipy.stats as stats
import sys
import tempfile
import os
import h5py
import glob
import traceback
from hmmsort import fileReaders as fr
import scipy.interpolate as interpolate
import scipy.io as mio
from hmmsort import extraction
import time
import blosc

from hmmsort import utility

np.seterr(all='warn')
#only raise an error if we are dividing by zero; this usually means we made a
#mistake somewhere
np.seterr(divide='raise')
#if os.path.isdir('/Volumes/Chimera/tmp'):
#    tempfile.tempdir = '/Volumes/Chimera/tmp'
#elif os.path.isdir('/Volumes/DataX/tmp'):
#    tempfile.tempdir = '/Volumes/DataX/tmp'

def gatherSpikeFormsFromGroup(group=1,sessionName=None,baseDir='hmmsort',globPattern=None):
    if globPattern == None:
        if sessionName == None:
            #assume
            pass

    files = glob.glob(globPattern)
    spikeForms = []
    p = []
    for f in files:
        try:
            dataFile = h5py.File(f,'r')
        except:
            continue
        try:
            spikeForms.extend(dataFile['after_noise']['spikeForms'][:])
            p.extend(dataFile['after_noise']['p'][:])
        except:
            pass
        finally:
            dataFile.close()
    spikeForms = np.array(spikeForms)
    p = np.array(p)

    return spikeForms,p

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
    #err = weave.inline(code,['p','_np','q','g','winlength','P','spklength','M'])
    return g

def learnTemplatesFromFile(dataFile,group=None,channels=None,save=True,outfile=None,chunksize=1000000,nFileChunks=None,fileChunkId=None,divideByGain=False,reorder=False, max_size=None, offset=0, **kwargs):
    """
    Learns templates from the file pointed to by dataFile. The data should be
    stored channel wise in int16 format, i.e. the resulting array should have
    dimensions timepoints X channels. The first 4 bytes of the file should
    contain an uint32 designating the size of the header. The header itself
    should contain the number of channels, encoded as an uint8, and the sampling
    rate, encoded as an uint32.
    Group indicates which tetrode group to analyze; the channels will be found
    by reading the descriptor file corresponding to the requested data file. The
    variable chunksize indicates the amount of data to load into memory. The
    variables nFileChunks and fileChunkId indicate, respectively, the number of
    file that the data file is divide into, and the chunk to process.
    """
    if not os.path.isfile(dataFile):
        print "File at path %s could not be found " % (dataFile,)
        return [], []

    print "Reading data from file %s" %(dataFile, )
    # check what kind of file we are dealing with
    fname,ext = os.path.splitext(dataFile)
    if ext == '.mat':
        if h5py.is_hdf5(dataFile):
            ff = h5py.File(dataFile,'r')
            if "rh" in ff.keys():
                data = ff["rh/data/analogData"][:].flatten()
                sampling_rate = ff["rh/data/analogInfo/SampleRate"][:].flatten()
            elif "highpassdata" in ff.keys():  # this is a non-object file
                data = ff["highpassdata/data/data"][:].flatten()
                sampling_rate = ff["highpassdata/data/sampling_rate"][:].flatten()
            ff.close()
        else:
            rdata = mio.loadmat(dataFile)
            data = rdata['rplhighpass']['data'][0,0]['analogData']
            sampling_rate = rdata['rplhighpass']['data'][0,0]['analogInfo'][0,0]['sampleRate']
    else:
        data,sr = extraction.readDataFile(dataFile)
        sampling_rate = sr
    head,tail = os.path.split(dataFile)
    if data.ndim == 2:
        nchannels = data.shape[0]
        descriptorFile = '%s_descriptor.txt' % (tail[:tail.rfind('_')],)
        if not os.path.isfile(descriptorFile):
            descriptor = {'gr_nr': np.arange(1,nchannels+1),
                          'ch_nr':np.arange(nchannels),
                          'channel_status': np.ones((nchannels,),dtype=np.bool)}
        else:
            #get group information form the descriptor
            descriptor = fr.readDescriptor(descriptorFile)
        if reorder == True:
            reorder = np.loadtxt('reorder.txt',dtype=np.int)-1
            data = data[reorder, :]
        if np.all(channels == None):
            channels = np.where(descriptor['gr_nr'][descriptor['channel_status']]==group)[0]
        else:
            group = descriptor['gr_nr'][np.lib.arraysetops.in1d(descriptor['ch_nr'],channels)]
            group = np.unique(group)
        cdata = data[channels, :].T.copy(order='C')
        del data
    else:
        cdata = data[:,None]
    if divideByGain and 'gain' in descriptor:
        cdata = cdata/np.float(descriptor['gain'])

    if max_size is not None:
        cdata = cdata[offset:offset+max_size,:]
    if nFileChunks!=None and fileChunkId!=None:
        #we should only process parts of the file
        fileChunkSize = np.ceil(1.0*cdata.shape[0]/nFileChunks)

        cdata = cdata[fileChunkId*fileChunkSize:(fileChunkId+1)*fileChunkSize, :]
    if save:
        if outfile == None:
            name,ext = os.path.splitext(tail)
            if not ext[1:].isdigit():  # if the ext is not a chunkindex
                # this is to avoid having .eg. mat.hdf5 as an extension
                ext = ""
            name = name.replace('highpass','templates')
            if not os.path.isdir('hmmsort'):
                os.mkdir('hmmsort')
            if fileChunkId == None:
                outfile = 'hmmsort/%sg%.4d%s.hdf5' % (name,group,ext)
            else:
                outfile = 'hmmsort/%sg%.4d%s.%d.hdf5' % (name,group,ext,fileChunkId)
        else: #if we did specify an outfile, make sure that the directory exist
            pth,fname = os.path.split(outfile)
            if not pth:
                pth = "."
            if not os.path.isdir(pth):
                os.mkdir(pth)
        try:
            outf = h5py.File(outfile,'a')
        except IOError:
            #file exists; what do we do?
            sys.stderr.write("An error occurred trying to open the file %s...\n" %(outfile,))
	    sys.stderr.flush()
            sys.exit(0)
    if save:
        outf.close()
    else:
        outfile = False
    spikeForms,cinv = learnTemplates(cdata,samplingRate=sampling_rate,
                                     chunksize=chunksize,
                                     saveToFile=outfile,**kwargs)
    if spikeForms != None and 'second_learning' in spikeForms and spikeForms['second_learning']['after_sparse']['spikeForms'].shape[0]>=1:
        if save:
            #reopen to save the last result
            try:
                outf.close()
            except ValueError:
                #this probably means that the file was not opened
                pass
            outf = h5py.File(outfile,'a')
            try:
                outf['spikeForms'] = spikeForms['second_learning']['after_noise']['spikeForms']
                outf['p'] = spikeForms['second_learning']['after_noise']['p']
                outf['cinv'] = cinv
                outf.flush()
                outf.close()
            except:
                pass
    else:
        print "No spikeforms found"

    #make sure we close the file
    if save:
        #this will fail some times; make sure the erorr doesn't propagate in
       #that case
        try:
            outf.close()
        except ValueError:
            pass


    return spikeForms,cinv

def learnTemplates(data,splitp=None,debug=True,save=False,samplingRate=None, saveToFile=False,redo=False,iterations=3,spike_length=1.5, maxp=12.0, **kwargs):
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
    if samplingRate == None:
        samplingRate = 30000.0
    states = kwargs.get('states')
    if states is None:
        #set the number of states corresponding to 1.5 ms
        states = int(np.ceil(spike_length*samplingRate/1000.))
        kwargs['states'] = states
    if splitp == None:
        #set the minimum firing rate at 0.5 Hz
        splitp = 0.5/samplingRate
    else:
        splitp = splitp/samplingRate
    if save:
        #open a file to save the spkforms to
        pass
    learnf = utility.learn
    if saveToFile:
        try:
            outFile = h5py.File(saveToFile,'a')
            print "Saving to file %s" % (saveToFile,)
        except IOError:
            print "Could not open file %s..." % (saveToFile,)
            saveToFile = False
    spikeForms = {}
    if saveToFile:
        if not redo:
            if 'all' in outFile:
                spkform = outFile['all']['spikeForms'][:]
                p = outFile['all']['p'][:]
                cinv = outFile['cinv'][:]
                spikeForms['all']  = {'spikeForms': spkform,
                                   'p': p}
            if 'after_combine' in outFile:
                spkform = outFile['after_combine']['spikeForms'][:]
                p = outFile['after_combine']['p'][:]
                spikeForms['after_combine']  = {'spikeForms': spkform,
                                   'p': p}
            if 'after_sparse' in outFile:
                spkform = outFile['after_sparse']['spikeForms'][:]
                p = outFile['after_sparse']['p'][:]
                spikeForms['after_sparse'] = {'spikeForms': spkform,
                                   'p': p}
            if 'after_noise' in outFile:
                spkform = outFile['after_noise']['spikeForms'][:]
                p = outFile['after_noise']['p'][:]
                spikeForms['after_noise']  = {'spikeForms': spkform,
                                   'p': p}
            if 'second_learning' in outFile:
                if 'spikeForms' in outFile['second_learning']:
                    spkform = outFile['second_learning']['spikeForms'][:]
                    p = outFile['second_learning']['p'][:]
                    spikeForms['second_learning'] = {'spikeForms': spkform,
                                                     'p': p}
                    if 'after_sparse' in outFile['second_learning']:
                        spkform = outFile['second_learning']['after_sparse']['spikeForms'][:]
                        p = outFile['second_learning']['after_sparse']['p'][:]
                        spikeForms['second_learning']['after_sparse'] = {'spikeForms': spkform,
                                                     'p': p}
                    if 'after_noise' in outFile['second_learning']:
                        spkform = outFile['second_learning']['after_noise']['spikeForms'][:]
                        p = outFile['second_learning']['after_noise']['p'][:]
                        spikeForms['second_learning']['after_noise'] = {'spikeForms': spkform,
                                                     'p': p}

    if not 'all' in spikeForms:
        ss, mm = extraction.computeStd(data.T, 4)
        ss = ss/4
        ss = ss*ss
        cinv = 1.0/ss
        data,spkform,p,cinv = learnf(data,iterations=iterations,debug=debug,
                                     levels=data.shape[1],cinv=cinv, **kwargs)
        try:
            outFile['cinv'] = cinv
        except:
            print "Could not save inverse covariance matrix"
        spikeForms['all'] = {'spikeForms': spkform,'p': p}
        if saveToFile:
            if not 'all' in outFile:
                outFile.create_group('all')
            outFile['all']['spikeForms'] = spkform
            outFile['all']['p'] = p
            outFile.flush()
    if not 'after_combine' in spikeForms:
        spkform,p = combineSpikes(spkform,p,cinv,data.shape[0],maxp=maxp,
                                 tolerance=1)
        spikeForms['after_combine'] = {'spikeForms':spkform,'p':p}
        if saveToFile and len(p)>0:
            if not 'after_combine' in outFile:
                outFile.create_group('after_combine')
            outFile['after_combine']['spikeForms'] = spkform
            outFile['after_combine']['p'] = p
            outFile.flush()
    else:
        spkform = spikeForms['after_combine']['spikeForms']
        p = spikeForms['after_combine']['p']
    if not 'after_noise' in spikeForms:
        spkform,p,idx = removeStn(spkform,p,cinv,data,kwargs.get('min_snr',4.0))
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
    else:
        spkform = spikeForms['after_noise']['spikeForms']
        p = spikeForms['after_noise']['p']
    if not 'after_sparse' in spikeForms:
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
    else:
        spkform = spikeForms['after_sparse']['spikeForms']
        p = spikeForms['after_sparse']['p']

    if len(spkform)>0:
        if not 'second_learning' in spikeForms:
            #learn some more
            data,spkform,p,cinv = learnf(data,spkform,iterations=2,cinv=cinv,p=p,**kwargs)
            spikeForms['second_learning'] = {'spikeForms':spkform,'p':p}
            if saveToFile and len(p)>0:
                if not 'second_learning' in outFile:
                    outFile.create_group('second_learning')
                outFile['second_learning']['spikeForms'] = spkform
                outFile['second_learning']['p'] = p
                outFile['cinv'][:] = cinv
                outFile.flush()
        else:
            spkform = spikeForms['second_learning']['spikeForms']
            p = spikeForms['second_learning']['p']

        #remove sparse waveforms
        if len(spkform)>0:
            if not 'after_sparse' in spikeForms['second_learning']:
                spkform,p = removeSparse(spkform,p,splitp)
                spikeForms['second_learning']['after_sparse'] = {'spikeForms':spkform,'p':p}
                if saveToFile and len(p)>0:
                    if not 'after_sparse' in outFile['second_learning']:
                        outFile['second_learning'].create_group('after_sparse')
                    outFile['second_learning']['after_sparse']['spikeForms'] = spkform
                    outFile['second_learning']['after_sparse']['p'] = p
                    outFile.flush()
            else:
                spkform = spikeForms['second_learning']['after_sparse']['spikeForms']
                p = spikeForms['second_learning']['after_sparse']['p']


        #remove spikes that are too small
        if len(spkform)>0:
            if not 'after_noise' in spikeForms['second_learning']:
                spkform,p,idx = removeStn(spkform,p,cinv,data,kwargs.get('small_thresh',1))
                if saveToFile and len(p)>0:
                    if not 'after_noise' in outFile['second_learning']:
                        outFile['second_learning'].create_group('after_noise')
                    outFile['second_learning']['after_noise']['spikeForms'] = spkform
                    outFile['second_learning']['after_noise']['p'] = p
                    outFile.flush()
            else:
                spkform = spikeForms['second_learning']['after_noise']['spikeForms']
                p = spikeForms['second_learning']['after_noise']['p']
                idx = range(len(p))

            print "Included because of sigma: "
            s = ['%d ' %(i,) for i in idx]
            print s
            spikeForms['second_learning']['after_noise'] = {'spikeForms':spkform,'p':p}
        if saveToFile:
            outFile.close()

    return spikeForms,cinv


def combineSpikes(spkform_old,pp,cinv,winlen,tolerance=4,
                  alpha=0.001,maxp=12.0):

    winlen = winlen/tolerance
    spks,dim,spklen = spkform_old.shape
    j =-1
    k = spks-1
    p = np.zeros((spks,))
    forms = np.zeros(spkform_old.shape)
    ind = np.zeros((spks,))
    for i in xrange(spks):
        t = np.trace(np.dot(np.dot(np.atleast_2d(cinv),spkform_old[i]),spkform_old[i].T))
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
    #for r in xrange(numspikes):
    r =0
    while r < numspikes:
        r+=1
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
        shift = np.ones((spks,), dtype=np.int)
        for i in xrange(1,spks):
            difference_old = np.inf
            for j in xrange(spklennew):
                shifted_template = np.concatenate((splineform[:,j:,i],splineform[:,:j,i]),axis=1)
                difference = np.trace(np.dot(splineform[:,:,0]-shifted_template,np.dot(np.atleast_2d(cinv),splineform[:,:,0]-shifted_template).T))
                if difference < difference_old:
                    difference_old = difference
                    shift[i] = j

            splineform[:,:,i] = np.concatenate((splineform[:,shift[i]:,i],splineform[:,:shift[i],i]),axis=1)
            splineform_test[:,:,i] = np.concatenate((splineform_test[:,shift[i]:,i],splineform_test[:,:shift[i],i]),axis=1)

        index,docombine,value = combineTest(spks,splineform,splineform_test,cinv,winlen,p,maxp,alpha)

        while docombine:
            r+=1
            splineform[:,:,index[0]] = (p[index[0]]*splineform[:,:,index[0]] +
                                        p[index[1]]*splineform[:,:,index[1]])/p[index].sum()
            p[index[0]] = p[index].sum()
            s = splineform.shape
            s = s[:-1] + (s[-1]-1,)
            splineform_new = np.zeros(s)
            s = splineform_test.shape
            s = s[:-1] + (s[-1]-1,)
            splineform_test_new = np.zeros(s)
            p_new = np.zeros((spks-1,))
            shift_new = np.zeros((spks-1,), dtype=np.int)
            k =- 1
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

        print "Could not combine any more: value %.3f" % (value,)
        sys.stdout.flush()

        #shift back
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
        #ptemp =np.zeros((spks,))
        #forms = np.zeros(spkform.shape)
        spkform = np.roll(spkform,-1,axis=0)
        p = np.roll(p,-1)
        """
        for i in xrange(spks):
            forms[i] = spkform[i % spks]
            ptemp[i] = p[i % spks]
        spkform = forms
        p = ptemp
        """

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
    _cinv = np.atleast_2d(cinv)
    splineform = splineform[:,splineform_test[0,:,:].sum(-1)==spks,:]
    h = np.zeros((spks,spks))
    pvalt = np.zeros((spks,spks))
    teststat = np.zeros((spks,spks))

    for i in xrange(spks-1):
        for j in xrange(i+1,spks):
            diffnorm = np.zeros((splineform.shape[0],splineform.shape[1]/10+1))
            for k in xrange(splineform.shape[0]):
                diffnorm[k,:] = interpolate.spline(np.arange(splineform.shape[1]),splineform[k,:,i]-splineform[k,:,j],np.arange(0,splineform.shape[1]+1,10))
            teststat[i,j] = np.trace(np.dot(diffnorm.T,np.dot(_cinv,diffnorm)))/(1.0/(min(winlen*p[i],maxp))+1.0/min(winlen*p[j],maxp))
            if teststat[i,j] < diffnorm.size:
                pvalt[i,j] = 1
            else:
                #compute variance
                v = np.concatenate(([np.sqrt(teststat[i,j])/2,-np.sqrt(teststat[i,j])/2],np.zeros((diffnorm.size-2,))),axis=0).var()
                h[i,j],pvalt[i,j] = (diffnorm.size-1)*v,stats.distributions.chi2.sf((diffnorm.size-1)*v,diffnorm.size-1)

            #symmetry
            pvalt[j,i] = pvalt[i,j]


    teststat += teststat.T + np.eye(teststat.shape[0])*teststat.max(0).max()*10

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
    if data is None:
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

    import getopt
    try:

        opts,args = getopt.getopt(sys.argv[1:],'',longopts=['sourceFile=','group=',
                                                            'minFiringRate=','outFile=',
                                                            'combine','chunkSize=',
                                                            'debug',
                                                            'fileChunkSize=','redo',
                                                            'max_size=', 'offset=',
                                                            'basePath=','channels=',
                                                            'reorder','iterations=',
                                                            'tempPath=',
                                                            'outputFile=',
                                                            'initFile=','states=','initOnly', 'maxp=', 'min_snr=',
                                                            'states='])

        if len(sys.argv) == 1:
            #print help message and quit
            print """Usage: hmm_learn.py --sourceFile <sourceFile> --group
            <channle number> --outFile <outfile name>  [--chunkSize 100000]
            [--minFiringRate 0.5 ] [--iterations 3] [--initOnly] [--max_size INF] [--maxp 12.0] [--min_snr 4.0] [--states 45]
            """

            sys.exit(0)
        opts = dict(opts)

        dataFileName = opts.get('--sourceFile')
        outFileName = opts.get('--outFile')
        group = int(opts.get('--group','1'))
        splitp = np.float(opts.get('--minFiringRate','0.5'))
        chunkSize = min(int(opts.get('--chunkSize','100000')),100000)
        maxSize = int(opts.get('--max_size', sys.maxint))
        offset = int(opts.get('--offset', 0))
        debug = opts.has_key('--debug')
        redo = opts.has_key('--redo')
        reoder = opts.has_key('--reorder')
        iterations = int(opts.get('--iterations',3))
        tempPath = opts.get('--tempPath','/Volumes/Scratch')
        initFile = opts.get('--initFile')
        states = opts.get('--states')
        initOnly = '--initOnly' in opts.keys()
        maxp = np.float(opts.get('--maxp', 12.0))
        min_snr = np.float(opts.get('--min_snr', 4.0))
        if initOnly:
            if outFileName is not None:
                #simply initialize an empty hdf5 file
                pth,fname = os.path.split(outFileName)
                if not pth:
                    pth = "."
                if not os.path.isdir(pth):
                    os.mkdir(pth)
                if not h5py.is_hdf5(outFileName):
                    outf = h5py.File(outFileName,'a')
                    outf.close()
            sys.exit(0)

        if states is not None:
            states = int(states)
        if not os.path.isdir(tempPath):
            print """The Requested tempPath %s does not exist. Reverting to the
            system default""" % (tempPath,)
            tempPath = None
        #parse the channel input
        channels = None
        if '--channels' in opts:
            chs = opts['channels']
            chs = chs.split(',')
            channels = []
            for c in chs:
                if '-' in c:
                    start,_,stop = c.partition('-')
                    channels.extend(range(int(start),int(stop)+1))
                else:
                    channels.append(int(c))

        if '--combine' in opts:
#get all the data file, read the spkforms from each, then combine them
            #get the descriptor, if any
            descriptorFile = glob.glob('*_descriptor.txt')
            if len(descriptorFile)==0:
                sys.stderr.write("No descriptpr file found. Exiting..\n")
		sys.stderr.flush()
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
            #dataFileName = files[0]
            spkforms = []
            p = []
            useFiles = []
            for f in sortFiles:
                try:
                    dataFile = h5py.File(f,'r')
                    spkforms.extend(dataFile['after_noise']['spikeForms'][:])
                    p.extend(dataFile['after_noise']['p'][:])
                    dataFile.close()
                    useFiles.append(f)
                except:
                    continue
            files = useFiles
            spkforms = np.array(spkforms)
            p = np.array(p)
            print "Found a total of %d spikeforms..." % (spkforms.shape[0],)
#get descriptor information
            #base = dataFileName[:dataFileName.rfind('_')]
            #descriptorFile = '%s_descriptor.txt' % (dataFileName[:dataFileName.rfind('_')],)
            #if not os.path.isfile(descriptorFile):
#sometimes the descriptor is located one level up
            #    descriptorFile = '../%s' % (descriptorFile,)
            descriptor = fr.readDescriptor(descriptorFile)
            channels = np.where(descriptor['gr_nr'][descriptor['channel_status']]==group)[0]
            #check for the presence of a session hdf5 file
            cfileName = '%s.hdf5' %( base,)
            cinv = None
            winlen = None
            alldata = None
            if redo:
                dataFile = h5py.File('hmmsort/%sg%.4d.hdf5' %(base,group),'w')
            else:
                dataFile = h5py.File('hmmsort/%sg%.4d.hdf5' %(base,group),'a')
            if os.path.isfile(cfileName):
                cfile = h5py.File(cfileName,'r')
                winlen = cfile['highpass'].shape[1]
                alldata = cfile['highpass'][channels,:]
                if 'highpass_cov' in cfile:
                    c = cfile['highpass_cov'][:][channels[:,None],channels[None,:]]
                    cinv = np.linalg.pinv(c)
                else:
                    highpass = cfile['highpass']
                    C = np.cov(alldata)
                    cinv = np.linalg.pinv(C)

            if np.all(cinv == None):
                if os.path.isfile('%s_highpass.bin' %(base,)):
                    nchs = descriptor['channel_status'].sum()
                    alldata,sr = extraction.readDataFile('%s_highpass.bin' %(
                        base,))
                    alldata = alldata[channels,:]
                else:
                    if '--basePath' in opts:
                        dataFiles = glob.glob('%s/%s_highpass.[0-9]*' % (opts.get('--basePath'),base,))
                    else:
                        dataFiles = glob.glob('%s_highpass.[0-9]*' % (base,))
                    #read the first data file to get the number of channels
                    if len(dataFiles)==0:
                        raise IOError('No datafile found')
                    #get the header size
                    hs = np.fromfile(dataFiles[0],dtype=np.uint32,count=1)
                    data,sr = extraction.readDataFile(dataFiles[0])
                    nchs = data.shape[0]
                    #nchs = sum(descriptor['gr_nr']>0)
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
                    sizes =  [os.stat(f).st_size for f in dataFiles]
                    total_size = ((np.array(sizes)-hs)/2/nchs).sum()
                    alldata = np.memmap('/tmp/%s.all' %(base,),dtype=np.int16,shape=(len(channels),total_size),mode='w+')
                    offset = 0
                    for f in dataFiles:
                        print "Loading data from file %s..." %(f,)
                        sys.stdout.flush()
                        data,sr = extraction.readDataFile(f)

                        #data = np.memmap(f,mode='r',dtype=np.int16,offset=73,shape=((os.stat(f).st_size-73)/2/nchs,nchs))
                        alldata[:,offset:offset+data.shape[1]] = data[channels,:]
                        alldata.flush()
                        offset+=data.shape[1]

                print "Computing inverse covariance matrix..."
                sys.stdout.flush()
                cinv = np.linalg.pinv(np.cov(alldata))
                winlen = alldata.shape[1]
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
            print "Removed %d small templates.." %( q-len(p),)
            if len(spkforms)>0:
                try:
                    dataFile.create_group('spikeFormsLarge')
                    dataFile['spikeFormsLarge']['spikeForms'] = spkforms
                    dataFile['spikeFormsLarge']['p'] = p
                    dataFile.flush()
                except:
                    #already exists
                    pass
            if len(spkforms)>1:
                print "Combining templates..."
                sys.stdout.flush()
                spkforms,p = combineSpikes(spkforms,p,cinv,winlen,maxp=maxp)
                print "Left with %d templates .." %(len(p),)
            if len(spkforms)>0:
                try:
                    dataFile['spikeForms'] = spkforms
                    dataFile['p'] = p
                    dataFile.close()
                except:
                    pass

        else:
            #check for the presence of an SGE_TASK_ID
            tid = None
            nchunks = None
            if 'SGE_TASK_ID' in os.environ and 'SGE_TASK_FIRST' in os.environ and os.environ.get('SGE_TASK_FIRST','') != 'undefined':
                #signifies that we should use split the file
                tfirst = int(os.environ.get('SGE_TASK_FIRST',0))
                tlast = int(os.environ.get('SGE_TASK_LAST',0))
                nchunks = tlast-tfirst+1
                tid = int(os.environ['SGE_TASK_ID'])-1
                print "Analyzing file %s in %d chunks. Analyzing chunk %d...." %(dataFileName,nchunks,tid+1)
                sys.stdout.flush()

            try:
                spikeForms,cinv = learnTemplatesFromFile(dataFileName, group, splitp=splitp,
                                                         outfile=outFileName, chunksize=chunkSize,
                                                         debug=debug,
                                                         nFileChunks=nchunks, fileChunkId=tid,
                                                         redo=redo,iterations=iterations,
                                                        tempPath=tempPath,initFile=initFile,states=states, max_size=maxSize, offset=offset, min_snr=min_snr)
            except IOError:
                sys.stderr.write("Could not read/write to file\n")
                traceback.print_exc(file=sys.stderr)
		sys.stderr.flush()
                sys.exit(99)
    except SystemExit as ee:
        # honour the request to quit by simply re-issuing the call to exit with the correct code
        sys.exit(ee.code)
    except:
        sys.stderr.write("An error occurred\n")
        sys.stderr.flush()
        traceback.print_exc(file=sys.stderr)
        sys.exit(100)
