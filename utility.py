import numpy as np
import numba
import sys
import tempfile
import time
import pylab as plt

@numba.autojit
def forward(data, W, g, spklength, 
            winlength, c, q, N, p,f):
    tiny = np.exp(-700)
    P = p.sum()
    for t in xrange(1, winlength):
        f[:] = W - data[t]
        #f = np.exp(-0.5*(y*np.dot(c, y)).sum(0)) + tiny
        f[:] = np.exp(-0.5*(f*f*c)) + tiny
        g[:, t] = g[q, t - 1]
        gg = g[1:2 + (N - 1)*(spklength - 1):(spklength - 1), t]
        a = 0.
        for k in xrange(gg.shape[0]):
            a += gg[k] 
        g[0, t] = a + g[0, t] - g[0, t - 1]*P
        g[1:2 + (N - 1) * (spklength - 1):(spklength - 1), t] = g[0, t - 1] * p
        g[:, t] = g[:, t]*f[:] + tiny
        a = 0.
        for k in xrange(g.shape[0]):
            a += g[k, t]
        g[:, t] = g[:, t] / (a+tiny)

def forward2(data, W, g, spklength, 
            winlength, c, q, N, p,f):
    tiny = np.exp(-700)
    for t in xrange(1, winlength):
        f[:] = W-data[t]
        #f = np.exp(-0.5*(y*np.dot(c, y)).sum(0)) + tiny
        f[:] = np.exp(-0.5*(f*f*c)) + tiny
        g[:, t] = g[q, t - 1]
        g[0, t] = (g[1:2 + (N - 1)*(spklength - 1):(spklength - 1), t].sum() + 
                    g[0, t] - g[0, t - 1]*p.sum())
        g[1:2 + (N - 1) * (spklength - 1):(spklength - 1), t] = g[0, t - 1] * p
        g[:, t] = g[:, t]*f[:] + tiny
        g[:, t] = g[:, t] / (g[:, t].sum()+tiny)

@numba.autojit
def backward(data, W, g, spklength, 
            winlength, c, q, N, p, b,f):
    tiny = np.exp(-700)
    P = p.sum()
    for t in xrange(winlength - 2, -1, -1):
        f[:] = W - data[t + 1]
        f[:] = np.exp(-0.5*(f*f*c)) + tiny
        b = b*f + tiny
        b[q] = b
        b[0] = (1 - P)*b[-1]
        gg = b[:(N-1)*(spklength - 1) + 1:(spklength - 1)]
        a = 0.
        for k in xrange(gg.shape[0]):
            a += p[k]*gg[k]
        b[0] += a
        b[(spklength - 1):1 + (N - 1)*(spklength - 1):(spklength - 1)] = b[-1]
        a = 0.
        for k in xrange(b.shape[0]):
            a += b[k]
        b = b / (a + tiny)
        g[:, t] = g[:, t] * b + tiny

def learn(data,spkform=None,iterations=10,cinv=None,p=None,splitp=None,dosplit=True,states=60,
                chunksize=10000,debug=False,levels=4,tempPath=None,**kwargs):
    """
    This function runs the baum-welch algorithm on the specified data, learning spiking templates. The input data should have dimensions of datapts X channels. This code runs on the data in chunks, offloading data to disk when not in use. This allows it to analyse arbitrarily long sequences of data.
    """
    prestates = states/3
    poststates = states/3
    if spkform == None:
        neurons = 8
        amp = np.random.random(size=(neurons,levels))+0.5
        #amp = amp/amp.max(1)[:, None]
        spkform = np.concatenate((np.zeros((levels, prestates)),
                                  np.sin(np.linspace(0,3*np.pi,prestates))[None,:].repeat(levels,0),
                                  np.zeros((levels,poststates))),axis=1)[None,:,:]*amp[:,:,None]*(np.median(np.abs(data),axis=0)/0.6745)[None,:,None]
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
        if data.shape[1]>1:
            c = np.linalg.pinv(np.cov(data.T))
        else:
            #single dimension
            c = 1.0/data.var(0)
    else:
        c = cinv
    if p == None:
        p = 1.e-8*np.ones((N,))
    else:
        if len(p) < len(spkform):
            p = p.repeat(N,0)

    if splitp == None:
        splitp = .5/40000

    p_reset = p
    
    winlength,dim = data.shape
    spklength = spkform.shape[-1]
    W = spkform[:,:,1:].transpose((1,0,2)).reshape(dim,N*(spklength-1))
    W = np.concatenate((np.zeros((dim,1)),W),axis=1)
    W = W.flatten()
    #this is an index vector
    q = np.concatenate(([N*(spklength-1)],np.arange(N*(spklength-1))),axis=0)
    tiny = np.exp(-700)
    nchunks = int(np.ceil(1.0*data.shape[0]/chunksize))
    chunks = np.append(np.arange(0,data.shape[0],chunksize),[data.shape[0]])
    chunksizes = np.diff(chunks).astype(np.int)
    nchunks = len(chunksizes)
    dtf = 0
    dtb = 0
    for bw in xrange(iterations):
        print "Iteration %d of %d" % (bw + 1, 
                                     iterations)
        sys.stdout.flush()
        W = W.flatten()
        fid = tempfile.TemporaryFile(dir=tempPath)
        p = p_reset
        g = np.zeros((N*(spklength-1)+1,chunksize))
        b = np.zeros((g.shape[0],))
        f = np.zeros(W.shape)
        g[0,0] = 1
        b[0] = 1
        #compute probabilities  
        #note that we are looping over number of states here; the most we are
        #creating is 2Xdata.nbytes
        #forward
        print "\tRunning forward algorithm..."
        sys.stdout.flush()
        for i in xrange(nchunks):
            print "\t\tAnalyzing chunk %d of %d" % (i+1, nchunks) 
            t1 = time.time()
            forward(data[chunks[i]:chunks[i+1]],W,g,spklength,chunksizes[i],cinv,
                   q,N,p,f)
            t2 = time.time()
            #compute average duration
            dtf = (dtf*i + (t2 - t1))/(i + 1)
            print "That took %.2f seconds. ETTG: %.2f" %(t2-t1,
                                                         dtf*(nchunks-(i + 1)))
            g = g[:, :chunksizes[i]]
            kk = 0
            while kk < 100:
                #try saving the file
                try:
                    g.tofile(fid)
                except ValueError:
                    """
                    for some reason, sometimes we get a value error here. If that
                    happens, just report the exception and let sge know an error
                    occured
                    """
                    kk += 1
                    time.sleep(10)
                else:
                    #if no exception occurred, we mangaed to save the file, so
                    #break out of the loop
                    break
                    #traceback.print_exc(file=sys.stdout)
                    #if __name__ == '__main__':
                        #only exit if we are running this as a script
                   #     sys.exit(99)
            if kk == 100:
                #if we reach here it means that we could not save the file
                if __name__ == '__main__':
                    print """Could not save temporary file, most likely because of
                    lack of disk space"""
                    sys.exit(99)
                else:
                    #raise an IO error
                    raise IOError('Could not save temporary file')
            g[:, 0] = g[:, -1]

        #backward
        print "\tRunning backward algorithm..."
        sys.stdout.flush()
        G = np.zeros((g.shape[0], )) 
        for i in xrange(nchunks - 1, -1, -1):
            print "\t\tAnalyzing chunk %d of %d" % (i + 1, nchunks) 
            t1 = time.time()
            backward(data[chunks[i]:chunks[i+1]], W, g, spklength,
                    chunksizes[i], cinv, q, N, p, b, f)
            t2 = time.time()
            dtb = (dtb*i + (t2 - t1))/(i + 1)
            print "That took %.2f seconds. ETTG: %.2f" %(t2-t1,
                                                         dtb*(i))
            g = g / (g.sum(0) + tiny)
            G += g.sum(1)
            #rewind file
            fid.seek(-(N*(spklength - 1) + 1)*chunksizes[i]*g[0, 0].nbytes, 1)
            g.tofile(fid)

        #TODO: This stop could be quite memory intensive
        W = np.zeros(W.shape)
        fid.seek(0)
        t1 = time.time()
        for i in xrange(nchunks):
            g = np.fromfile(fid,dtype=np.float,
                            count=(N*(spklength-1)+1)*
                            chunksizes[i]).reshape(N*(spklength-1)+1, 
                            chunksizes[i])
            W += np.dot(data[chunks[i]:chunks[i+1], :].T, g.T).flatten()
        t2 = time.time()
        print "Constructing W tok %.2f seconds" % (t2 - t1, )
        W = W / G[None,:]
        W[:,0] = 0
        p = np.zeros((N, ))
        fid.seek(0)
        D = np.memmap(tempfile.TemporaryFile(),dtype=np.float,shape=data.shape,mode='w+')
        t1 = time.time()
        for i in xrange(nchunks):
            g = np.fromfile(fid, dtype=np.float,
                            count=(N*(spklength - 1) + 1)*chunksizes[i])
            g = g.reshape(N*(spklength-1) + 1, chunksizes[i])
            p += g[1::(spklength - 1),:].sum(1)

            D[chunks[i]:chunks[i+1], :] = (W[:, :, None]*g[None, :, :]).sum(1).T
        t2 = time.time()
        print "Constructing D tok %.2f seconds" % (t2 - t1, )
        fid.close() 
        p = p / winlength
        if data.shape[1] > 1:
            cinv = np.linalg.pinv(np.cov((data-D).T))
        else:
            cinv = 1.0/(data-D).var(0)
        #we don't need D any more
        del D
        maxamp = np.zeros((len(spkform),))
        for j in xrange(len(spkform)):
            spkform[j] = np.concatenate((W[:,0][:,None],
                                         W[:,j*(spklength-1)+1:(j+1)*(spklength-1)+1]),
                                        axis=1)
            maxamp[j] = (spkform[j]*(np.dot(cinv, spkform[j]))).sum(0).max(0)

        nspikes = p*winlength

        print "\tSpikes found per template: " 
        print ' '.join((map(lambda s: '%.2f' %s,nspikes)))
        sys.stdout.flush()
        if dosplit:
            #remove templates with too low firing rate and replace with a new
            #guess
            print "\tTrying to split clusters..."
            for i in xrange(len(spkform)):
                if p[i] < splitp:
                    #remove template i and replace with template j
                    try:
                        j = np.where((p>=np.median(p))*
                                     (p > splitp*4)*(maxamp>10))[0]
                        j = j[np.random.random_integers(0,len(j)-1, 
                                                        size=(1, 1))]
                        W[:,i*(spklength-1)+1:(i+1)*(spklength-1)] = W[:,j*(spklength - 1) +
                                                                       1:(j + 1)*(spklength - 1)]*.98
                        p[i] = p[j] / 2
                        p[j] =p[j] / 2
                        print "\t\tWaveformsupdate: %d <- %d" % (i, j)
                        sys.stdout.flush()
                        break
                    except:
                        print "\t\tClustersplitting failed"
                        sys.stdout.flush()

        if debug:
            for j in xrange(len(spkform)):
                plt.subplot(len(spkform), 1,j+1)
                plt.gca().clear()
                plt.plot(x.T, spkform[j].T)
            plt.draw()    
        

    #del g,fit 
    del g 
    for j in xrange(len(spkform)):
        spkform[j,:,:-1] = np.concatenate((W[:,1][:,None], 
                                           W[:,j*(spklength-1)+1:(j+1)*(spklength-1)])
                                          ,axis=1)
    
    return data,spkform,p,cinv
        
def prepare_neurons(neurons, levels, states, data):
    prestates = states/3
    poststates = states/3
    amp = np.random.random(size=(neurons, levels))+0.5
#amp = amp/amp.max(1)[:, None]
    spkform = np.concatenate((np.zeros((levels, prestates)),
                              np.sin(np.linspace(0,3*np.pi,prestates))[None,:].repeat(levels,0),
                              np.zeros((levels,poststates))),axis=1)[None,:,:]*amp[:,:,None]*(np.median(np.abs(data),axis=0)/0.6745)[None,:,None]

    return spkform

def prepare_test(N=8,states=60,dim=1):
    N = 8
    dim = 1
    spklength = 60
    spkform = prepare_neurons(N,1,60,np.random.random(size=(100000,1)))
    data = np.random.random(size=(100000,1))
    chunksize = data.shape[0]
    cinv = 1.0/data.var(0)
    p = 1.e-8*np.ones((N,))
    q = np.concatenate(([N*(spklength-1)],np.arange(N*(spklength-1))),axis=0)
    W = spkform[:,:,1:].transpose((1,0,2)).reshape(dim,N*(spklength-1))
    W = np.concatenate((np.zeros((dim,1)),W),axis=1)
    W = W.flatten()
    f = np.zeros(W.shape)
    g = np.zeros((N*(spklength-1)+1,chunksize))
    b = np.zeros((g.shape[0],))
    
    return N,W,g,q,p,cinv,data.flatten(),spklength,f,b
