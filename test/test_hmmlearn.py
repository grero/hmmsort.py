import hmmsort
import hmmsort.hmm_learn
import h5py
import os
import numpy as np

#download link https://cortex.nus.edu.sg:6949/sharing/cdrEAsNik
#https://cortex.nus.edu.sg:6949/sharing/Q4YXxbofM
def test_spikeforms():
    pwd = os.getcwd()
    os.chdir("test")
    spikeForms, cinv = hmmsort.hmm_learn.learnTemplatesFromFile("highpass.mat", 1,chunksize=80000, version=3, debug=True, iterations =3, states=60,max_size=800000, min_snr=4.0,maxp=12.0)

    ff = h5py.File("spike_templates_master.hdf5","r")
    mSpikeForms = ff["spikeForms"][:]
    sSpikeForms = spikeForms["second_learning"]["spikeForms"]
    assert mSpikeForms.shape == sSpikeForms.shape
    #figure out the relative shift
    min_pts_m = mSpikeForms[:].argmin(2).flatten()
    min_pts_s = sSpikeForms.argmin(2).flatten()
    nstates = mSpikeForms.shape[-1]
    ss = np.zeros((mSpikeForms.shape[0],))
    for i in xrange(len(ss)):
        #find the smallest shift
        ii = np.argmin([np.abs(sSpikeForms[i,0,min_pts_s[i]] - mSpikeForms[j,0,min_pts_m[j]]) for j in xrange(mSpikeForms.shape[0])])
        ds = min_pts_s[i] - min_pts_m[ii]
        d = sSpikeForms[i,0, ds:] - mSpikeForms[ii, 0, :nstates-ds]
        d2 = d*d
        ee = (sSpikeForms[i,0,:]**2).sum()
        ss[i] = d2.sum()/ee
    #accept the result if the deviation is less than 5% of the original spike template energy"
    assert (ss < 0.05).all() 
    os.chdir(pwd)

