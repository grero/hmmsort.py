import hmmsort
import hmmsort.hmm_learn
import h5py
import os

#download link https://cortex.nus.edu.sg:6949/sharing/cdrEAsNik
#https://cortex.nus.edu.sg:6949/sharing/Q4YXxbofM
def test_spikeforms():
    pwd = os.getcwd()
    os.chdir("test")
    spikeForms, cinv = hmmsort.hmm_learn.learnTemplatesFromFile("highpass.mat", 1,chunksize=80000, version=3, debug=True, iterations =3, states=60,max_size=800000, min_snr=4.0,maxp=12.0)

    ff = h5py.File("spike_templates_master.hdf5","r")
    mSpikeForms = ff["spikeForms"]
    sSpikeForms = spikeForms["second_learning"]["spikeForms"]
    assert mSpikeForms.shape == sSpikeForms.shape
    d = sSpikeForms - mSpikeForms
    d2 = d*d
    assert d2.sum() < 1e-3
    os.chdir(pwd)

