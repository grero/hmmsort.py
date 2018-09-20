import hmmsort
import hmmsort.create_spiketrains as create_spiketrains
import matplotlib
import os
import pytest
import urllib2
import tempfile
import hashlib
import scipy.io as mio

def download_file(fname):
    _file = urllib2.urlopen("http://cortex.nus.edu.sg/testdata/array01/channel001/%s" % (fname,))
    with open(fname, "wb") as output:
        output.write(_file.read())

@pytest.mark.order2
def test_pick_lines(qtbot):
    dd = tempfile.mkdtemp()
    cwd = os.getcwd()
    os.chdir(dd)
    #download data
    download_file("hmmsort.mat")
    download_file("spike_templates.hdf5")
    window = create_spiketrains.ViewWidget()
    window.show()
    window.select_waveforms()
    qtbot.addWidget(window)
    #simulate a pick event
    event = matplotlib.backend_bases.PickEvent("A pick", window.figure.canvas, [],window.figure.axes[0].lines[0]) 
    window.pick_event(event)
    event = matplotlib.backend_bases.PickEvent("A pick", window.figure.canvas, [],window.figure.axes[0].lines[1]) 
    window.pick_event(event)
    window.save_spiketrains(notify=False) 
    #check that we got what we expected
    q1 =  mio.loadmat("cell01/spiketrain.mat")
    hh1 = hashlib.sha1(q1["timestamps"].tostring()).hexdigest()
    assert hh1 == '8093ffc2459eb613bcc10f65e34b0c47568e22ae'
    q2 =  mio.loadmat("cell02/spiketrain.mat")
    hh2 = hashlib.sha1(q2["timestamps"].tostring()).hexdigest()
    assert hh2 == '03de5a1a19919c3ede7030d760aebcf671379c7f'
    os.unlink("hmmsort.mat")
    os.unlink("spike_templates.hdf5")
    os.unlink("cell01/spiketrain.mat")
    os.unlink("cell02/spiketrain.mat")
    os.rmdir("cell01")
    os.rmdir("cell02")
    os.chdir(cwd)
    os.rmdir(dd)
