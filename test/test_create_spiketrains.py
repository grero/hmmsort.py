import hmmsort
import hmmsort.create_spiketrains as create_spiketrains
import matplotlib
import os

def test_pick_lines(qtbot):
    cwd = os.getcwd()
    os.chdir("test/array01/channel001") 
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
    os.chdir(cwd)
