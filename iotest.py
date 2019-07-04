#!/usr/bin/env python
import numpy as np
import blosc
import tempfile
import os

tempPath = "/hpctmp2/lsihr/tmp/" 
if __name__ == "__main__":
	for i in xrange(10):
		x = np.random.rand(0.125*1024**3)
		gp = blosc.pack_array(x)
		fid = tempfile.NamedTemporaryFile(dir=tempPath,delete=False,prefix=str(i))
		
		fid.write(gp)
		fid.flush()
		fid.close()
		os.unlink(fid.name)
