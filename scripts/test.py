from chll import chll
import numpy as np
import matplotlib.pyplot as plt

def init_aligned(sim):
    sim.nx[:,:,:] = 1.0
    sim.ny[:,:,:] = 0.0
    sim.nz[:,:,:] = 0.0


i = 1
sim = chll.ChLLSim(name = "test"+str(i),
			  ni=128,
			  nj=128,
			  nk=128,
			  kbt=0.05,
			  d = 0.01,
			  KK = 1.0,
			  g = 0.0,
			  rho = 0.0,overwrite=True)

init_aligned(sim)
sim.init_rand_s()
sim.run(1000,100,save=False)
#sim.plot_config()
#sim.output_old()
	
