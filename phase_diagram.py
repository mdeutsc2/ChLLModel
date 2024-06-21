from chll import chll
import numpy as np
import matplotlib.pyplot as plt

def init_random(sim):
	for i in range(sim.ni):
		for j in range(sim.nj):
			for k in range(sim.nk):
				costh = 2.0*(np.random.rand()-0.5)
				sinth = np.sqrt(1.0-costh*costh)
				phi = np.random.rand()*2*np.pi
				cosphi = np.cos(phi)
				sinphi = np.sin(phi)
				sim.nx[i,j,k] = sinth*cosphi
				sim.ny[i,j,k] = sinth*sinphi
				sim.nz[i,j,k] = costh

def init_aligned(sim):
	sim.nx[:,:,:] = 1.0
	sim.ny[:,:,:] = 0.0
	sim.nz[:,:,:] = 0.0


for i in range(5):
	sim = chll.ChLLSim(name = "test"+str(i),ni=64,nj=64,nk=64,kbt=0.05,d = 0.01,KK = 1.0,rho = 0.0)

init_random(sim)
sim.init()
sim.run(1000,50,save=True)
#sim.output_old()
	
