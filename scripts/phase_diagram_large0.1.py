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

def init_aligned2(sim):
        sim.nx[:,:,:] = 1.0/np.sqrt(2)
        sim.ny[:,:,:] = 0.0
        sim.nz[:,:,:] = 1.0/np.sqrt(2)

K = 1.0
temp = 0.1
simename = "phasediaglarge_alignedlong4_K"+str(round(K,3))+"_kbt"+str(round(temp,5))
sim = chll.ChLLSim(name = simename,ni=512,nj=512,nk=512,kbt=temp,d = 0.01,KK = K,g=0.0,rho = 0.0)
init_aligned(sim)
sim.init()
sim.run(400000,100,save=True)
sim.plot_config()

