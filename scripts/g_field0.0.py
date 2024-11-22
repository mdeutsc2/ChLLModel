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


K = 0.0
for temp in np.arange(0.05,0.51,0.05):
        for g in np.arange(0.0,0.251,0.05):
                simename = "gfield2_K"+str(round(K,3))+"_kbt"+str(round(temp,5))+"_g"+str(round(g,5))
                sim = chll.ChLLSim(name = simename,ni=64,nj=64,nk=64,kbt=temp,d = 0.01,KK = K,g = g,rho = 0.0)
                init_aligned(sim)
                sim.init()
                sim.run(1000000,100,save=True)
                sim.plot_config()

