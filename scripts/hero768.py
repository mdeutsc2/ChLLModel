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


def load_checkpoint(sim):
    data = np.load("./data/hero7681/hero7681_data.npz")
    sim.nx[:,:,:] = data['nx']
    sim.ny[:,:,:] = data['ny']
    sim.nz[:,:,:] = data['nz']
    s = data['s']
    return s

i = 2
sim = chll.ChLLSim(name = "hero768"+str(i),
			  ni=768,
			  nj=768,
			  nk=768,
			  kbt=0.25,
			  d = 0.01,
			  KK = 0.5,
			  g = 0.0,
			  rho = 0.0,overwrite=False)

#init_aligned(sim)
#sim.init_rand_s()

s = load_checkpoint(sim)
sim.init_with_s(s)

sim.run(25000,100,save=True)
sim.plot_config()
#sim.output_old()
	
