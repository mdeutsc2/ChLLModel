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

def load_checkpoint(sim,K,temp):
        data = np.load("./data/pdanneal_K"+str(round(K,3))+"_kbt"+str(round(temp,5))+"/pdanneal_K"+str(round(K,3))+"_kbt"+str(round(temp,5))+"_data.npz")
        sim.nx[:,:,:] = data['nx']
        sim.ny[:,:,:] = data['ny']
        sim.nz[:,:,:] = data['nz']
        s = data['s']
        return s

N = 128
K = 1.75
prevT = 0.25
for i,temp in enumerate(np.arange(0.2,0.04,-0.05)):
    simname = "pdanneal_K"+str(round(K,3))+"_kbt"+str(round(temp,5))
    sim = chll.ChLLSim(name = simname,ni=N,nj=N,nk=N,kbt=temp,d = 0.01,KK = K,g=0.0,rho = 0.0)
    s = load_checkpoint(sim,K,prevT)
    sim.init_with_s(s)
    prevT = temp
    sim.run(2000000,100,save=True)
    sim.plot_config()

