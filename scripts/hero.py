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
        sim.nx[:,:,:] = 1.0/np.sqrt(2)
        sim.ny[:,:,:] = 0.0
        sim.nz[:,:,:] = 1.0/np.sqrt(2)


def load_checkpoint(sim,i):
    data = np.load("./data/hero480"+str(i)+"/hero480"+str(i)+"_data.npz")
    sim.nx[:,:,:] = data['nx']
    sim.ny[:,:,:] = data['ny']
    sim.nz[:,:,:] = data['nz']
    s = data['s']
    return s

for i in range(20,21): #20
    sim = chll.ChLLSim(name = "hero480"+str(i),
                          ni=480,
                          nj=480,
                          nk=480,
                          kbt=0.05,
                          d = 0.01,
                          KK = 0.75,
                          g = 0.0,
                          rho = 0.0,overwrite=False)
    if i == 1:
        init_aligned(sim)
        sim.init_rand_s()
    else:
        s = load_checkpoint(sim,i-1)
        sim.init_with_s(s)

    sim.run(100000,100,save=True)
    sim.plot_config()	

