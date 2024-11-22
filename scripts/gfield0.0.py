from chll import chll
import numpy as np
import matplotlib.pyplot as plt

def init_aligned2(sim):
    sim.nx[:,:,:] = 1.0/np.sqrt(2.0)
    sim.ny[:,:,:] = 0.0
    sim.nz[:,:,:] = 1.0/np.sqrt(2.)

N = 128
K = 0.0
temp = 0.05
for g in np.arange(0.0,0.251,0.025):
    simename = "gfield_K"+str(round(K,3))+"_kbt"+str(round(temp,5))+"_g"+str(round(g,5))
    sim = chll.ChLLSim(name = simename,ni=N,nj=N,nk=N,kbt=temp,d = 0.01,KK = K,g = g,rho = 0.0)
    init_aligned2(sim)
    sim.init_rand_s()
    sim.run(1000000,100,save=True)
    sim.plot_config()

