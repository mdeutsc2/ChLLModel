from chll import chll
import numpy as np
import matplotlib.pyplot as plt




N = 32
K = 3.0
temp = 0.05
simname = "test2_chiral_K"+str(round(K,3))+"_kbt"+str(round(temp,5))
sim = chll.ChLLSim(name = simname,ni=N,nj=N,nk=N,kbt=temp,d = 0.01,KK = K,g=0.0,rho = 0.0)
sim.nx[:,:,:] = 1.0
sim.ny[:,:,:] = 0.0
sim.nz[:,:,:] = 0.0
s = np.zeros_like(sim.nx)
s.fill(-1.0)
sim.init_with_s(s)
sim.run(100000,100,save=True)
sim.plot_config()

