from chll import chll
import numpy as np
import matplotlib.pyplot as plt


N = 128
K = 3.0
temp = 0.05
simname = "test2_chiral_K"+str(round(K,3))+"_kbt"+str(round(temp,5))
sim = chll.ChLLSim(name = simname,ni=N,nj=N,nk=N,kbt=temp,d = 0.01,KK = K,g=0.0,rho = 0.0)
data = np.load("./data/pdanneal/pdanneal_K3.0_kbt0.05/pdanneal_K3.0_kbt0.05_data.npz")
sim.nx[:,:,:] = data['nx']
sim.ny[:,:,:] = data['ny']
sim.nz[:,:,:] = data['nz']
s = data['s']
print("Loaded S: ",np.mean(s))
s.fill(1.0)
print("New S: ",np.mean(s))
sim.init_with_s(s)
sim.plot_config()
sim.run(100,1,save=True)
#print("After sim: ",np.mean(sim.s))
#sim.plot_config()

