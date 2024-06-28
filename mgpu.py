from chll import chll
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, current_process, Queue

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

def sim_wrapper(i):
	gpu_id = queue.get()
	try:
		sim = chll.ChllSim(name="gmpu_test"+str(i),ni=16,nj=16,nk=16,kbt=0.05,d=0.01,KK=0.1,rho=0.0)
		init_aligned(sim)
		sim.init()
		sim.run(1000,save=False)
	finally:
		queue.put(gpu_id)

queue = Queue()
for gpu_ids in range(2):
	queue.put(gpuids)

pool = Pool(processes = 2)
for _ in pool.imap_unordered(sim_wrapper,range(2)):
	pass
pool.close()
pool.join()
	
"""
for temp in np.arange(0.05,1.51,0.05):
	for K in np.arange(0.0,1.1,0.1):
		simename = "phasediag_alignedlong2_K"+str(round(K,3))+"_kbt"+str(round(temp,5))
		sim = chll.ChLLSim(name = simename,ni=64,nj=64,nk=64,kbt=temp,d = 0.01,KK = K,rho = 0.0)
		init_aligned(sim)
		sim.init()
		sim.run(200000,50,save=True)
		sim.plot_config()

for temp in np.arange(0.05,1.51,0.05):
	for K in np.arange(0.0,0.11,0.01):
		simename = "phasediag_alignedlong2_K"+str(round(K,3))+"_kbt"+str(round(temp,5))
		sim = chll.ChLLSim(name = simename,ni=64,nj=64,nk=64,kbt=temp,d = 0.01,KK = K,rho = 0.0)
		init_aligned(sim)
		sim.init()
		sim.run(200000,50,save=True)
		sim.plot_config()

for temp in np.arange(0.05,1.51,0.05):
	for K in np.arange(0.0,0.011,0.001):
		simename = "phasediag_alignedlong2_K"+str(round(K,3))+"_kbt"+str(round(temp,5))
		sim = chll.ChLLSim(name = simename,ni=64,nj=64,nk=64,kbt=temp,d = 0.01,KK = K,rho = 0.0)
		init_aligned(sim)
		sim.init()
		sim.run(200000,50,save=True)
		sim.plot_config()
"""
