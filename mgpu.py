from mpi4py import MPI
from cupy import cuda
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


mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

gpu_id = mpi_rank % cuda.runtime.getDeviceCount()
print("Rank: ",mpi_rank,"/",mpi_size," gpu_id ",gpu_id)
      
for iK in np.arange(0.0,2.1,0.25)[mpi_rank::2]:
        for temp in np.arange(0.05,1.251,0.05):
                print("Running: K=",iK," kbt=",temp," on ", mpi_rank)


"""
with cuda.Device(gpu_id):
        i = mpi_rank+1
        sim = chll.ChLLSim(name="gmpu_test"+str(i),ni=64,nj=64,nk=64,kbt=0.05,d=0.01,KK=0.1,rho=0.0)
        init_aligned(sim)
        sim.init()
        sim.run(1000,50,save=False)
"""
mpi_comm.Barrier()



