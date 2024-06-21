from chll import ChLLSim
import numpy as np

sim = ChLLSim(ni=32,
			  nj=32,
			  nk=32,
			  kbt=0.05,
			  d = 0.01,
			  KK = 1.0,
			  rho = 0.0)
sim.init()
print("init done")
sim.run(1000,100)
sim.output_old()
sim.save_vtk("test.vti")
	
