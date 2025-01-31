from chll import chll
import numpy as np
import sys

def init_random(sim):
    for i in range(sim.ni):
        for j in range(sim.nj):
            for k in range(sim.nk):
                costh = 2.0*(np.random.rand()-0.5)
                sinth = np.sqrt(1.0-costh*costh)
                phi = np.random.rand()*2.0*np.pi
                cosphi = np.cos(phi)
                sinphi = np.sin(phi)
                sim.nx[i,j,k] = sinth*cosphi
                sim.ny[i,j,k] = sinth*sinphi
                sim.nz[i,j,k] = costh

N = 32
K = float(sys.argv[1])
prevT = 1.5
for i,T in enumerate(np.arange(1.5,0.49,-0.05)):
    simname = "pdanneal4_K"+str(round(K,3))+"_kbt"+str(round(T,5))
    sim = chll.ChLLSim(name = simname,ni=N,nj=N,nk=N,kbt=T,d=0.01,KK=K,g=0.0,rho=0.0)
    if i == 0:
        init_random(sim)
        sim.init_rand_s()
    else:
        old_data = "./data/pdanneal4_K"+str(round(K,3))+"_kbt"+str(round(prevT,5))
        old_data += "/pdanneal4_K"+str(round(K,3))+"_kbt"+str(round(prevT,5))+"_data.npz"
        data = np.load(old_data)
        sim.nz[:,:,:] = data['nz']
        sim.nx[:,:,:] = data['nx']
        sim.ny[:,:,:] = data['ny']
        s = data['s']
        sim.init_with_s(s)
    prevT = T
    sim.run(2000000,100,save=True)
    sim.plot_config()
