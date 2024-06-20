import numpy as np
import warp as wp
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from kernels import *

class ChLLSim:
    def __init__(self,device,ni,nj,nk,kbt,d,KK,rho):
        # parameters
        assert device in ["cpu","cuda:0","cuda:1"]
        self.device = device
        wp.init()
        wp.build.clear_kernel_cache()
        wp.set_device(self.device)
        
        self.ni = ni
        self.nj = nj
        self.nk = nk
        self.kbt = 0.05
        self.d = 0.01
        self.KK = 1.0
        self.rho = 0.0

        # internal parameters
        self.nsub = int((ni*nj*nk)/2.0)
        self.n3 = ni*nj*nk
        print(self.nsub)
        # arrays
        self.nx = wp.zeros((self.ni,self.nj,self.nk),dtype=wp.float64)
        self.ny = wp.zeros((self.ni,self.nj,self.nk),dtype=wp.float64)
        self.nz = wp.zeros((self.ni,self.nj,self.nk),dtype=wp.float64)
        self.s = wp.zeros((self.ni,self.nj,self.nk),dtype=wp.int32)
        self.naccept = wp.zeros((self.ni,self.nj,self.nk),dtype=wp.int32)
        self.nflip = wp.zeros((self.ni,self.nj,self.nk),dtype=wp.int32)
        self.dope = wp.zeros((self.ni,self.nj,self.nk),dtype=wp.int32)
        self.sl1 = wp.zeros(self.nsub,dtype=wp.int32)
        self.sl2 = wp.zeros((self.nsub),dtype=wp.int32)
        self.e = wp.zeros((self.ni,self.nj,self.nk),dtype=wp.float64)

    def setup(self):
        nxi = np.zeros((self.ni,self.nj,self.nk),dtype=np.float64)
        nyi = np.zeros((self.ni,self.nj,self.nk),dtype=np.float64)
        nzi = np.zeros((self.ni,self.nj,self.nk),dtype=np.float64)
        si  = np.zeros((self.ni,self.nj,self.nk),dtype=np.int32)
        sl1i = np.zeros(self.nsub,dtype=wp.int32)
        sl2i = np.zeros(self.nsub,dtype=wp.int32)
        dopei = np.zeros((self.ni,self.nj,self.nk),dtype=np.int32)
        for i in range(self.ni):
            for j in range(self.nj):
                for k in range(self.nk):
                    si[i,j,k] = 1
                    if (np.random.rand() <= 0.5):
                        si[i,j,k] = -1
                    dopei[i,j,k] = 0
                    costh = 2.0*(np.random.rand()-0.5)
                    sinth = np.sqrt(1.0-costh*costh)
                    phi = np.random.rand()*np.pi*2.0
                    cosphi = np.cos(phi)
                    sinphi = np.sin(phi)
                    nxi[i,j,k] = sinth*cosphi
                    nyi[i,j,k] = sinth*sinphi
                    nzi[i,j,k] = costh
        self.nx = wp.from_numpy(nxi,dtype=wp.float64)
        self.ny = wp.from_numpy(nyi,dtype=wp.float64)
        self.nz = wp.from_numpy(nzi,dtype=wp.float64)
        self.s = wp.from_numpy(si,dtype=wp.int32)
        self.dope = wp.from_numpy(dopei,dtype=wp.int32)
        nsub1 = 0
        nsub2 = 0
        for i in range(self.ni):
            for j in range(self.nj):
                for k in range(self.nk):
                    index = i+j*self.ni + k*self.ni*self.nj
                    if ((i+j+k)%2 != 0):
                        sl1i[nsub1] = index
                        nsub1 += 1
                        #print(i,j,k,index)
                    else:
                        sl2i[nsub2] = index
                        nsub2 += 1
                        #print(i,j,k,index)
        #for i in range(nsub1):
        #    print(sl1i[i],sl2i[i])
        print(sl1i.shape,sl2i.shape)
        self.sl1 = wp.from_numpy(sl1i,dtype=wp.int32)
        self.sl2 = wp.from_numpy(sl2i,dtype=wp.int32)
        print("SETUP DONE")

    def run(self,num_steps):
        shape = (self.ni,self.nj,self.nk)
        pbar = tqdm(total=num_steps)
        for istep in range(num_steps):
            #print(istep)
            wp.launch(reset_counters,dim=shape, inputs = [self.naccept,self.nflip])
            wp.launch(evolve, dim=self.nsub, inputs = [self.sl1,
                                                   self.nx,self.ny,self.nz,
                                                   self.s,self.naccept,self.nflip,
                                                   self.kbt,self.KK,self.d,
                                                   self.ni,self.nj,self.nk,np.random.randint(100000000)])
            wp.launch(evolve, dim=self.nsub, inputs = [self.sl2,
                                                       self.nx,self.ny,self.nz,
                                                       self.s,self.naccept,self.nflip,
                                                       self.kbt,self.KK,self.d,
                                                       self.ni,self.nj,self.nk,np.random.randint(1000000000)])
            paccept = np.sum(np.float64(self.naccept.numpy()))/self.n3
            if (paccept < 0.4):
                self.d = self.d*0.995
            elif (paccept > 0.6):
                self.d = self.d/0.995
            if (istep % 50) == 0:
                wp.launch(etot, dim=shape, inputs = [self.e,
                                                     self.nx,self.ny,self.nz,
                                                     self.s,
                                                     self.KK,
                                                     self.ni,
                                                     self.nj,
                                                     self.nk])
                total_energy = np.sum(self.e.numpy())/self.n3
                e_excess = np.sum(np.float64(self.s.numpy()))/self.n3
                faccept = np.sum(np.float64(self.nflip.numpy()))/self.n3
                if istep == 0:
                    pbar.write(f'istep \t E_total    Ent. Excess     Paccept     Faccept      D')
                pbar.write(f'{istep:06} | {round(total_energy,7)} | {round(e_excess,7)} | {round(paccept,7)} | {round(faccept,7)} | {round(self.d,7)}')
                pbar.update(50)

    def output(self):
        nxo = self.nx.numpy()
        nyo = self.ny.numpy()
        nzo = self.nz.numpy()
        so = self.s.numpy()
        fa = open("LLMC_py-configa.dat","w")
        fb = open("LLMC_py-configb.dat","w")
        scale = 0.4
        for i in range(self.ni):
            j = int(self.nj/2)
            for k in range(self.nk):
                x1 = i-scale*nxo[i,j,k]
                x2 = i+scale*nxo[i,j,k]
                z1 = k-scale*nzo[i,j,k]
                z2 = k+scale*nzo[i,j,k]
                if so[i,j,k] == 1:
                    fa.write(str(x1)+" "+str(z1)+"\n")
                    fa.write(str(x2)+" "+str(z2)+"\n")
                    fa.write("\n")
                else:
                    fb.write(str(x1)+" "+str(z1)+"\n")
                    fb.write(str(x2)+" "+str(z2)+"\n")
                    fb.write("\n")
        fa.close()
        fb.close()
            
        

if __name__ == "__main__":
    simulation = ChLLSim("cpu",32,32,32,0.05,0.01,1.0,0.0)
    simulation.setup()
    num_steps = 1000
    simulation.run(num_steps)
    simulation.output()
