import os,time
import numpy as np
import taichi as ti
from math import pi

vec2 = ti.math.vec2
vec3 = ti.math.vec3

@ti.func
def idx2ijk(index:int,ni:int,nj:int,nk:int):
        i = (index)%ni
        j = (index)%((ni*nj)/ni)
        k = index/(ni*nj)
        return int(i),int(j),int(k)

@ti.func
def energy(nx:ti.types.ndarray(),
           ny:ti.types.ndarray(),
           nz:ti.types.ndarray(),
           nnx:float,
           nny:float,
           nnz:float,
           snew:int,
           sold:ti.types.ndarray(),
           g:float,
           KK:float,
           i:int,j:int,k:int,
           ni:int,nj:int,nk:int) -> float:
        energy = 0.0
        #ip1
        ip1 = i + 1
        if (ip1 <= ni-1):
                dott = nnx*nx[ip1,j,k] + nny*ny[ip1,j,k] + nnz*nz[ip1,j,k]
                crossx = nny*nz[ip1,j,k] - nnz*ny[ip1,j,k]
                sfac = 0.5*(snew + sold[ip1,j,k])
                energy += (1.0 - dott*dott) - KK*dott*crossx*sfac
        #im1
        im1 = i - 1
        if (im1 >= 0):
                dott = nnx*nx[im1,j,k] + nny*ny[im1,j,k] + nnz*nz[im1,j,k]
                crossx = nny*nz[im1,j,k] - nnz*ny[im1,j,k]
                sfac = 0.5*(snew+sold[im1,j,k])
                energy += (1.0 - dott*dott) - KK*dott*crossx*sfac
        # jp1
        jp1 = j + 1
        if (jp1 <= nj-1):
                dott = nnx*nx[i,jp1,k] + nny*ny[i,jp1,k] + nnz*nz[i,jp1,k]
                crossy = nnz*nx[i,jp1,k] - nnx*nz[i,jp1,k]
                sfac = 0.5*(snew+sold[i,jp1,k])
                energy += (1.0-dott*dott)-KK*dott*crossy*sfac
		
	# jm1
        jm1 = j - 1
        if (jm1 >= 0):
                dott = nnx*nx[i,jm1,k] + nny*ny[i,jm1,k] + nnz*nz[i,jm1,k]
                crossy = nnz*nx[i,jm1,k] - nnx*nz[i,jm1,k]
                sfac = 0.5*(snew+sold[i,jm1,k])
                energy += (1.0-dott*dott)+KK*dott*crossy*sfac
		
	# kp1
        kp1 = k + 1
        if (kp1 <= nk-1):
                dott = nnx*nx[i,j,kp1] + nny*ny[i,j,kp1] + nnz*nz[i,j,kp1]
                crossz = nnx*ny[i,j,kp1] - nny*nx[i,j,kp1]
                sfac = 0.5*(snew+sold[i,j,kp1])
                energy += (1.0-dott*dott)-KK*dott*crossz*sfac
	# km1
        km1 = k - 1
        if (km1 >= 0):
                dott = nnx*nx[i,j,km1] + nny*ny[i,j,km1] + nnz*nz[i,j,km1]
                crossz = nnx*ny[i,j,km1] - nny*nx[i,j,km1]
                sfac = 0.5*(snew+sold[i,j,km1])
                energy += (1.0-dott*dott)+KK*dott*crossz*sfac
        energy -= g * snew
        return energy

@ti.kernel
def etot(nx:ti.types.ndarray(),
         ny:ti.types.ndarray(),
         nz:ti.types.ndarray(),
         s:ti.types.ndarray(),
         g:float,KK:float) -> float:
        e = 0.0
        for i,j,k in ti.ndrange(nx.shape[0],nx.shape[1],nx.shape[2]):
                ip1 = i%nx.shape[0]
                jp1 = j%nx.shape[1]
                kp1 = k + 1
                ddot = nx[i,j,k] * nx[ip1,j,k] + ny[i,j,k]*ny[ip1,j,k] + nx[i,j,k]*nz[ip1,j,k]
                crossx = ny[i,j,k]*nz[ip1,j,k]-nz[i,j,k]*ny[ip1,j,k]
                sfac = 0.5*(s[i,j,k]+s[ip1,j,k])
                e += (1.0 - ddot*ddot) - KK*ddot*crossx*sfac

                ddot = nx[i,j,k]*nx[i,jp1,k] + ny[i,j,k]*ny[i,jp1,k] + nz[i,j,k]*nz[i,jp1,k]
                crossy = nx[i,j,k]*nx[i,jp1,k] - nx[i,j,k]*nz[i,jp1,k]
                sfac = 0.5 * (s[i,j,k]+s[i,jp1,k])
                e += (1.0 - ddot*ddot) - KK*ddot*crossy*sfac

                if (kp1 <= nx.shape[2] - 1):
                        ddot = nx[i,j,k]*nx[i,j,kp1] + ny[i,j,k]*ny[i,j,kp1] + nz[i,j,k]*nz[i,j,kp1]
                        crossz = nx[i,j,k]*ny[i,j,kp1] - ny[i,j,k]*nx[i,j,kp1]
                        sfac = 0.5*(s[i,j,k]+s[i,j,kp1])
                        e += (1.0 - ddot*ddot) - KK*ddot*crossz*sfac
                e -= g * s[i,j,k]
        return e / float(nx.shape[0]*nx.shape[1]*nx.shape[2])
        
@ti.kernel
def evolve(sl:ti.types.ndarray(),
           nx:ti.types.ndarray(),
           ny:ti.types.ndarray(),
           nz:ti.types.ndarray(),
           s:ti.types.ndarray(),
           g:float,
           KK:float,
           d:float,
           kbt:float,
           nsub:int,
           ni:int,
           nj:int,
           nk:int) -> vec2:
        naccept = 0
        nflip = 0
        for itry in ti.ndrange((0,nsub)):
                i,j,k = idx2ijk(sl[itry],ni,nj,nk)
                ip1 = i + 1
                im1 = i - 1
                jp1 = j + 1
                jm1 = j - 1
                kp1 = k + 1
                km1 = k - 1
                nnx = nx[i,j,k]
                nny = ny[i,j,k]
                nnz = nz[i,j,k]
                eold = energy(nx,ny,nz,nnx,nny,nnz,s[i,j,k],s,g,KK,i,j,k,ni,nj,nk)
                if (ti.abs(nnz)>0.999):
                        phi = ti.randn()*2.0*pi
                        xxnew = nnx+d*ti.cos(phi)
                        yynew = nny+d*ti.sin(phi)
                        zznew = nnz
                        rsq = ti.sqrt(xxnew*xxnew + yynew*yynew + zznew*zznew)
                        nnx = xxnew/rsq
                        nny = yynew/rsq
                        nnz = zznew/rsq
                else:
                        ux = -nny
                        uy = nnx
                        uz = 0.0
                        rsq = ti.sqrt(ux*ux + uy*uy)
                        ux = ux/rsq
                        uy = uy/rsq
                        uz = uz/rsq
                        vx = -nnx*nnx
                        vy = -nnz*nny
                        vz = nnx*nnx+nny*nny
                        rsq = ti.sqrt(vx*vx+vy*vy+vz*vz)
                        vx = vx/rsq
                        vy = vy/rsq
                        vz = vz/rsq
                        phi = ti.randn()*2.0*pi
                        dcosphi = d*ti.cos(phi)
                        dsinphi = d*ti.sin(phi)
                        nnx = nnx + dcosphi*ux + dsinphi*vx
                        nny = nny + dcosphi*uy + dsinphi*vy
                        nnz = nnz + dcosphi*uz + dsinphi*vz

                        rsq = ti.sqrt(nnx*nnx + nny*nny + nnz*nnz)
                        nnx = nnx/rsq
                        nny = nny/rsq
                        nnz = nnz/rsq

                # calculate enew w/ trial spin
                enew = energy(nx,ny,nz,nnx,nny,nnz,s[i,j,k],s,g,KK,i,j,k,ni,nj,nk)
                # metropolis algorithm
                if (enew < eold):
                        nx[i,j,k] = nnx
                        ny[i,j,k] = nny
                        nz[i,j,k] = nnz
                        naccept += 1
                else:
                        if ti.randn() <= ti.exp(-(enew-eold)/kbt):
                                nx[i,j,k] = nnx
                                ny[i,j,k] = nny
                                nz[i,j,k] = nnz
                                naccept += 1
                # monte-carlo for switching chirality
                nnx = nx[i,j,k]
                nny = ny[i,j,k]
                nnz = nz[i,j,k]
                eold = energy(nx,ny,nz,nnx,nny,nnz,s[i,j,k],s,g,KK,i,j,k,ni,nj,nk)
                snew = -s[i,j,k]
                enew = energy(nx,ny,nz,nnx,nny,nnz,snew,s,g,KK,i,j,k,ni,nj,nk)
                # metropolis
                if (enew < eold):
                        s[i,j,k] = snew
                        nflip += 1
                else:
                        if ti.randn() <= ti.exp(-(enew-eold)/kbt):
                                s[i,j,k] = snew
                                nflip += 1
        return vec2(naccept,nflip)
                        
        

class ChLLSim:
        def __init__(self,ni,nj,nk,kbt,d,KK,g,rho,name=None,overwrite=False):
                # parameters
                self.name = name
                self.ni = ni
                self.nj = nj
                self.nk = nk
                self.kbt = kbt
                self.d = d
                self.KK = KK
                self.g = g
                self.rho = rho

                # internal parameters
                self.overwrite=overwrite
                self.simpath=None
                self.nsub = int((self.ni*self.nk*self.nk)/2)

                #arrays
                self.nx = ti.ndarray(dtype=float, shape=(self.ni, self.nj, self.nk))
                self.ny = ti.ndarray(dtype=float, shape=(self.ni, self.nj, self.nk))
                self.nz = ti.ndarray(dtype=float, shape=(self.ni, self.nj, self.nk))
                self.s = ti.ndarray(dtype=int, shape=(self.ni, self.nj, self.nk))
                self.dope = ti.ndarray(dtype=int, shape=(self.ni,self.nj,self.nk))
                self.sl1 = ti.ndarray(dtype=int, shape=(self.nsub))
                self.sl2 = ti.ndarray(dtype=int, shape=(self.nsub))

                self.m = None
                self.msteps = None
                self.setup_folders()

        def init(self,init_type):
                s_init = np.random.choice([-1,1],(self.ni,self.nj,self.nk))
                self.s.from_numpy(s_init)
                nx_init = np.zeros((self.ni,self.nj,self.nk),dtype=float)
                ny_init = np.zeros((self.ni,self.nj,self.nk),dtype=float)
                nz_init = np.zeros((self.ni,self.nj,self.nk),dtype=float)
                if type(init_type) == str and init_type.lower() == "random":
                        for i in range(self.ni):
                                for j in range(self.nj):
                                        for k in range(self.nk):
                                                costh = 2.0*(np.random.rand()-0.5)
                                                sinth = np.sqrt(1.0-costh*costh)
                                                phi = np.random.rand()*2*np.pi
                                                cosphi = np.cos(phi)
                                                sinphi = np.sin(phi)
                                                nx_init[i,j,k] = sinth*cosphi
                                                ny_init[i,j,k] = sinth*sinphi
                                                nz_init[i,j,k] = costh
                elif type(init_type) == tuple:
                        nx_init[:,:,:] = init_type[0]
                        ny_init[:,:,:] = init_type[1]
                        nz_init[:,:,:] = init_type[2]
                else:
                        print("Error: invalid init type")
                        exit()
                self.nx.from_numpy(nx_init)
                self.ny.from_numpy(ny_init)
                self.nz.from_numpy(nz_init)
                
                sl1_init = np.zeros(self.nsub,dtype=int)
                sl2_init = np.zeros(self.nsub,dtype=int)
                nsub1 = 0
                nsub2 = 0
                for i in range(self.ni):
                        for j in range(self.nj):
                                for k in range(self.nk):
                                        idx = i + (j*self.ni) + k*(self.ni*self.nj)
                                        if (i+j+k)%2 != 0:
                                                sl1_init[nsub1] = idx
                                                nsub1 += 1
                                        else:
                                                sl2_init[nsub2] = idx
                                                nsub2 += 1

                self.sl1.from_numpy(sl1_init)
                self.sl2.from_numpy(sl2_init)

        def setup_folders(self):
                # this assumes that every sim script is run from the ChLLModel root folder
                data_folder = "./data"
                if self.name == None:
                        print("> Not saving sim")
                else:
                        if os.path.exists(data_folder):
                                simname_folder_path = os.path.join(data_folder,self.name)
                                if os.path.exists(simname_folder_path):
                                        print(f"> ERROR: '{self.name}' in '{data_folder}'!")
                                        if self.overwrite:
                                                print(f"> Overwriting '{self.name}' in '{data_folder}' ... ")
                                        else:
                                                exit()
                                else:
                                        os.makedirs(simname_folder_path,exist_ok=True)
                                        print(f"> {simname_folder_path}")
                                        self.simpath= simname_folder_path
                        else:
                                print(f"> ERROR: '{data_folder}' does not exist!")
                                exit()

        def save_params(self,nsteps,nout):
                params_name = self.name+"_params.txt"
                f = open(os.path.join(self.simpath,params_name),"w")
                f.write(self.name + "\n")
                f.write("nsteps = "+str(nsteps)+" nout = "+ str(nout)+"\n")
                f.write("ni,nj,nk = "+str(self.ni)+" "+str(self.nj)+" "+str(self.nk)+"\n")
                f.write("kbt = "+str(self.kbt)+"\n")
                f.write("d = "+str(self.d)+"\n")
                f.write("KK = "+str(self.KK)+"\n")
                f.write("g = "+str(self.g) + "\n")
                f.write("rho = "+str(self.rho)+"\n")
                f.close()

        def save_config(self):
                if self.name != None:
                        if os.path.exists(self.simpath):
                                datafile_name = self.name+"_data.npz"
                                datafile_path = os.path.join(self.simpath,datafile_name)
                                if os.path.isfile(datafile_path):
                                        if self.overwrite:
                                                np.savez_compressed(datafile_path,nx = self.nx,ny = self.ny,nz = self.nz,s = self.s,dope = self.dope)
                                                print(f"> Wrote '{datafile_path}'")
                                        else:
                                                print(f"> ERROR: '{datafile_path}' exists and overwrite=False")
                                                exit()
                                else:
                                        np.savez_compressed(datafile_path,nx = self.nx,ny = self.ny,nz = self.nz,s = self.s,dope = self.dope)
                                        print(f"> Wrote '{datafile_path}'")
                        else:
                                print(f"> ERROR: '{self.simpath}' does not exist")
                                exit()

        def save_macros(self):
                if self.name != None:
                        if os.path.exists(self.simpath):
                                csv_name = self.name+".csv"
                                csv_path =  os.path.join(self.simpath,csv_name)
                                if os.path.isfile(csv_path):
                                        if self.overwrite:
                                                fcsv = open(csv_path,"w")
                                                fcsv.write("Step,TotalEnergy,Eexcess,Paccept,Faccept,D\n")
                                                for i in range(len(self.msteps)):
                                                        fcsv.write(str(int(self.msteps[i]))+","+str(self.m[i,0])+","+str(self.m[i,1])+","+str(self.m[i,2])+","+str(self.m[i,3])+","+str(self.m[i,4])+"\n")
                                                        fcsv.close()
                                                print(f"> Wrote '{csv_path}'")
                                        else:
                                                print(f"> ERROR: '{csv_path}' exists and overwrite=False")
                                                exit()
                                else:
                                        fcsv = open(csv_path,"w")
                                        fcsv.write("Step,TotalEnergy,Eexcess,Paccept,Faccept,D\n")
                                        for i in range(len(self.msteps)):
                                                fcsv.write(str(int(self.msteps[i]))+","+
                                                           str(self.m[i,0])+","+
                                                           str(self.m[i,1])+","+
                                                           str(self.m[i,2])+","+
                                                           str(self.m[i,3])+","+
                                                           str(self.m[i,4])+"\n")
                                        fcsv.close()
                                        print(f"> Wrote '{csv_path}'")
                        else:
                                print(f"> ERROR: '{self.simpath}' does not exist")
                                exit()

        def plot_config(self):
                import matplotlib.pyplot as plt
                X,Z = np.meshgrid(np.arange(0,self.ni),np.arange(0,self.nk))
                mp = int(np.floor(self.nj/2))
                u = self.nx[:,mp,:]
                v = self.ny[:,mp,:]
                w = self.nz[:,mp,:]
                s_color = self.s[:,mp,:]
                fig = plt.figure(figsize=(10,10))
                ax = fig.add_subplot(111)
                ax.quiver(X,Z,u,w,s_color,pivot='mid',headlength=0,headwidth=0,headaxislength=0,scale_units='xy',scale=0.75,cmap='coolwarm')
                ax.set_xlabel('X')
                ax.set_ylabel('Z')
                ax.set_title(self.name)
                pngname = self.name+".png"
                plt.savefig(os.path.join(self.simpath,pngname))
                
        def run(self,nsteps,nout,save=False):
                #self.save_params(nsteps,nout)
                self.msteps = np.zeros((int(nsteps/nout)),dtype=np.int32,order='F')
                self.m = np.zeros((int(nsteps/nout),5),dtype=np.float64, order='F')
                print("> RUNNING")
                st = time.perf_counter()
                n3 = self.ni*self.nj*self.nk
                nsub = int(n3/2)
                m_count = 0
                print("Steps,\tEnergy,\tE. excess,\tPaccept,\tFaccept,\tD")
                for istep in range(nsteps):
                        naccept = 0
                        nflip = 0
                        res = evolve(self.sl1,self.nx,self.ny,self.nz,self.s,self.g,self.KK,self.d,self.kbt,nsub,self.ni,self.nj,self.nk)
                        naccept += res[0]
                        nflip += res[1]
                        res = evolve(self.sl1,self.nx,self.ny,self.nz,self.s,self.g,self.KK,self.d,self.kbt,nsub,self.ni,self.nj,self.nk)
                        naccept += res[0]
                        nflip += res[1]
                        paccept = float(naccept)/float(self.ni*self.nj*self.nk)
                        if (istep < 10000):
                                if paccept < 0.4:
                                      self.d = self.d*0.995
                                elif paccept > 0.6:
                                        self.d = self.d/0.995
                        if (istep % nout == 0):
                                faccept = float(nflip)/float(n3)
                                total_energy = etot(self.nx,self.ny,self.nz,self.s,self.g,self.KK)
                                e_excess = float(sum(self.s))/float(n3)
                                print(f"{istep}\t{np.round(total_energy,5)}\t\t{np.round(e_excess,5)}\t{np.round(paccept,5)}\t\t{np.round(faccept,5)}\t\t{np.round(self.d,3)}")
                                self.msteps[m_count] = istep
                                self.m[m_count,0] = total_energy
                                self.m[m_count,1] = e_excess
                                self.m[m_count,2] = paccept
                                self.m[m_count,3] = faccept
                                self.m[m_count,4] = self.d
                                m_count += 1
                et = time.perf_counter()
                if save:
                        self.save_macros()
                        self.save_config()
                print("> DONE! ",et-st)
                
                
                
if (__name__ == "__main__"):
        ti.init(arch=ti.cpu,default_ip=ti.i64, default_fp=ti.f64)
        sim = ChLLSim(ni=64,nj=64,nk=64,kbt=0.5,d = 0.1,KK=0.0,g=0.0,rho=0.0)
        sim.init((1,0,0))
        sim.run(1000,50)
        print("Done!")
