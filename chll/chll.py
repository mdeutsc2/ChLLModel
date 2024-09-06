import os,time
import ctypes as ct
import numpy as np
from numpy import ctypeslib

# load the shared library
libchll_mod = ctypeslib.load_library("libchll_mod",'./chll/')

ftn_init = libchll_mod.init
ftn_init.argtypes = [ctypeslib.ndpointer(dtype=np.float64,ndim=3,flags='F_CONTIGUOUS'), # nx
					 ctypeslib.ndpointer(dtype=np.float64,ndim=3,flags='F_CONTIGUOUS'), # ny
					 ctypeslib.ndpointer(dtype=np.float64,ndim=3,flags='F_CONTIGUOUS'), # nz
					 ctypeslib.ndpointer(dtype=np.int32,ndim=3,flags='F_CONTIGUOUS'),	# s
					 ctypeslib.ndpointer(dtype=np.int32,ndim=3,flags='F_CONTIGUOUS'),	# dope
					 ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags='F_CONTIGUOUS'),	# sl1
					 ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags='F_CONTIGUOUS'),	# sl2
					 ct.POINTER(ct.c_int),ct.POINTER(ct.c_int),ct.POINTER(ct.c_int),ct.POINTER(ct.c_int)]
ftn_init.restype = None

ftn_run = libchll_mod.run
ftn_run.argtypes = [ct.POINTER(ct.c_int),ct.POINTER(ct.c_int), ct.POINTER(ct.c_int),#nstart,nstop,nout
					 ctypeslib.ndpointer(dtype=np.float64,ndim=3,flags='F_CONTIGUOUS'), # nx
					 ctypeslib.ndpointer(dtype=np.float64,ndim=3,flags='F_CONTIGUOUS'), # ny
					 ctypeslib.ndpointer(dtype=np.float64,ndim=3,flags='F_CONTIGUOUS'), # nz
					 ctypeslib.ndpointer(dtype=np.int32,ndim=3,flags='F_CONTIGUOUS'),	# s
					 ctypeslib.ndpointer(dtype=np.int32,ndim=3,flags='F_CONTIGUOUS'),	# dope
					 ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags='F_CONTIGUOUS'),	# sl1
					 ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags='F_CONTIGUOUS'),	# sl2
					 ct.POINTER(ct.c_double), # g
					 ct.POINTER(ct.c_double), # KK
					 ct.POINTER(ct.c_double), # d
					 ct.POINTER(ct.c_double), # kbt
					 ct.POINTER(ct.c_int),ct.POINTER(ct.c_int),ct.POINTER(ct.c_int),ct.POINTER(ct.c_int),
					 ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags='F_CONTIGUOUS'),
					 ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags='F_CONTIGUOUS')]
ftn_run.restype = None

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
		self.simpath = None
		self.nsub = int((self.ni*self.nj*self.nk)/2)

		# arrays
		self.nx = np.zeros((self.ni,self.nj,self.nk),dtype=np.float64)
		self.ny = np.zeros((self.ni,self.nj,self.nk),dtype=np.float64)
		self.nz = np.zeros((self.ni,self.nj,self.nk),dtype=np.float64)
		self.s = np.zeros((self.ni,self.nj,self.nk),dtype=np.int32)
		self.dope = np.zeros((self.ni,self.nj,self.nk),dtype=np.int32)
		self.sl1 = np.zeros(self.nsub,dtype=np.int32)
		self.sl2 = np.zeros(self.nsub,dtype=np.int32)

		self.m = None # measurables
		self.msteps = None
		self.setup_folders()
		# needs to be called before any modification
		if not self.nx.flags['F_CONTIGUOUS']:
			self.nx = np.zeros((self.ni,self.nj,self.nk),dtype=np.float64,order='F')
		if not self.ny.flags['F_CONTIGUOUS']:
			self.ny = np.zeros((self.ni,self.nj,self.nk),dtype=np.float64,order='F')
		if not self.nz.flags['F_CONTIGUOUS']:
			self.nz = np.zeros((self.ni,self.nj,self.nk),dtype=np.float64,order='F')
		if not self.s.flags['F_CONTIGUOUS']:
			self.s = np.zeros((self.ni,self.nj,self.nk),dtype=np.int32,order='F')
		if not self.dope.flags['F_CONTIGUOUS']:
			self.dope = np.zeros((self.ni,self.nj,self.nk),dtype=np.int32,order='F')
		if not self.sl1.flags['F_CONTIGUOUS']:
			self.sl1 = np.zeros(int((self.ni*self.nj*self.nk)/2),dtype=np.int32,order='F')
		if not self.sl2.flags['F_CONTIGUOUS']:
			self.sl1 = np.zeros(int((self.ni*self.nj*self.nk)/2),dtype=np.int32,order='F')

	def init(self,init_s = None):
		if not init_s:
			for i in range(self.ni):
				for j in range(self.nj):
					for k in range(self.nk):
						self.s[i,j,k] = 1
						if np.random.rand() <= 0.5:
							self.s[i,j,k] = -1
		else:
			self.s[:,:,:] = init_s
			
		if ((np.mean(np.sqrt(self.nx*self.nx + self.ny*self.ny + self.nz*self.nz))) == 0.0):
			print("> Nx,Ny,Nz not initialized, setting nx[:,:,:] = 1.0")
			for i in range(self.ni):
				for j in range(self.nj):
					for k in range(self.nk):
						self.nx[i,j,k] = 1.0
						self.ny[i,j,k] = 0.0
						self.nz[i,j,k] = 0.0
		ftn_init(self.nx,self.ny,self.nz,self.s,self.dope,self.sl1,self.sl2,
				 ct.c_int(self.ni),ct.c_int(self.nj),ct.c_int(self.nk),ct.c_int(self.nsub))
		print("> Init Done!")

	def setup_folders(self):
		# Note: this assumes that every sim script is run from the ChLLModel root folder
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
						os.makedirs(simname_folder_path,exist_ok=True)
					else:
						exit()
				else:
					os.makedirs(simname_folder_path, exist_ok=True)
					print(f"> {simname_folder_path}")
				self.simpath = simname_folder_path
			else:
				print(f"ERROR: '{data_folder}' does not exist.")
				exit()

	def save_params(self,nsteps,nout):
		params_name = self.name + self.name+"_params.txt"
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
			
		
	def run(self,nsteps,nout,save=False):
		self.save_params(nsteps,nout)
		self.msteps = np.zeros((int(nsteps/nout)),dtype=np.int32,order='F')
		self.m = np.zeros((int(nsteps/nout),5),dtype=np.float64, order='F')
		print("> RUNNING")
		st = time.perf_counter()
		ftn_run(ct.c_int(1),ct.c_int(nsteps),ct.c_int(nout),
				self.nx,
				self.ny,
				self.nz,
				self.s,
				self.dope,
				self.sl1,
				self.sl2,
				ct.c_double(self.g),
				ct.c_double(self.KK),
				ct.c_double(self.d),
				ct.c_double(self.kbt),
				ct.c_int(self.ni),ct.c_int(self.nj),ct.c_int(self.nk),ct.c_int(self.nsub),
				self.msteps,self.m)
		et = time.perf_counter()
		if save:
			self.save_macros()
			self.save_config()
		print("> DONE! ",et-st)
		"""
		for i in range(1,nsteps,nout):
			nstart = i
			nstop = i + nout-1
			ftn_run(ct.c_int(nstart),ct.c_int(nstop),
				self.nx,
				self.ny,
				self.nz,
				self.s,
				self.dope,
				self.sl1,
				self.sl2,
				ct.c_double(self.KK),
				ct.c_double(self.d),
				ct.c_double(self.kbt),
				ct.c_int(self.ni),ct.c_int(self.nj),ct.c_int(self.nk),ct.c_int(self.nsub))
		#self.finalize()
		"""
	
	def save_config(self):
		if self.name != None:
			if os.path.exists(self.simpath):
				datafile_name = self.name+"_data.npz"
				datafile_path = os.path.join(self.simpath,datafile_name)
				if os.path.isfile(datafile_path):
					if self.overwrite:
						np.savez_compressed(datafile_path,
											nx = self.nx,
											ny = self.ny,
											nz = self.nz,
											s = self.s,
											dope = self.dope)
						print(f"> Wrote '{datafile_path}'")
					else:
						print(f"> ERROR: '{datafile_path}' exists and overwrite=False")
						exit()
				else:
					np.savez_compressed(datafile_path,
										nx = self.nx,
										ny = self.ny,
										nz = self.nz,
										s = self.s,
										dope = self.dope)
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
							fcsv.write(str(int(self.msteps[i]))+","+
									   str(self.m[i,0])+","+
									   str(self.m[i,1])+","+
									   str(self.m[i,2])+","+
									   str(self.m[i,3])+","+
									   str(self.m[i,4])+"\n")
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
		
	def output_old(self):
		fa = open(self.name+".dat","w")
		fb = open(self.name+".dat","w")
		scale = 0.4
		for i in range(self.ni):
			j = int(self.nj/2)
			for k in range(self.nk):
				x1 = i-scale*self.nx[i,j,k]
				x2 = i+scale*self.nx[i,j,k]
				z1 = k-scale*self.nz[i,j,k]
				z2 = k+scale*self.nz[i,j,k]
				if self.s[i,j,k] == 1:
					fa.write(str(x1)+" "+str(z1)+"\n")
					fa.write(str(x2)+" "+str(z2)+"\n")
					fa.write("\n")
				else:
					fb.write(str(x1)+" "+str(z1)+"\n")
					fb.write(str(x2)+" "+str(z2)+"\n")
					fb.write("\n")
		fa.close()
		fb.close()

		
