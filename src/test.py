import ctypes as ct
import numpy as np
from numpy import ctypeslib

# load the shared library
libchll_mod = ctypeslib.load_library("libchll_mod",'./')

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
ftn_run.argtypes = [ct.POINTER(ct.c_int), #nsteps
					 ctypeslib.ndpointer(dtype=np.float64,ndim=3,flags='F_CONTIGUOUS'), # nx
					 ctypeslib.ndpointer(dtype=np.float64,ndim=3,flags='F_CONTIGUOUS'), # ny
					 ctypeslib.ndpointer(dtype=np.float64,ndim=3,flags='F_CONTIGUOUS'), # nz
					 ctypeslib.ndpointer(dtype=np.int32,ndim=3,flags='F_CONTIGUOUS'),	# s
					 ctypeslib.ndpointer(dtype=np.int32,ndim=3,flags='F_CONTIGUOUS'),	# dope
					 ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags='F_CONTIGUOUS'),	# sl1
					 ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags='F_CONTIGUOUS'),	# sl2
					 ct.POINTER(ct.c_double), # KK
					 ct.POINTER(ct.c_double), # d
					 ct.POINTER(ct.c_double), # kbt
					 ct.POINTER(ct.c_int),ct.POINTER(ct.c_int),ct.POINTER(ct.c_int),ct.POINTER(ct.c_int)]
ftn_run.restype = None

ftn_step = libchll_mod.step
ftn_step.argtypes = [ct.POINTER(ct.c_int), #istep
					 ctypeslib.ndpointer(dtype=np.float64,ndim=3,flags='F_CONTIGUOUS'), # nx
					 ctypeslib.ndpointer(dtype=np.float64,ndim=3,flags='F_CONTIGUOUS'), # ny
					 ctypeslib.ndpointer(dtype=np.float64,ndim=3,flags='F_CONTIGUOUS'), # nz
					 ctypeslib.ndpointer(dtype=np.int32,ndim=3,flags='F_CONTIGUOUS'),	# s
					 ctypeslib.ndpointer(dtype=np.int32,ndim=3,flags='F_CONTIGUOUS'),	# dope
					 ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags='F_CONTIGUOUS'), # rand1
					 ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags='F_CONTIGUOUS'), # rand2
					 ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags='F_CONTIGUOUS'),	# sl1
					 ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags='F_CONTIGUOUS'),	# sl2
					 ct.POINTER(ct.c_double), # KK
					 ct.POINTER(ct.c_double), # d
					 ct.POINTER(ct.c_double), # kbt
					 ct.POINTER(ct.c_int),ct.POINTER(ct.c_int),ct.POINTER(ct.c_int),ct.POINTER(ct.c_int)]
ftn_step.restype = None

class ChLLSim:
	def __init__(self,ni,nj,nk,kbt,d,KK,rho):
		# parameters
		self.ni = ni
		self.nj = nj
		self.nk = nk
		self.kbt = kbt
		self.d = d
		self.KK = KK
		self.rho = rho

		# internal parameters
		self.nsub = int((self.ni*self.nj*self.nk)/2)

		# arrays
		self.nx = np.zeros((self.ni,self.nj,self.nk),dtype=np.float64)
		self.ny = np.zeros((self.ni,self.nj,self.nk),dtype=np.float64)
		self.nz = np.zeros((self.ni,self.nj,self.nk),dtype=np.float64)
		self.s = np.zeros((self.ni,self.nj,self.nk),dtype=np.int32)
		self.dope = np.zeros((self.ni,self.nj,self.nk),dtype=np.int32)
		self.sl1 = np.zeros(self.nsub,dtype=np.int32)
		self.sl2 = np.zeros(self.nsub,dtype=np.int32)

	def init(self):
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
		ftn_init(self.nx,self.ny,self.nz,self.s,self.dope,self.sl1,self.sl2,
				 ct.c_int(self.ni),ct.c_int(self.nj),ct.c_int(self.nk),ct.c_int(self.nsub))

	def finalize(self):
		if not self.nx.flags['F_CONTIGUOUS']:
			self.nx = np.zeros((self.ni,self.nj,self.nk),dtype=np.float64,order='C')
		if not self.ny.flags['F_CONTIGUOUS']:
			self.ny = np.zeros((self.ni,self.nj,self.nk),dtype=np.float64,order='C')
		if not self.nz.flags['F_CONTIGUOUS']:
			self.nz = np.zeros((self.ni,self.nj,self.nk),dtype=np.float64,order='C')
		if not self.s.flags['F_CONTIGUOUS']:
			self.s = np.zeros((self.ni,self.nj,self.nk),dtype=np.int32,order='C')
		if not self.dope.flags['F_CONTIGUOUS']:
			self.dope = np.zeros((self.ni,self.nj,self.nk),dtype=np.int32,order='C')
		if not self.sl1.flags['F_CONTIGUOUS']:
			self.sl1 = np.zeros(int((self.ni*self.nj*self.nk)/2),dtype=np.int32,order='C')
		if not self.sl2.flags['F_CONTIGUOUS']:
			self.sl1 = np.zeros(int((self.ni*self.nj*self.nk)/2),dtype=np.int32,order='C')
		
	def run2(self,nsteps):
		ftn_run(ct.c_int(nsteps),
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
		self.finalize()
		print("DONE!")
		
	def run(self,nsteps):
		rand1 = np.zeros((self.nsub,2),dtype=np.float64,order='F')
		rand2 = np.zeros((self.nsub,2),dtype=np.float64,order='F')
		for istep in range(1,nsteps+1):
			ftn_step(ct.c_int(istep),
					self.nx,
					self.ny,
					self.nz,
					self.s,
					self.dope,
					rand1,
					rand2,
					self.sl1,
					self.sl2,
					ct.c_double(self.KK),
					ct.c_double(self.d),
					ct.c_double(self.kbt),
					ct.c_int(self.ni),ct.c_int(self.nj),ct.c_int(self.nk),ct.c_int(self.nsub))
		self.finalize()
		print("DONE!")

	def output(self):
		fa = open("LLMC_py-configa.dat","w")
		fb = open("LLMC_py-configb.dat","w")
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

		
if __name__ == '__main__':
	sim = ChLLSim(64,64,64,0.05,0.01,1.0,0.0)
	sim.init()
	print("init done")
	sim.run(1000)
	sim.output()
	
