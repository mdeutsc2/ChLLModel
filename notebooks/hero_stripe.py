import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import os
import freud
from tqdm.auto import tqdm
from numba import njit,prange
import scipy

#@njit(parallel=True)
@njit
def get_sc(s,kvec):
    sc = 0.0
    for i in prange(s.shape[0]):
        for j in range(s.shape[1]):
            for k in range(s.shape[2]):
                sc += s[i,j,k]*np.exp(-1j * np.dot(kvec,np.array([float(i+1),float(j+1),float(k+1)])))
    return sc

def structure_factor(data,bins,k_min,k_max,plane):
    N3 = (data.shape[0]*data.shape[1]*data.shape[2])
    # kv_pre = (2*np.pi)/(np.sqrt(3)*data.shape[0])
    # kvs = kv_pre*np.arange(bins)
    kvs = np.linspace(k_min,k_max,bins)#(0.0,0.5,500)
    sc = np.zeros(len(kvs),dtype=complex)
    for idx,kv in enumerate(tqdm(kvs)):
        kvec = kv*np.array(plane)
        sc[idx] = get_sc(data,kvec)/N3
    sf = sc.real*sc.real + sc.imag*sc.imag
    return sf,kvs

sims = ["../data/hero4801/hero4801_data.npz",
       "../data/hero4802/hero4802_data.npz",
       "../data/hero4803/hero4803_data.npz",
       "../data/hero4804/hero4804_data.npz",
       "../data/hero4805/hero4805_data.npz",
       "../data/hero4806/hero4806_data.npz",
       "../data/hero4807/hero4807_data.npz",
       "../data/hero4808/hero4808_data.npz",
       "../data/hero4809/hero4809_data.npz"]

planes = [[1,0,0],[0,1,0],[0,0,1], # faces
          [1,1,1],[1,1,-1,],[1,-1,1],[1,-1,-1,]] #diagonals
for datafile in sims:
    print(datafile)
    data = np.load(datafile)
    for p in planes:
        sf,kvs = structure_factor(data['s'],500,0.0,0.5,p)
        textstr = "Peak: "+str(np.round(kvs[np.argmax(sf)],5))+"\n"+"2*Pi/Peak: "+str(np.round((2.0*np.pi)/kvs[np.argmax(sf)],5))+"\n"+"1/Peak: "+str(np.round(1/kvs[np.argmax(sf)],5))
        # print("Peak:\t",kvs[np.argmax(sf)])
        # print("2*Pi/Peak:\t",(2.0*np.pi)/kvs[np.argmax(sf)])
        # print("1/Peak:\t",1/kvs[np.argmax(sf)])
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.set_title(str(p))
        ax.set_xlabel("k")
        ax.set_ylabel("Structure Factor")
        ax.scatter(kvs,sf,marker='x')
        ax.plot(kvs,sf,linewidth=2)
        ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        fname = datafile.split("/")[-1].split(".")[0].split("_")[0]+"_"+str(p[0])+str(p[1])+str(p[2])+".png"
        fig.savefig(fname)
