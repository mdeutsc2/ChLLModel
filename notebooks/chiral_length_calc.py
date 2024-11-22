import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d 
import os
from tqdm.auto import tqdm
from numba import njit,prange
import scipy
from joblib import Parallel,delayed

def load_csv(fname,verbose=True):
    if verbose:
        with open(fname) as f:
            print(f.readline().strip('\n'))
    return np.loadtxt(fname,skiprows=1,delimiter=",")
                       
@njit
def calc_s_corr(s,nshells):
    g = np.zeros(nshells)
    c = np.zeros(nshells)
    ni,nj,nk = s.shape
    #s_avg2 = np.mean(s)**2
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                #print(i,j,k)
                for di in np.arange(-nshells,nshells+1):
                    for dj in np.arange(-nshells,nshells+1):
                        for dk in np.arange(-nshells,nshells+1):
                            r2 = di*di + dj*dj + dk*dk
                            if r2 < nshells*nshells:
                                rr = np.sqrt(r2)
                                ii = i + di
                                jj = j + dj
                                kk = k + dk
                                if (ii < ni) and (jj < nj) and (kk < nk) and (ii >= 0) and (jj >= 0) and (kk >= 0):
                                    ishell = int(np.floor(rr))
                                    #assert ishell > 0
                                    g[ishell] = g[ishell] + s[i,j,k] * s[ii,jj,kk]# - s_avg2
                                    c[ishell] += 1
    for ishell in np.arange(nshells):
        g[ishell] = g[ishell]/c[ishell]# - s_avg2
    return g,c

@njit
def calc_s_corr2(s,nshells):
    g = np.zeros(nshells)
    c = np.zeros(nshells)
    ni,nj,nk = s.shape
    s_avg2 = np.mean(s)**2
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                #print(i,j,k)
                for di in np.arange(-nshells,nshells+1):
                    for dj in np.arange(-nshells,nshells+1):
                        for dk in np.arange(-nshells,nshells+1):
                            r2 = di*di + dj*dj + dk*dk
                            if r2 < nshells*nshells:
                                rr = np.sqrt(r2)
                                ii = i + di
                                jj = j + dj
                                kk = k + dk
                                if (ii < ni) and (jj < nj) and (kk < nk) and (ii >= 0) and (jj >= 0) and (kk >= 0):
                                    ishell = int(np.floor(rr))
                                    #assert ishell > 0
                                    g[ishell] = g[ishell] + s[i,j,k] * s[ii,jj,kk] - s_avg2
                                    c[ishell] += 1
    for ishell in np.arange(nshells):
        g[ishell] = g[ishell]/c[ishell]# - s_avg2
    return g,c
    
def plot_s_corr(fpath,g,savedir=None):
    nshells = len(g)
    if savedir is not None:
        newpath = os.path.join(os.getcwd(),savedir)
        if not os.path.isdir(newpath):
            os.makedirs(newpath)
    fig,ax = plt.subplots()
    ax.set_title(fpath.split("/")[-2])
    ax.plot(np.arange(1,nshells+1),g)
    ax.set_ylabel("Chiral Correlation")
    ax.set_xlabel("Shells")
    if savedir is not None:
        plt.savefig(os.path.join(newpath,fpath.split("/")[-2]+".png"))
    else:
        plt.show()

def func1(x, a, l):
    return a * np.exp(-x/l)

def func1c(x, a, l, c):
    return a * np.exp(-x/l) + c

def func2(x,a,l):
    return (a/x)*np.exp(-x/l)

def func2c(x,a,l,c):
    return (a/x)*np.exp(-x/l) + c

def fit_corr_exp1(g):
    nshells = len(g)
    popt,pcov = scipy.optimize.curve_fit(func1,np.arange(nshells),g)
    return popt[0],popt[1]

def fit_corr_exp1c(g):
    nshells = len(g)
    popt,pcov = scipy.optimize.curve_fit(func1c,np.arange(nshells),g)
    return popt[0],popt[1],popt[2]

def fit_corr_exp2(g):
    nshells = len(g)
    popt,pcov = scipy.optimize.curve_fit(func2,np.arange(nshells),g)
    return popt[0],popt[1]

def fit_corr_exp2c(g):
    nshells = len(g)
    popt,pcov = scipy.optimize.curve_fit(func2c,np.arange(nshells),g)
    return popt[0],popt[1],popt[2]

def plot_corr_fit(fpath,g,fit,a,l,savedir=None): #func1
    nshells = len(g)
    newpath = os.path.join(os.getcwd(),savedir)
    nshells = len(g)
    fig,ax = plt.subplots()
    ax.set_title(fpath.split("/")[-2])
    ax.plot(np.arange(1,nshells+1),g)
    label_str = "a="+str(np.round(a,5))+"\nL="+str(np.round(l,5))
    ax.plot(np.arange(1,nshells+1),fit,label=label_str)
    ax.set_ylabel("Chiral Correlation")
    ax.set_xlabel("Shells")
    ax.legend()
    plt.show()
    if savedir is not None:
        plt.savefig(os.path.join(newpath,fpath.split("/")[-2]+"_fit.png"))
    else:
        plt.show()

def plot_corr_fitc(fpath,g,fit,a,l,c,savedir=None):
    nshells = len(g)
    newpath = os.path.join(os.getcwd(),savedir)
    nshells = len(g)
    fig,ax = plt.subplots()
    ax.set_title(fpath.split("/")[-2])
    ax.plot(np.arange(1,nshells+1),g)
    label_str = "a="+str(np.round(a,5))+"\nL="+str(np.round(l,5))+"\nc="+str(np.round(c,5))
    ax.plot(np.arange(1,nshells+1),fit,label=label_str)
    ax.set_ylabel("Chiral Correlation")
    ax.set_xlabel("Shells")
    ax.legend()
    plt.show()
    if savedir is not None:
        plt.savefig(os.path.join(newpath,fpath.split("/")[-2]+"_fitc.png"))
    else:
        plt.show()


#NOTE: THIS TAKES ABOUT 22 HOURS TO RUN
def run(savename,fittype,stype):
    nshells=16
    # loading all of the csv's from all of the simulations with the macro measured data (total energy, ent. excess, Paccept....)
    simlist = []
    datafiles = []
    csvfiles = []
    for root, dirs, files in os.walk("../data/pdanneal/"):
        for file in files:
            if file.endswith(".npz"):
                datafiles.append(os.path.join(root,file))
            if file.endswith(".csv"):
                csvfiles.append(os.path.join(root,file))
        for name in dirs:
            simlist.append(name)
    simlist = sorted(simlist)
    datafiles = sorted(datafiles)
    csvfiles = sorted(csvfiles)
    assert len(simlist) == len(datafiles) == len(csvfiles)
    Ks = []
    kbts = []
    for csvfile in csvfiles:
        csv1 = load_csv(csvfile,verbose=False)
        K = float(csvfile.split("/")[3].split("_")[-2][1:])
        kbt = float(csvfile.split("/")[3].split("_")[-1][3:])
        Ks.append(K)
        kbts.append(kbt)
    Ks = sorted(list(set(Ks))) #list of all Ks w/ duplicates removed
    kbts = sorted(list(set(kbts))) # list of all Kbts w/ duplicates removed
    ch_l = np.zeros((len(kbts),len(Ks)))
    for datafile in tqdm(datafiles):
        data = np.load(datafile)
        K = float(datafile.split("/")[3].split("_")[-2][1:])
        kbt = float(datafile.split("/")[3].split("_")[-1][3:])
        if K in Ks:
            K_index = Ks.index(K)
            kbt_index = kbts.index(kbt)
            ni,nj,nk = data['nx'].shape    
            nshells = 16
            if stype == 1:
                g,c = calc_s_corr(np.array(data['s']),nshells)
            elif stype == 2:
                g,c = calc_s_corr2(np.array(data['s']),nshells)
            else:
                print("wong s type")
                print("stype=1, w/o -savg2")
                print("stype=2, -savg2")
                exit()
            plot_s_corr(datafile,g,savedir=savename)
            if fittype == "1":
                # a*exp(-x/l)
                a,l = fit_corr_exp1(g) 
                fit = func1(np.arange(nshells),a,l)
                plot_corr_fit(datafile,g,fit,a,l,savedir=savename)
            elif fittype == "1c":
                # a*exp(-x/l)+c
                a,l,c = fit_corr_exp1c(g) 
                fit = func1c(np.arange(nshells),a,l,c)
                plot_corr_fitc(datafile,g,fit,a,l,c,savedir=savename)
            elif fittype == "2":
                # (a/x)*exp(-x/l)
                a,l = fit_corr_exp2(g) 
                fit = func2(np.arange(1,nshells+1),a,l)
                plot_corr_fit(datafile,g,fit,a,l,savedir=savename)
            elif fittype == "2c":
                # (a/x)*exp(-x/l)+c
                a,l,c = fit_corr_exp2c(g) 
                fit = func2c(np.arange(1,nshells+1),a,l,c)
                plot_corr_fitc(datafile,g,fit,a,l,c,savedir=savename)
            else:
                print("wrong fit type")
                exit()
            
            if l > np.array(data['s']).shape[0]:
                l = np.array(data['s']).shape[0]
            ch_l[kbt_index,K_index] = l
    np.savez_compressed(name+".npz",ch_l=ch_l)

#name = "ch_len_pdanneal1"
#name = "ch_len_pdanneal1c"
#name = "ch_len_pdanneal2"
#name = "ch_len_pdanneal2c"
#name = "ch_len_pdanneal1-mean"
# name = "ch_len_pdanneal1c-mean"
# name = "ch_len_pdanneal2-mean"
# name = "ch_len_pdanneal2c-mean"
names = ["ch_len_pdanneal1","ch_len_pdanneal1c","ch_len_pdanneal2","ch_len_pdanneal2c","ch_len_pdanneal1-mean","ch_len_pdanneal1c-mean","ch_len_pdanneal2-mean","ch_len_pdanneal2c-mean"]
fittypes = ["1","1c","2","2c","1","1c","2","2c"]
corr_types = [1,1,1,1,2,2,2,2]

Parallel(n_jobs=len(names))(delayed(run)(names[i],fittypes[i],corr_types[i]) for i in range(len(names)))