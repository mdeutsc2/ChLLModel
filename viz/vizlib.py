#!/usr/local/anaconda3/bin/python3
import matplotlib.pyplot as plt
#import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from itertools import islice,chain
from tqdm.auto import tqdm
import sys,os,glob,gc
import linecache
from joblib import Parallel, delayed
from numpy.lib.format import open_memmap
import tempfile
import pandas as pd


#################################################
# FILE OPERATIONS
#################################################

def load_data_fast(filename):
    with open(filename) as f:
        num_particles = f.readline()
        dims = tuple(map(int,num_particles.rstrip().replace('(','').replace(')','').split(',')))
        N = dims[0]*dims[1]*dims[2]
        print("number of particles",N)
        num_lines = sum(1 for line in f)
        print("total number of lines in file", num_lines)
        f.seek(0) # return to beginning of the file
        first_step = int(next(islice(f,1,2),None))
        f.seek(0)
        last_step = int(next(islice(f,num_lines-N,num_lines-N+1),None)) # just the one line
        steps = [(first_step * i, (i-1)*N + i*2+1,(i-1)*N + i*2 + N) for i in range(1,int(last_step/first_step +1))]

        # generating a list of rows for pandas to skip
        rowskip = []
        prevstop = 0
        for entry in steps:
            start = entry[1]
            rowskip.append(list(np.arange(prevstop,start-1)))
            prevstop = entry[2]

        rowskip = list(chain(*rowskip)) # flattening the list of rows

        print("reading ",filename)
        vectors = np.empty((N,6,len(steps)),dtype=np.float32)
        energies = np.empty((N,1,len(steps)),dtype=np.float32)
        for i,chunk in tqdm(enumerate(pd.read_csv(filename,sep=" ",header=None,skiprows=rowskip,chunksize=N)),total=len(steps)):
            data = chunk.to_numpy(dtype=np.float32)
            vectors[:,:,i]= data[:,:6]
            energies[:,:,i]= data[:,6:]

        print("done!")
        return vectors,energies,steps,dims,N

def get_arrays_parallel_memmap(i,step,filename,start,stop,mmap_vecs,mmap_engy):
    with open(filename,"r") as f:
        data = np.loadtxt(islice(f,start-1,stop),dtype=np.single)
        mmap_vecs[:,:,i] = data[:,:6]
        mmap_engy[:,:,i] = data[:,6:]
        #del data,vectors,energies
        #gc.collect()

def load_data_parallel_memmap(filename,ncpus):
    # getting dims, N and array of all the steps
    with open(filename) as f:
        num_particles = f.readline()
        dims = tuple(map(int,num_particles.rstrip().replace('(','').replace(')','').split(',')))
        N = dims[0]*dims[1]*dims[2]
        print("number of particles",N)
        num_lines = sum(1 for line in f)
        print("total number of lines in file", num_lines)
        f.seek(0) # return to beginning of the file
        first_step = int(next(islice(f,1,2),None))
        f.seek(0)
        last_step = int(next(islice(f,num_lines-N,num_lines-N+1),None)) # just the one line
        steps = [(first_step * i, (i-1)*N + i*2+1,(i-1)*N + i*2 + N) for i in range(1,int(last_step/first_step +1))]

    print("got steps and line data from file")
    # create temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    temp_vecs_file = temp_dir.name+"/vectors.npy"
    temp_engy_file = temp_dir.name+"/energies.npy"
    # figure out size of memmaped array
    mmap_vecs = open_memmap(temp_vecs_file, mode="w+",dtype=np.single, shape=(N,6,len(steps)))
    mmap_engy = open_memmap(temp_engy_file, mode="w+",dtype=np.single, shape=(N,1,len(steps)))
    print("created ",temp_vecs_file," ",temp_engy_file)
    print("loading in parallel")
    #gc.collect()
    Parallel(n_jobs=ncpus,prefer="processes",batch_size="auto")(delayed(get_arrays_parallel_memmap)(i,steps[i][0],filename,steps[i][1],steps[i][2],mmap_vecs,mmap_engy) for i in tqdm(range(0,len(steps))))

    return mmap_vecs,mmap_engy,steps,dims,N

def get_arrays_parallel(i,step,filename,start,stop,vectors,energies):
    with open(filename,"r") as f:
        data = np.loadtxt(islice(f,start-1,stop),dtype=np.single)
        vectors[:,:,i]= data[:,:6]
        energies[:,:,i]= data[:,6:]

def load_data_parallel(filename,ncpus):
    # getting dims, N and array of all the steps
    with open(filename) as f:
        num_particles = f.readline()
        dims = tuple(map(int,num_particles.rstrip().replace('(','').replace(')','').split(',')))
        N = dims[0]*dims[1]*dims[2]
        print("number of particles",N)
        num_lines = sum(1 for line in f)
        print("total number of lines in file", num_lines)
        f.seek(0) # return to beginning of the file
        first_step = int(next(islice(f,1,2),None))
        f.seek(0)
        last_step = int(next(islice(f,num_lines-N,num_lines-N+1),None)) # just the one line
        steps = [(first_step * i, (i-1)*N + i*2+1,(i-1)*N + i*2 + N) for i in range(1,int(last_step/first_step +1))]

    vectors = np.empty((N,6,len(steps))) # creates an empty list for each set of vectors to be stored
    energies = np.empty((N,1,len(steps))) # creates an empty list for each set of energies to be stored
    print("got steps and line data from file")
    print("loading in parallel")
    #gc.collect()
    Parallel(n_jobs=ncpus, prefer="threads")(delayed(get_arrays_parallel)(i,steps[i][0],filename,steps[i][1],steps[i][2],vectors,energies) for i in tqdm(range(0,len(steps))))

#     # unpacking the loaded data
#     print("unpacking loaded data")
#     for i in range(0,len(steps)):
#         vectors[i] = data[i][0]
#         energies[i] = data[i][1]

    return vectors,energies,steps,dims,N

def load_aux_data(filename):
    stripped_fname = filename.split("/")[-1]
    directory = filename[:-len(stripped_fname)]
    param_name = stripped_fname[len("vectors_"):-len("_gpu.xyzv")]
    for file in glob.glob(directory+"*"+param_name+"*"):
        if stripped_fname in file:
            pass # file already loaded
        else:
            if "torque" in file:
                # load angle data
                print("loading "+file)
                angle_data = np.loadtxt(file,delimiter=",",skiprows=1)
            if "energy" in file:
                # load energy data
                print("loading "+file)
                energy_data = np.loadtxt(file, delimiter=",",skiprows=1)

    if "energy_data" not in locals() or "angle_data" not in locals():
        print("NOT ALL AUXILLARY ENERGY OR TORQUE DATA LOADED")
        for file in glob.glob(directory+"*"+param_name+"*"):
            print(file)
        return None,None,None
    else:
        return energy_data,angle_data,param_name

def load_data(filename):
    #getting N
    with open(filename) as f:
        num_particles = f.readline()
        dims = tuple(map(int,num_particles.rstrip().replace('(','').replace(')','').split(',')))
        N = dims[0]*dims[1]*dims[2]

    f = open(filename,"r")
    steps = np.loadtxt(islice(f,1,None,N+2)) # gets an array of all the steps
    f.close()

    vectors = [None] * len(steps) # creates an empty list for each set of vectors to be stored
    energies = [None] * len(steps) # creates an empty list for each set of energies to be stored
    for i in tqdm(range(1,len(vectors)+1)):
        f = open(filename,"r")
        start = (i-1)*N + i*2
        stop = start + N
        data = np.loadtxt(islice(f,start,stop)) #stores each set of vectors in a separate entry
        vectors[i-1] = data[:,:6]
        energies[i-1] = data[:,6:]
        f.close()

    print(steps[-1]," steps loaded")
    return vectors,energies,steps,dims,N

def load_defects(filename):
    with open(filename) as f:
        defects = np.loadtxt(f)

    return defects

#################################################
# 3D PLOTS
#################################################
def get_3d_arrows(vectors,dims,istep):
    # Make the grid
    #xyz_data vectors[0][:,0:2] #vectors[step][columns][select from those columns]
    x, y, z = np.meshgrid(np.arange(1, dims[0]+1, 1),
                        np.arange(1, dims[1]+1, 1),
                        np.arange(1, dims[2]+1, 1))

    # Make the direction data for the arrows
    u = np.reshape(vectors[istep][:,3],dims)
    v = np.reshape(vectors[istep][:,4],dims)
    w = np.reshape(vectors[istep][:,5],dims)
    return x,y,z,u,v,w

def get_2d_arrows(vectors,dims,istep,z):
    x,y = np.meshgrid(np.arange(1,dims[0]+1,1),np.arange(1,dims[1]+1,1))
    rows = np.where(vectors[istep][:,2]==z)[0]
    #vectors = vectors[istep][rows[0]]
    u = np.reshape(vectors[istep][rows][:,3],(dims[0],dims[1]))
    v = np.reshape(vectors[istep][rows][:,4],(dims[0],dims[1]))
    return x,y,u,v

def plot_3d_vec_frame(vectors,frame,dims):
    # Plots a single 3d frame from the animation
    fig,ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.set_title(frame) #NOTE frame-flag starts indexing by 1
    quiver = ax.quiver(*get_3d_arrows(vectors,dims,int(frame-1)),length=1, normalize=True, pivot='middle',arrow_length_ratio=0.01)
    ax.set_xlim(1,dims[0]+1)
    ax.set_ylim(1,dims[1]+1)
    ax.set_zlim(1,dims[2]+1)
    #fig.tightlayout()
    plt.show()

def plot_3d_energy_frame(energies,frame,dims,cutoff):
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    x,y,z = np.meshgrid(np.arange(0,dims[0],1),
                        np.arange(0,dims[1],1),
                        np.arange(0,dims[2],1))
    E = energies[:,:,frame-1]

    minE = np.min(E)
    maxE = np.max(E)
    #print("Max E: ",maxE," Min E: ",minE, " Cutoff: ",cutoff)

    E = np.where(E > cutoff, E, None)
    E = np.reshape(E,dims)
    E[dims[0]-1,:,:] = None
    #E[dims[0]-1,:,:]=0
    img = ax.scatter(x,y,z, c=E, vmin=minE, vmax=maxE, marker="s")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    #ax.set_zlim(1,39)
    fig.colorbar(img)
    plt.show()


def plot_3d_energy_anim_save(energies,steps,dims,cutoff):
    plt.ioff()
    for i in tqdm(range(len(steps))):
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        x,y,z = np.meshgrid(np.arange(0,dims[0],1),
                            np.arange(0,dims[1],1),
                            np.arange(0,dims[2],1))
        E = energies[:,:,i]

        minE = np.min(E)
        maxE = np.max(E)
        #print("Max E: ",maxE," Min E: ",minE, " Cutoff: ",cutoff)

        E = np.where(E > cutoff, E, None)
        E = np.reshape(E,dims)
        E[dims[0]-1,:,:] = None
        img = ax.scatter(x,y,z, c=E, vmin=minE, vmax=maxE, marker="s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(str(steps[i]) + "steps")
        fig.colorbar(img)
        plt.savefig("anim"+str(i).zfill(3)+".png")
        plt.close();

    try:
        os.popen(r'ffmpeg -r 2 -f image2 -s 1920x1080 -i anim%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p energy_animation.mp4').read()
        print("energy animation created")
    except:
        print("energy animation failed")

    os.popen("rm ./*.png")


#################################################
# 2D PLOTS
#################################################
def plot_flat_energy_frame(energies,steps,torque_data,frame,dims,Emax):
    #start_temp,end_temp = 0.1,0.1
    temp = 0.1
    #temp = np.around(np.linspace(start_temp,end_temp,int(1600000/2500))[frame-1],decimals=3)
    angle = np.around(np.rad2deg(torque_data[frame-1])-90,decimals=3)
    fig,ax=plt.subplots(1,1)
    x,y = np.meshgrid(np.arange(0,dims[0],1),
                      np.arange(0,dims[1],1))
    E = energies[:,:,frame-1]
    E = np.reshape(E,dims)
    E_flat = np.zeros((dims[0],dims[1]))
    for iz in range(dims[2]):
        E_flat = E_flat + E[:,:,iz]
    E_flat = E_flat/dims[2]
    #levels = np.linspace(0, 0.16, 64+1)
    levels = np.linspace(0,Emax,64)
    img = ax.contourf(y,x,E_flat,cmap="hot",levels=levels)
    ax.set_title(str(steps[frame-1][0]) + r' steps, $\delta \phi$ = ' + str(angle) +", " +str(temp) + r' $k_bt$')
    #ax.suptitle(plot_name,fontsize="small")
    ax.set_ylabel("y")
    ax.set_xlabel("x")
    #ax.set_xlim(80,240)
    #ax.set_ylim(80,240)
    #img = ax.scatter(y,x,c=E_flat)
    cbar = plt.colorbar(img,extend='max')
    cbar.set_label('Potential Energy')
    plt.show()

def get_2d_vecs(vectors,dims,istep,slx,sly,slz,orientation):
    # sl is slice in dimension of orientation
    if orientation == "xy":
        x,y = np.meshgrid(np.arange(1,dims[0]+1,1),np.arange(1,dims[1]+1,1))
        rows = np.where(vectors[:,:,istep][:,2]==slz)[0]
        #vectors = vectors[istep][rows[0]]
        u = np.reshape(vectors[:,:,istep][rows][:,3],(dims[0],dims[1]))
        v = np.reshape(vectors[:,:,istep][rows][:,4],(dims[0],dims[1]))
        return x,y,u,v
    if orientation == "xz":
        x,z = np.meshgrid(np.arange(1,dims[0]+1,1),np.arange(1,dims[2]+1,1))
        rows = np.where(vectors[:,:,istep][:,1] == sly)[0]
        u = np.reshape(vectors[:,:,istep][rows][:,3],(dims[0],dims[2]))
        v = np.reshape(vectors[:,:,istep][rows][:,5],(dims[0],dims[2]))
        return x,z,u,v
    if orientation == "yz":
        y,z = np.meshgrid(np.arange(1,dims[1]+1,1),np.arange(1,dims[2]+1,1))
        rows = np.where(vectors[:,:,istep][:,0] == slx)[0]
        u = np.reshape(vectors[:,:,istep][rows][:,4],(dims[1],dims[2]))
        v = np.reshape(vectors[:,:,istep][rows][:,5],(dims[1],dims[2]))
        return y,z,u,v

def plot_2d_vector_frame(vectors,frame,dims,slx,sly,slz,orientation):
    fig,ax = plt.subplots(1,1)
    x,y,u,v = get_2d_vecs(vectors,dims,frame-1,slx,sly,slz,orientation)
    quiver = ax.quiver(x,y,u,v,pivot='mid',headlength=0,headwidth=0,headaxislength=0,color="tab:blue",scale_units='xy',scale=0.25)
    ax.set_box_aspect(1)
    ax.set_adjustable("datalim")
    if orientation == "xy":
        ax.set_title("xy plane, z="+str(slz))
        ax.set_xlim(0,dims[0]+1)
        ax.set_ylim(0,dims[1]+1)
        ax.set_ylabel("y")
        ax.set_xlabel("x")
    if orientation == "xz":
        ax.set_title("xz plane, y="+str(sly))
        ax.set_xlim(0,dims[0]+1)
        ax.set_ylim(0,dims[2]+1)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
    if orientation == "yz":
        ax.set_title("yz plane, y="+str(slx))
        ax.set_xlim(0,dims[1]+1)
        ax.set_ylim(0,dims[2]+1)
        ax.set_xlabel("y")
        ax.set_ylabel("z")
    #fig.tight_layout()
    plt.show()


#################################################
# TIMESERIES FUNCTIONS (TORQUE/ENERGY)
#################################################


#################################################
# ARCHIVED/UNUSED FUNCTIONS
#################################################
def plot_3d_vec_animated(vectors,steps,N):
    def update_3d(i):
        global quiver
        #quiver.remove()
        ax.set_title("Frame="+str(i+1))
        quiver = ax.quiver(*get_3d_arrows(vectors,N,int(i)),length=1, normalize=True, pivot='middle',arrow_length_ratio=0.01)
        return quiver

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    #x,y,z,u,v,w = get_arrows(vectors,N,istep)
    ax.set_title("1")
    global quiver
    quiver = ax.quiver(*get_3d_arrows(vectors,N,0),length=1, normalize=True, pivot='middle',arrow_length_ratio=0.01)
    #ax.quiver(x, y, z, u, v, w, length=1, normalize=True, pivot='middle',arrow_length_ratio=0.01)
    ax.set_xlim(1,N+1)
    ax.set_ylim(1,N+1)
    ax.set_zlim(1,N+1)
    #frames = [(i/10)-1 for i in steps]

    ani = FuncAnimation(fig, update_3d, frames=[(i/10)-1 for i in steps], interval=50)
    plt.show()



def plot_all_2d_defects(vectors,defects,frame,N):
    fig,ax = plt.subplots(4,5,figsize=(20,8))
    ax = ax.ravel()
    for i in range(0,N):
        z = i+1
        x,y,u,v = get_2d_arrows(vectors,N,frame-1,z)
        xd = defects[np.where(defects[:,2]==z)][:,0]
        yd = defects[np.where(defects[:,2]==z)][:,1]
        ax[i].set_box_aspect(1)
        ax[i].set_adjustable("datalim")
        ax[i].set_title("z="+str(z))
        quiver = ax[i].quiver(x,y,u,v, pivot="mid",headlength=0,headwidth=0,headaxislength=0)
        ax[i].scatter(xd,yd,c="red",marker=".")
        ax[i].set_xlim(0,N+1)
        ax[i].set_ylim(0,N+1)

    fig.tight_layout()

def plot_all_2d_energy_frame(energies,frame,N):
    fig2,ax2 = plt.subplots(4,5,figsize=(20,8))
    ax2 = ax2.ravel()
    for i in range(0,N):
        z = i+1
        rows = np.where(vectors[frame-1][:,2]==z)
        E = energies[frame-1][rows[0]]
        E = np.reshape(E,(N,N))
        ax2[i].set_box_aspect(1)
        ax2[i].set_adjustable("datalim")
        ax2[i].set_title("E,z="+str(z))
        img = ax2[i].imshow(E,interpolation="bicubic")

    fig2.colorbar(img, ax=ax2.tolist())
    #fig2.tight_layout()
    #plt.show()

def plot_all_2d_vec_frame(vectors,frame,N):
    # load in defect data
    #with open("defects.csv") as f:
    #     defects = np.loadtxt(f)

    fig,ax = plt.subplots(4,5,figsize=(20,8))
    ax = ax.ravel()
    #E = energies[frame_flag-1]
    for i in range(0,N):
        z = i+1
        x,y,u,v = get_2d_arrows(vectors,N,frame-1,z)

        # getting defect data
        #xd = defects[np.where(defects[:,2]==z)][:,0]
        #yd = defects[np.where(defects[:,2]==z)][:,0]
        ax[i].set_box_aspect(1)
        ax[i].set_adjustable("datalim")
        ax[i].set_title("z="+str(z))
        #ax[i].imshow(E)
        quiver = ax[i].quiver(x,y,u,v,pivot='mid',headlength=0,headwidth=0,headaxislength=0)
        #quiver = ax[i].quiver(x,y,u,v,headlength=0.25,headwidth=0.25,headaxislength=0.25)
        #ax[i].scatter(xd,yd, c='red',marker='.')

        ax[i].set_xlim(0,N+1)
        ax[i].set_ylim(0,N+1)

    fig.tight_layout()
    #plt.show()

def plot_2d_vec_frame(vectors,frame,dims,z):
    fig,ax = plt.subplots(1,1)
    x,y,u,v = get_2d_arrows(vectors,dims,frame-1,z)
    quiver = ax.quiver(x,y,u,v,pivot='mid',headlength=0,headwidth=0,headaxislength=0,color="tab:blue")
    ax.set_box_aspect(1)
    ax.set_adjustable("datalim")
    ax.set_title("z="+str(z))
    ax.set_xlim(0,dims[1]+1)
    ax.set_ylim(0,dims[2]+1)
    fig.tight_layout()
    plt.show()

def plot_2d_energy_frame(vectors,energies,frame,dims,z):
    fig,ax=plt.subplots(1,1)
    rows = np.where(vectors[frame-1][:,2]==z)
    E = energies[frame-1][rows[0]]
    E = np.reshape(E,(dims[0],dims[1]))
    img = ax.imshow(E.T,interpolation="gaussian",extent=(1,dims[0],1,dims[1]))
    fig.colorbar(img)
    ax.set_box_aspect(1)
    ax.set_adjustable("datalim")
    ax.set_title("z="+str(z))
    ax.set_xlim(0,dims[0]+1)
    ax.set_ylim(0,dims[1]+1)
    fig.tight_layout()
    plt.show()

def plot_2d_all(vectors,energies,frame,dims,z,plt_vec_bool,plt_E_bool,plt_defect_bool):
    fig,ax=plt.subplots(1,1)
    if plt_vec_bool == True:
        x,y,u,v = get_2d_arrows(vectors,dims,frame-1,z)
        quiver = ax.quiver(x,y,u,v,pivot='mid',headlength=0,headwidth=0,headaxislength=0,color="tab:blue")
    if plt_E_bool == True:
        rows = np.where(vectors[frame-1][:,2]==z)
        E = energies[frame-1][rows[0]]
        E = np.reshape(E,(dims[0],dims[1]))
        img = ax.imshow(E,interpolation="gaussian",extent=(1,dims[0],1,dims[1]))
        fig.colorbar(img)#2.tolist())
    if plt_defect_bool == True:
        defects = load_defects("defects.csv")
        xd = defects[np.where(defects[:,2]==z)][:,0]
        yd = defects[np.where(defects[:,2]==z)][:,1]
        ax.scatter(xd,yd,c="red",marker=".")

    ax.set_box_aspect(1)
    ax.set_adjustable("datalim")
    ax.set_title("z="+str(z))
    ax.set_xlim(0,dims[0]+1)
    ax.set_ylim(0,dims[1]+1)
    fig.tight_layout()
    plt.show()
