#ll_sim_mgpu_module.jl
#mpiexecjl -np # julia ll_main.jl -t MGPU ./params.yml
__precompile__()
module ll_sim_mgpu_module
const dtype_gpu = Float32

using MPI
using MPIHaloArrays
using CUDA
using Random

import ..SimParams
import ..fill_vector  
import ..write_xyzv
import ..write_E
import ..write_T
import ..load_checkpoint
import ..create_checkpoint
import ..setup_env
import ..setup_defects

# include("./ll_sim_mgpu_kernels.jl")
# using .ll_sim_mgpu_kernels

export run_sim_mgpu

#KERNELS
function torque_thermo_gpu!(ld,tx,ty,tz,
    nx,ny,nz,
    E,
    fkick,periodic,K,CC)
    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    strx = blockDim().x * gridDim().x

    N = ld[1]
    i_offset,j_offset,k_offset = ld[2],ld[3],ld[4]
    Lx,Ly,Lz = ld[5],ld[6],ld[7]
    for index in idx:strx:N

        # convert 1d index to 3d reference
        i = mod1(index,Lx) # only returns (0,y]
        j = ceil(Int32,mod1((index/Lx),Ly))
        k = ceil(Int32, index/(Lx*Ly))

        #----- i+1,j,K
        # boundary conditions
        if periodic == true
        inab,jnab,knab = ifelse(i+1>Lx,1,i+1),j,k
        else
        inab,jnab,knab = ifelse(i+1>Lx,i,i+1),j,k
        end
        dp = nx[i,j,k]*nx[inab,jnab,knab] + ny[i,j,k]*ny[inab,jnab,knab]+nz[i,j,k]*nz[inab,jnab,knab]
        E[i,j,k] = E[i,j,k] + 1-(dp*dp)

        # thermostat
        costheta = 2.0 * rand() -1.0 #random float between -1,1
        theta = acos(costheta)
        phi = 2.0*pi*rand() # random float between 0,2π

        txbond = dp*(ny[i,j,k]*nz[inab,jnab,knab]-nz[i,j,k]*ny[inab,jnab,knab]) + sin(theta)*cos(phi)*randn()*fkick
        tybond = dp*(nz[i,j,k]*nx[inab,jnab,knab]-nx[i,j,k]*nz[inab,jnab,knab]) + sin(theta)*sin(phi)*randn()*fkick
        tzbond = dp*(nx[i,j,k]*ny[inab,jnab,knab]-ny[i,j,k]*nx[inab,jnab,knab]) + costheta*randn()*fkick
        CUDA.@atomic tx[i,j,k] = tx[i,j,k] + txbond
        CUDA.@atomic ty[i,j,k] = ty[i,j,k] + tybond
        CUDA.@atomic tz[i,j,k] = tz[i,j,k] + tzbond
        CUDA.@atomic tx[inab,jnab,knab] = tx[inab,jnab,knab] - txbond
        CUDA.@atomic ty[inab,jnab,knab] = ty[inab,jnab,knab] - tybond
        CUDA.@atomic tz[inab,jnab,knab] = tz[inab,jnab,knab] - tzbond

        #----- i,j+1,k
        # boundary conditions
        if periodic == true
        inab,jnab,knab = i,ifelse(j+1>Ly,1,j+1),k
        else
        inab,jnab,knab = i,ifelse(j+1>Ly,j,j+1),k
        end

        dp = nx[i,j,k]*nx[inab,jnab,knab] + ny[i,j,k]*ny[inab,jnab,knab]+nz[i,j,k]*nz[inab,jnab,knab]
        E[i,j,k] = E[i,j,k] + 1-(dp*dp)

        # thermostat
        costheta = 2.0 * rand() -1.0 #random float between -1,1
        theta = acos(costheta)
        phi = 2.0*pi*rand() # random float between 0,2π

        txbond = dp*(ny[i,j,k]*nz[inab,jnab,knab]-nz[i,j,k]*ny[inab,jnab,knab]) + sin(theta)*cos(phi)*randn()*fkick
        tybond = dp*(nz[i,j,k]*nx[inab,jnab,knab]-nx[i,j,k]*nz[inab,jnab,knab]) + sin(theta)*sin(phi)*randn()*fkick
        tzbond = dp*(nx[i,j,k]*ny[inab,jnab,knab]-ny[i,j,k]*nx[inab,jnab,knab]) + costheta*randn()*fkick
        CUDA.@atomic tx[i,j,k] = tx[i,j,k] + txbond
        CUDA.@atomic ty[i,j,k] = ty[i,j,k] + tybond
        CUDA.@atomic tz[i,j,k] = tz[i,j,k] + tzbond
        CUDA.@atomic tx[inab,jnab,knab] = tx[inab,jnab,knab] - txbond
        CUDA.@atomic ty[inab,jnab,knab] = ty[inab,jnab,knab] - tybond
        CUDA.@atomic tz[inab,jnab,knab] = tz[inab,jnab,knab] - tzbond

        #----- i,j,k+1
        # boundary conditions (if k+1 > Lz then dot product is of the same slice i.e. open boundary conditions)
        inab,jnab,knab = i,j,ifelse(k+1>Lz,k,k+1)
        dp = nx[i,j,k]*nx[inab,jnab,knab]+ny[i,j,k]*ny[inab,jnab,knab]+nz[i,j,k]*nz[inab,jnab,knab]
        E[i,j,k] = E[i,j,k] + 1-(dp*dp)

        # thermostat
        costheta = 2.0 * rand() -1.0 #random float between -1,1
        theta = acos(costheta)
        phi = 2.0*pi*rand() # random float between 0,2π

        txbond = dp*(ny[i,j,k]*nz[inab,jnab,knab]-nz[i,j,k]*ny[inab,jnab,knab]) + sin(theta)*cos(phi)*randn()*fkick
        tybond = dp*(nz[i,j,k]*nx[inab,jnab,knab]-nx[i,j,k]*nz[inab,jnab,knab]) + sin(theta)*sin(phi)*randn()*fkick
        tzbond = dp*(nx[i,j,k]*ny[inab,jnab,knab]-ny[i,j,k]*nx[inab,jnab,knab]) + costheta*randn()*fkick
        CUDA.@atomic tx[i,j,k] = tx[i,j,k] + txbond
        CUDA.@atomic ty[i,j,k] = ty[i,j,k] + tybond
        CUDA.@atomic tz[i,j,k] = tz[i,j,k] + tzbond
        CUDA.@atomic tx[inab,jnab,knab] = tx[inab,jnab,knab] - txbond
        CUDA.@atomic ty[inab,jnab,knab] = ty[inab,jnab,knab] - tybond
        CUDA.@atomic tz[inab,jnab,knab] = tz[inab,jnab,knab] - tzbond

        # scaling potential energy
        E[i,j,k] = 0.5*(K*CC)*E[i,j,k]
    end

    return nothing
end


function torque_gpu!(ld,tx,ty,tz,
    nx,ny,nz,
    E,
    periodic,K,CC)
    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    strx = blockDim().x * gridDim().x
    N = ld[1]
    i_offset,j_offset,k_offset = ld[2],ld[3],ld[4]
    Lx,Ly,Lz = ld[5],ld[6],ld[7]
    
    for index in idx:strx:N

        # convert 1d index to 3d reference
        i = mod1(index,Lx)+i_offset
        j = ceil(Int32,mod1((index/Lx),Ly))+j_offset
        k = ceil(Int32, index/(Lx*Ly))+k_offset

        #----- i+1,j,K
        # boundary conditions
        if periodic == true
        inab,jnab,knab = ifelse(i+1>Lx,1,i+1),j,k
        else 
        inab,jnab,knab = ifelse(i+1>Lx,i,i+1),j,k
        end
        dp = nx[i,j,k]*nx[inab,jnab,knab] + ny[i,j,k]*ny[inab,jnab,knab]+nz[i,j,k]*nz[inab,jnab,knab]
        E[i,j,k] = E[i,j,k] + 1-(dp*dp)

        # calculate the x,y,z components of the cross product between the same two vectors
        cx = ny[i,j,k]*nz[inab,jnab,knab]-nz[i,j,k]*ny[inab,jnab,knab]
        cy = nz[i,j,k]*nx[inab,jnab,knab]-nx[i,j,k]*nz[inab,jnab,knab]
        cz = nx[i,j,k]*ny[inab,jnab,knab]-ny[i,j,k]*nx[inab,jnab,knab]

        txbond = dp*cx
        tybond = dp*cy
        tzbond = dp*cz
        tx[i,j,k] = tx[i,j,k] + txbond
        ty[i,j,k] = ty[i,j,k] + tybond
        tz[i,j,k] = tz[i,j,k] + tzbond

        #----- i-1,j,K
        # boundary conditions
        if periodic == true
        inab,jnab,knab = ifelse(i-1<1,Lx,i+1),j,k
        else 
        inab,jnab,knab = ifelse(i-1<1,i,i+1),j,k
        end
        dp = nx[i,j,k]*nx[inab,jnab,knab] + ny[i,j,k]*ny[inab,jnab,knab]+nz[i,j,k]*nz[inab,jnab,knab]
        E[i,j,k] = E[i,j,k] + 1-(dp*dp)

        # calculate the x,y,z components of the cross product between the same two vectors
        cx = ny[i,j,k]*nz[inab,jnab,knab]-nz[i,j,k]*ny[inab,jnab,knab]
        cy = nz[i,j,k]*nx[inab,jnab,knab]-nx[i,j,k]*nz[inab,jnab,knab]
        cz = nx[i,j,k]*ny[inab,jnab,knab]-ny[i,j,k]*nx[inab,jnab,knab]

        txbond = dp*cx
        tybond = dp*cy
        tzbond = dp*cz
        tx[i,j,k] = tx[i,j,k] + txbond
        ty[i,j,k] = ty[i,j,k] + tybond
        tz[i,j,k] = tz[i,j,k] + tzbond

        #----- i,j+1,k
        # boundary conditions
        if periodic == true
        inab,jnab,knab = i,ifelse(j+1>Ly,1,j+1),k
        else
        inab,jnab,knab = i,ifelse(j+1>Ly,j,j+1),k 
        end
        dp = nx[i,j,k]*nx[inab,jnab,knab] + ny[i,j,k]*ny[inab,jnab,knab]+nz[i,j,k]*nz[inab,jnab,knab]
        E[i,j,k] = E[i,j,k] + 1-(dp*dp)

        # calculate the x,y,z components of the cross product between the same two vectors
        cx = ny[i,j,k]*nz[inab,jnab,knab]-nz[i,j,k]*ny[inab,jnab,knab]
        cy = nz[i,j,k]*nx[inab,jnab,knab]-nx[i,j,k]*nz[inab,jnab,knab]
        cz = nx[i,j,k]*ny[inab,jnab,knab]-ny[i,j,k]*nx[inab,jnab,knab]

        txbond = dp*cx
        tybond = dp*cy
        tzbond = dp*cz
        tx[i,j,k] = tx[i,j,k] + txbond
        ty[i,j,k] = ty[i,j,k] + tybond
        tz[i,j,k] = tz[i,j,k] + tzbond

        #----- i,j-1,k
        # boundary conditions
        if periodic == true
        inab,jnab,knab = i,ifelse(j+1<1,Ly,j+1),k
        else
        inab,jnab,knab = i,ifelse(j-1<1,j,j+1),k 
        end
        dp = nx[i,j,k]*nx[inab,jnab,knab] + ny[i,j,k]*ny[inab,jnab,knab]+nz[i,j,k]*nz[inab,jnab,knab]
        E[i,j,k] = E[i,j,k] + 1-(dp*dp)

        # calculate the x,y,z components of the cross product between the same two vectors
        cx = ny[i,j,k]*nz[inab,jnab,knab]-nz[i,j,k]*ny[inab,jnab,knab]
        cy = nz[i,j,k]*nx[inab,jnab,knab]-nx[i,j,k]*nz[inab,jnab,knab]
        cz = nx[i,j,k]*ny[inab,jnab,knab]-ny[i,j,k]*nx[inab,jnab,knab]

        txbond = dp*cx
        tybond = dp*cy
        tzbond = dp*cz
        tx[i,j,k] = tx[i,j,k] + txbond
        ty[i,j,k] = ty[i,j,k] + tybond
        tz[i,j,k] = tz[i,j,k] + tzbond

        #----- i,j,k+1
        # boundary conditions (if k+1 > Lz then dot product is of the same slice i.e. open boundary conditions)
        inab,jnab,knab = i,j,ifelse(k+1>Lz,1,k+1)
        if k == Lz
        # top slice energy should not depend on bottom slice energy
        dp = nx[i,j,k]*nx[inab,jnab,k] + ny[i,j,k] * ny[inab,jnab,k] + nz[i,j,k]*nz[inab,jnab,k]
        else
        dp = nx[i,j,k]*nx[inab,jnab,knab]+ny[i,j,k]*ny[inab,jnab,knab]+nz[i,j,k]*nz[inab,jnab,knab]
        end
        E[i,j,k] = E[i,j,k] + 1-(dp*dp)

        # calculate the x,y,z components of the cross product between the same two vectors
        cx = ny[i,j,k]*nz[inab,jnab,knab]-nz[i,j,k]*ny[inab,jnab,knab]
        cy = nz[i,j,k]*nx[inab,jnab,knab]-nx[i,j,k]*nz[inab,jnab,knab]
        cz = nx[i,j,k]*ny[inab,jnab,knab]-ny[i,j,k]*nx[inab,jnab,knab]

        txbond = dp*cx
        tybond = dp*cy
        tzbond = dp*cz
        tx[i,j,k] = tx[i,j,k] + txbond
        ty[i,j,k] = ty[i,j,k] + tybond
        tz[i,j,k] = tz[i,j,k] + tzbond

        #----- i,j,k-1
        # boundary conditions (if k+1 > Lz then dot product is of the same slice i.e. open boundary conditions)
        inab,jnab,knab = i,j,ifelse(k-1<1,Lz,k+1)
        if k == Lz
        # top slice energy should not depend on bottom slice energy
        dp = nx[i,j,k]*nx[inab,jnab,k] + ny[i,j,k] * ny[inab,jnab,k] + nz[i,j,k]*nz[inab,jnab,k]
        else
        dp = nx[i,j,k]*nx[inab,jnab,knab]+ny[i,j,k]*ny[inab,jnab,knab]+nz[i,j,k]*nz[inab,jnab,knab]
        end
        E[i,j,k] = E[i,j,k] + 1-(dp*dp)

        # calculate the x,y,z components of the cross product between the same two vectors
        cx = ny[i,j,k]*nz[inab,jnab,knab]-nz[i,j,k]*ny[inab,jnab,knab]
        cy = nz[i,j,k]*nx[inab,jnab,knab]-nx[i,j,k]*nz[inab,jnab,knab]
        cz = nx[i,j,k]*ny[inab,jnab,knab]-ny[i,j,k]*nx[inab,jnab,knab]

        txbond = dp*cx
        tybond = dp*cy
        tzbond = dp*cz
        tx[i,j,k] = tx[i,j,k] + txbond
        ty[i,j,k] = ty[i,j,k] + tybond
        tz[i,j,k] = tz[i,j,k] + tzbond

        # scaling potential energy
        E[i,j,k] = 0.5*(K*CC)*E[i,j,k]
    end
    return nothing
end

function position_gpu!(ld,CC,dt,nx,ny,nz,tx,ty,tz,KE)
    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    strx = blockDim().x * gridDim().x

    N = ld[1]
    i_offset,j_offset,k_offset = ld[2],ld[3],ld[4]
    Lx,Ly,Lz = ld[5],ld[6],ld[7]
    for index in idx:strx:N
        i = mod1(index,Lx)+i_offset
        j = ceil(Int32,mod1((index/Lx),Ly))+j_offset
        k = ceil(Int32, index/(Lx*Ly))+k_offset
        wx = CC*tx[i,j,k]
        wy = CC*ty[i,j,k]
        wz = CC*tz[i,j,k]

        wXnx = wy*nz[i,j,k]
        wXny = wz*nx[i,j,k]
        wXnz = wx*ny[i,j,k]

        #update directors
        nx[i,j,k] = nx[i,j,k]+wXnx*dt
        ny[i,j,k] = ny[i,j,k]+wXny*dt
        nz[i,j,k] = nz[i,j,k]+wXnz*dt
    end
    return nothing
end

function normalize_gpu!(ld,nx,ny,nz)
    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    N = ld[1]
    i_offset,j_offset,k_offset = ld[2],ld[3],ld[4]
    Lx,Ly,Lz = ld[5],ld[6],ld[7]

    for index in idx:stride:N
        i = mod1(index, Lx)+i_offset
        j = ceil(Int32,mod1(index/Lx,Ly))+j_offset
        k = ceil(Int32,index/(Lx*Ly))+k_offset
        s = sqrt(nx[i,j,k]*nx[i,j,k] + ny[i,j,k]*ny[i,j,k] + nz[i,j,k]*nz[i,j,k])
        nx[i,j,k] = nx[i,j,k]/s
        ny[i,j,k] = ny[i,j,k]/s
        nz[i,j,k] = nz[i,j,k]/s
    end
    return nothing
end

# FUNCTIONS
function gen_MPI_topo(nprocs,nhalo)
    if nprocs == 4
        #create toplogy
        topo = CartesianTopology(comm, [2,2,1], [false, false,false])
        @assert length(CUDA.devices()) == nprocs "# GPUS != #CPUS"
        gpus = [i for i in 0:nprocs-1]
        for p in 0:nprocs-1
            if rank == p
                CUDA.device!(p)
                println("$(CUDA.name(device())) assigned to $(rank)")
            end
        end
    elseif nprocs == 2
        topo = CartesianTopology(comm, [2,1,1], [false,false,false])
        @assert length(CUDA.devices()) == nprocs "# GPUS != #CPUS"
        gpus = [i for i in 0:nprocs-1]
        for p in 0:nprocs-1
            if rank == p
                CUDA.device!(p)
                println("$(CUDA.name(device())) assigned to $(rank)")
            end
        end
    else
        GC.gc()
        MPI.Finalize()
        @error "unexpected number of processes for topology splitting, try 2 or 4"
        exit()
    end
    return topo,gpus
end

function cuda_info()
    @info "MGPU, MPI+CUDA"
    if CUDA.functional()
        @info string("CUDA version: ",CUDA.version())
        for (itr,dev) in enumerate(CUDA.devices())
            @info string(itr,". ",CUDA.name(device()))
        end
        @info string("CUDA-aware MPI: $(MPI.has_cuda())")
    else
        @warn "CUDA not available"
    end
end

function print_cuda_mem_stats(proc)
    if rank == proc
         println("$(proc): $(CUDA.memory_status())\n")
    end
end

function progress(params,istep,pbar_counter,elapsed_sum,elapsed_counter)
    # getting average iteration time
    avg_it = lpad(round(elapsed_sum / elapsed_counter, digits=3),5)
    eta = lpad(Int(ceil(((elapsed_sum/elapsed_counter)*(params.nsteps-istep))/60)),4)
    # calculating average memory throughput
    # nx,ny,nz are read + written (2*3), tx,ty,tz are read + written (2*3), tx_inabs, ... , E are written to (9+1), tx,tz,tz,tx_inabs,...E are written to (13) (zeros)
    T_eff = lpad(round(((2*3+2*3+9+1+13)*1/1e9*params.dimensions[1]*params.dimensions[2]*params.dimensions[3]*sizeof(dtype_gpu))/(elapsed_sum/elapsed_counter), digits = 4),6)
    # calculating VRAM usage
    used_mem = Base.format_bytes(CUDA.total_memory()-CUDA.available_memory())
    total_mem = Base.format_bytes(CUDA.total_memory())
    mem_ratio = round(100*((CUDA.total_memory()-CUDA.available_memory())/CUDA.total_memory()), digits=3)
    println(istep, ' '^(length(digits(params.nsteps))-length(digits(istep))), " (",' '^(3-length(digits(floor(Int,istep/params.nsteps*100)))), istep/params.nsteps * 100,"%) ", '='^(2*pbar_counter), ' '^(20-2*pbar_counter), " Avg. iter: $avg_it s, ETA: $eta min, Mem. throughput: $T_eff GB/s , VRAM usage: $used_mem/$total_mem ($mem_ratio%)")
    pbar_counter += 1
    elapsed_sum = 0
    return elapsed_sum, pbar_counter
end

function run_sim_mgpu(params::SimParams;save::Bool=true)
    #MPI Initialization
    MPI.Init()
    global comm = MPI.COMM_WORLD
    global rank = MPI.Comm_rank(comm)
    global nprocs = MPI.Comm_size(comm)
    global root = 0 # root rank
    if rank == root
        @info "run_sim_mgpu"
        cuda_info()
        setup_env(save) #setting up the data folder
    end

    #MPI Topology
    @assert params.periodic == false "no periodic yet"
    nhalo = 1
    topology,gpus = gen_MPI_topo(nprocs,nhalo)

    # SETUP
    # unpacking and generating data on the CPU
    CC = params.CC 
    dt = params.dt
    esave = ceil(params.isave/10)
    N3 = params.dimensions[1]*params.dimensions[2]*params.dimensions[3]
    total_defects = sum(size(params.defects["ndefects"]))

    phi = fill_vector(params.phi)
    kbt = fill_vector(params.kbt)
    fkick = sqrt.(8*kbt*params.dt)
    # ni,nj,nk are the size of the local array on each MPI process
    ni = ifelse(topology.global_dims[1]>1,div(params.dimensions[1],topology.global_dims[1],RoundNearest),params.dimensions[1])
    nj = ifelse(topology.global_dims[2]>1,div(params.dimensions[2],topology.global_dims[2],RoundNearest),params.dimensions[2])
    nk = ifelse(topology.global_dims[3]>1,div(params.dimensions[3],topology.global_dims[3],RoundNearest),params.dimensions[3])

    # DATA INITIALIZATION
    #Random.seed!(987654321)
    if !isempty(params.loadckpt) # loading from a checkpoint
        ckpt_params,nx,ny,nz,E = load_checkpoint(params.loadckpt)
        if nfields(params) != nfields(ckpt_params) && rank == root
            println(propertynames(ckpt_params))
            @error "Checkpoint params and user-defined params mismatch"
        end
    else
        nx_g = rand(dtype_gpu, params.dimensions).-convert(dtype_gpu,0.5)
        ny_g = rand(dtype_gpu, params.dimensions).-convert(dtype_gpu,0.5)
        nz_g = rand(dtype_gpu, params.dimensions).-convert(dtype_gpu,0.5)
    end
    if total_defects > 0
        nx_g,ny_g,nz_g = setup_defects(params,nx_g,ny_g,nz_g)
        #nx,ny,nz = setup_lines(params,nx,ny,nz)
        s = sqrt.(nx_g.*nx_g .+ ny_g.*ny_g .+ nz_g.*nz_g)
        nx_g = nx_g./s
        ny_g = ny_g./s
        nz_g = nz_g./s
        nxb_g,nyb_g,nzb_g = nx_g[:,:,1:2],ny_g[:,:,1:2],nz_g[:,:,1:2] #setting original defect structure
        #note: we have to preserve these as 3-dimensional arrays due to the way MPIHaloArrays scatterglobal works
    else
        # if no defects specified, make bottom substrate planar aligned
        nyb_g = fill(1.0,(params.dimensions[1],params.dimensions[2]))
        nzb_g,nxb_g = zeros((params.dimensions[1],params.dimensions[2])),zeros((params.dimensions[1],params.dimensions[2]))
    end
    
    tx_g = zeros(dtype_gpu,params.dimensions) 
    ty_g = zeros(dtype_gpu,params.dimensions)
    tz_g = zeros(dtype_gpu,params.dimensions)
    E_g = zeros(dtype_gpu,params.dimensions)
    KE_g = zeros(dtype_gpu,params.dimensions)

    # SCATTER DATA
    nx = scatterglobal(nx_g,root, nhalo, topology; do_corners = true)
    ny = scatterglobal(ny_g,root, nhalo, topology; do_corners = true)
    nz = scatterglobal(nz_g,root, nhalo, topology; do_corners = true)

    tx = scatterglobal(tx_g,root, nhalo, topology; do_corners = true)
    ty = scatterglobal(ty_g,root, nhalo, topology; do_corners = true)
    tz = scatterglobal(tz_g,root, nhalo, topology; do_corners = true)

    E = scatterglobal(E_g,root, nhalo, topology; do_corners = true)
    KE = scatterglobal(KE_g,root, nhalo, topology; do_corners = true)

    nxb = scatterglobal(nxb_g,root, nhalo, topology; do_corners = true)
    nyb = scatterglobal(nyb_g,root, nhalo, topology; do_corners = true)
    nzb = scatterglobal(nzb_g,root, nhalo, topology; do_corners = true)
    ilo,ihi,jlo,jhi,klo,khi = local_domain_indices(nx)
    iglo,ighi,jglo,jghi,kglo,kghi = global_domain_indices(nx)
    # CUDA LOADING SCALARS
        #ld is an array containing local domain data
        #ld[1] = ni*nj*nk
        #ld[2] -> ld[4] contains i_offset,j_offset,k_offsest
        #ld[5-7] contains Lx,Ly,Lz which are local array sizes
        #ld[8-12] contains ilo,ihi,jlo,jhi,klo,khi
    ld = cu([ni*nj*nk,ilo-1,jlo-1,klo-1,ihi-ilo+1,jhi-jlo+1,khi-klo+1,ilo,ihi,jlo,jhi,klo,khi])
    periodic_d = cu(params.periodic)
    CC_d = cu(CC)
    K_d = cu(params.K)
    dt_d = cu(dt)

    #CUDA LOADING ARRAYS
    nx_d = CuArray(nx.data)
    ny_d = CuArray(ny.data)
    nz_d = CuArray(nz.data)
    tx_d = CuArray(tx.data)
    ty_d = CuArray(ty.data)
    tz_d = CuArray(tz.data)
    E_d = CuArray(E.data)
    KE_d = CuArray(KE.data)
    nxb_d,nyb_d,nzb_d = CuArray(nxb.data),CuArray(nyb.data),CuArray(nzb.data)
    for p in 0:nprocs-1
        print_cuda_mem_stats(p)
    end
	
	# CUDA PRE-COMPILING KERNELS
    ctime = CUDA.@elapsed begin
        #compiling and optimizing GPU kernels
        # note, just passing dummy/empty data since launch = false
        pos_kernel = @cuda launch=false position_gpu!(ld,CC_d,dt_d,nx_d,ny_d,nz_d,tx_d,ty_d,tz_d,KE_d)
        pos_config = launch_configuration(pos_kernel.fun)
        println("$(rank): position_gpu! kernel compiled successfully $(pos_config)")
        pos_threads = pos_config.threads
        pos_blocks = pos_config.blocks
        
        norm_kernel = @cuda launch=false normalize_gpu!(ld,nx_d,ny_d,nz_d)
        norm_config = launch_configuration(norm_kernel.fun)
        println("$(rank): normalize_gpu! kernel compiled successfully $(norm_config)")
        norm_threads = norm_config.threads 
        norm_blocks = norm_config.blocks

        if params.thermobool == true
            fkick_d = cu(fkick[1])
            ttorque_kernel = @cuda launch=false torque_thermo_gpu!(ld,tx_d,ty_d,tz_d,nx_d,ny_d,nz_d,E_d,fkick_d,periodic_d,K_d,CC_d)
            ttorque_config = launch_configuration(ttorque_kernel.fun)
            println("$(rank): torque_thermo_gpu! kernel compiled successfully $(ttorque_config)")
            ttorque_threads = ttorque_config.threads
            ttorque_blocks = ttorque_config.blocks
        else
            torque_kernel = @cuda launch=false torque_gpu!(ld,tx_d,ty_d,tz_d,nx_d,ny_d,nz_d,E_d,periodic_d,K_d,CC_d)
            #torque_kernel = @cuda launch=false torque_gpu!(tx_d,ty_d,tz_d,nx_d,ny_d,nz_d,E_d)
            torque_config = launch_configuration(torque_kernel.fun)
            println("$(rank): torque_gpu! kernel compiled successfully $(torque_config)")
            torque_threads = torque_config.threads
            torque_blocks = torque_config.blocks
        end
    end
    for p in 0:nprocs-1
        if rank == p
            println("$(rank): Compilation time: $ctime s")
        end
    end
    #normalize the positions before first step
    CUDA.@sync @cuda threads=norm_config.threads blocks=norm_config.blocks normalize_gpu!(ld,nx_d,ny_d,nz_d)
    # fill halo of position arrays before first step

    pbar = [params.nsteps/100; [i*params.nsteps/10 for i in 1:10]] 
    pbar_counter = 0; elapsed_sum = 0; elapsed_counter = 0

    MPI.Barrier(comm)
    # MAIN SIMULATION LOOP
    for istep in 1:params.nsteps
            elapsed_sum += CUDA.@elapsed begin
            elapsed_counter += 1
        if rank == root
            #progress update and statistics
            if istep % pbar[pbar_counter+1] == 0
                elapsed_sum,pbar_counter = progress(params,istep,pbar_counter,elapsed_sum,elapsed_counter)
            end
        end
        
        #updating torques
        if params.thermobool == true
            fkick_d = cu(fkick[istep])
            CUDA.@sync @cuda threads=ttorque_threads blocks=ttorque_blocks torque_thermo_gpu!(ld,tx_d,ty_d,tz_d,nx_d,ny_d,nz_d,E_d,fkick_d,periodic_d,K_d,CC_d)
        else
            # running the torque gpu kernel without a thermostat
            CUDA.@sync @cuda threads=torque_threads blocks=torque_blocks torque_gpu!(ld,tx_d,ty_d,tz_d,nx_d,ny_d,nz_d,E_d,periodic_d,K_d,CC_d)
        end

        # torque halo updates, swap nx_d to nx.data and then update_halo! then transfer back to gpu
        tx.data = Array(tx_d)
        ty.data = Array(ty_d)
        tz.data = Array(tz_d)
        updatehalo!(tx)
        updatehalo!(ty)
        updatehalo!(tz)
        tx_d = CuArray(tx.data)
        ty_d = CuArray(ty.data)
        tz_d = CuArray(tz.data)

        # updating positions
        CUDA.@sync @cuda threads=pos_threads blocks=pos_blocks position_gpu!(ld,CC_d,dt_d,nx_d,ny_d,nz_d,tx_d,ty_d,tz_d,KE_d)
                
        # position halo update
        nx.data = Array(nx_d)
        ny.data = Array(ny_d)
        nz.data = Array(nz_d)
        updatehalo!(nx)
        updatehalo!(ny)
        updatehalo!(nz)
        nx_d = CuArray(nx.data)
        ny_d = CuArray(ny.data)
        nz_d = CuArray(nz.data)

        #if total_defects > 1
        # resetting the defects on the bottom substrate
            nx_d[:,:,1] = nxb_d[:,:,1]
            ny_d[:,:,1] = nyb_d[:,:,1]
            nz_d[:,:,1] = nzb_d[:,:,1]
            # setting the top substrate with an anchoring conditions
            nx_d[:,:,params.dimensions[3]] .= cos(phi[istep])
            ny_d[:,:,params.dimensions[3]] .= sin(phi[istep])
            nz_d[:,:,params.dimensions[3]] .= 0
        #end 

        #renormalizing
        CUDA.@sync @cuda threads=norm_threads blocks=norm_blocks normalize_gpu!(ld,nx_d,ny_d,nz_d)

        #writing out the data
        if save==true
            #MPI.Barrier(comm) # syncs processes
            #gathering nx,ny and nz from all the MPI workers onto root process            
            if istep%esave == 0
                #writing energies to file
                E.data = Array(E_d)
                KE.data = Array(KE_d)
                E_out = gatherglobal(E)
                KE_out = gatherglobal(Ke)
                if rank == root
                e_filename = string("energy_"*params.simname*"_gpu.csv")
                write_E(e_filename,istep,sum(E_out)/N3,sum(KE_out)/N3) # TODO is summing on GPU faster?

                # writing torque on top substrate and angles out
                t_filename = string("torque_"*params.simname*"_gpu.csv")
                avg_T = sum(tx_out[:,:,params.dimensions[3]] .+ ty_out[:,:,params.dimensions[3]] .+ tz_out[:,:,params.dimensions[3]])/(params.dimensions[1]*params.dimensions[2])
                write_T(t_filename,istep,phi[istep],avg_T)
                end
            end
            if istep%params.isave == 0
                nx.data = Array(nx_d)
                ny.data = Array(ny_d)
                nz.data = Array(nz_d)
                nx_out = gatherglobal(nx)
                ny_out = gatherglobal(ny)
                nz_out = gatherglobal(nz)
                if !@isdefined E_out
                    E.data = Array(E_d)
                    E_out = gatherglobal(E)
                end
                if rank == root
                    write_xyzv(Float16,string("vectors_"*params.simname*"_gpu.xyzv"),params,istep,nx_out,ny_out,nz_out,E_out)
                end
                    #write_array_data(params,istep,Array(nx_d),Array(ny_d),Array(nz_d),Array(E_d))
            end
        end

        #saving checkpoints
        if !iszero(params.checkpoints) && istep in params.checkpoints
            if istep%params.isave == 0
                if rank == root
                    create_checkpoint(params,istep,nx_out,ny_out,nz_out,E_out)
                end
            else
                #gather nx,ny and nz from all the MPI workers onto root process
                nx.data = Array(nx_d)
                ny.data = Array(ny_d)
                nz.data = Array(nz_d)
                nx_out = gatherglobal(nx)
                ny_out = gatherglobal(ny)
                nz_out = gatherglobal(nz)
                if rank == root
                    create_checkpoint(params,istep,nx_out,ny_out,nz_out,E_out)
                end
            end
        end

        # zeroing out the torques
        tx_d .=0 # = CUDA.zeros(params.dimensions)
        ty_d .=0 # = CUDA.zeros(params.dimensions)
        tz_d .=0 # = CUDA.zeros(params.dimensions)
        E_d .=0 # = CUDA.zeros(params.dimensions)
        KE_d .=0
        end #elapsed
    end #steps
    return nothing
end
end

