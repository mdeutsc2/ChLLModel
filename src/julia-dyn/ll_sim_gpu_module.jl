#ll_sim_gpu_module.jl
__precompile__()
module ll_sim_gpu_module

const dtype_gpu = Float32

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
export run_sim_gpu

function __init__()
    @info "GPU"
    if CUDA.functional()
        @info string("CUDA version: ",CUDA.version())
        @info string("1.",CUDA.name(device()))
    else
        @warn "CUDA not available"
    end
end


function torque_thermo_gpu!(tx,ty,tz,
                            nx,ny,nz,
                            E,
                            fkick,periodic)
    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    strx = blockDim().x * gridDim().x

    Lx,Ly,Lz = size(nx)
    for index in idx:strx:length(nx)

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
    end

    return nothing
end


function torque_gpu!(tx,ty,tz,
                    nx,ny,nz,
                    E,
                    periodic)
    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    strx = blockDim().x * gridDim().x

    Lx,Ly,Lz = size(nx)
    for index in idx:strx:length(nx)

        # convert 1d index to 3d reference
        i = mod1(index,Lx)
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
    end
    return nothing
end

function position_gpu!(CC,dt,nx,ny,nz,tx,ty,tz,KE)
    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x

    strx = blockDim().x * gridDim().x

    #Lx,Ly,Lz = size(nx)
    for i in idx:strx:length(nx)
        # # convert 1d index to 3d reference
        # i = mod1(idx,Lx)
        # j = ceil(Int32,mod1((idx/Lx),Ly))
        # k = ceil(Int32, idx/(Lx*Ly))
        # angular velocity is proportional to torque
        # wx = CC*tx[i]
        # wy = CC*ty[i]
        # wz = CC*tz[i]

        # calculate w cross n, component by component
        #calculate w cross n, component by component
        wXnx = CC*ty[i]*nz[i]
        wXny = CC*tz[i]*nx[i]
        wXnz = CC*tx[i]*ny[i]

        # calculate kinetic energy
        KE[i] = 0.5*(wXnx*wXnx + wXny*wXny + wXnz*wXnz)
        #update directors
        nx[i] = nx[i]+wXnx*dt
        ny[i] = ny[i]+wXny*dt
        nz[i] = nz[i]+wXnz*dt

    end
    #for index in idx:strx:length(nx)
        # i = mod1(index,Lx)
        # j = ceil(Int32,mod1((index/Lx),Ly))
        # k = ceil(Int32, index/(Lx*Ly))
        # wx = CC*tx[i,j,k]
        # wy = CC*ty[i,j,k]
        # wz = CC*tz[i,j,k]

        # wXnx = wy*nz[i,j,k]
        # wXny = wz*nx[i,j,k]
        # wXnz = wx*ny[i,j,k]

        # #update directors
        # nx[i,j,k] = nx[i,j,k]+wXnx*dt
        # ny[i,j,k] = ny[i,j,k]+wXny*dt
        # nz[i,j,k] = nz[i,j,k]+wXnz*dt
    # end
    return nothing
end

function normalize_gpu!(nx,ny,nz)
    index = (blockIdx().x-1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i in index:stride:length(nx)
        s = sqrt(nx[i]*nx[i] + ny[i]*ny[i] + nz[i]*nz[i])
        nx[i] = nx[i]/s
        ny[i] = ny[i]/s
        nz[i] = nz[i]/s
    end
    return nothing
end

function setup_lines(params::SimParams,nx::Array{dtype_gpu,3},ny::Array{dtype_gpu,3},nz::Array{dtype_gpu,3})::Tuple{Array{dtype_gpu, 3}, Array{dtype_gpu, 3}, Array{dtype_gpu, 3}}
    @assert params.dimensions[1] == params.dimensions[2] string("xy dimension mismatch: ",params.dimensions[1],"!=",params.dimensions[2])
    k = 1 # for the bottom substrate
    L = 20 # period of the director stripe patter from Babakhanova et. al. 2020
    for i = 1:params.dimensions[1]
        for j in 1:params.dimensions[2]
            theta = (pi * i )/ L
            nx[i,j,k] = cos(theta)
            ny[i,j,k] = sin(theta)
            nz[i,j,k] = 0
        end
    end
    return nx,ny,nz
end

function setup_defects(params::SimParams,nx::Array{dtype_gpu,3},ny::Array{dtype_gpu,3},nz::Array{dtype_gpu,3})::Tuple{Array{dtype_gpu, 3}, Array{dtype_gpu, 3}, Array{dtype_gpu, 3}}
    @assert params.dimensions[1] == params.dimensions[2] string("xy dimension mismatch: ",params.dimensions[1],"!=",params.dimensions[2])
    println("Defects:")
    ndefects = params.defects["ndefects"]
    if length(size(ndefects)) == 1
        ydefects = 1
        xdefects = size(ndefects)[1]
    else
        ydefects = size(ndefects)[2]
        xdefects = size(ndefects)[1]
    end
    spacing_string = split(params.defects["spacing"],",")
    spacing = (parse(Int,spacing_string[1][2:end]),parse(Int,spacing_string[2][1]))
    #spacing = params.defects["spacing"]
    th = zeros(params.dimensions[1],params.dimensions[2])
    xx = zeros((xdefects,ydefects))
    yy = zeros((xdefects,ydefects))
    q = zeros((xdefects,ydefects))
    
    if sum(size(ndefects)) > 1
        id = 0
        for ii in 1:xdefects
            for jj in 1:ydefects
                id += 1
                if !isempty(spacing) # test if scaling parameters are specified
                    if spacing[1] != 0
                        xx[id] = ii*spacing[1] + 0.5
                        xx[id] = xx[id] +(1-(spacing[1]*(xdefects+1))/params.dimensions[1])*(params.dimensions[1]/2)
                    else
                        xx[id] = ii*(params.dimensions[1]/(xdefects+1))+0.5
                    end
                    if spacing[2] != 0 
                        yy[id] = jj*spacing[2] + 0.5
                        yy[id] = yy[id] + (1-(spacing[2](ydefects+1))/params.dimensions[2])*(params.dimensions[2]/2)
                    else
                        yy[id] = jj*(params.dimensions[2]/(ydefects+1))+0.5
                    end
                else 
                    xx[id] = ii*(params.dimensions[1]/(xdefects+1))+0.5
                    yy[id] = jj*(params.dimensions[2]/(ydefects+1))+0.5
                end
                q[id] = ndefects[ii,jj]
                # if sum(ndefects) == 4
                #     if id == 2 || id == 3
                #         q[id] = -0.5
                #     end
                # elseif sum(ndefects) != 4
                #     if id%2 == 0
                #         q[id] = -0.5
                #     end
                # end
                println("\t",xx[id],"\t",yy[id],"\t",q[id])
            end
        end

    else
        id = 1
        xx[1] = params.dimensions[1]/2 +0.5
        yy[1] = params.dimensions[1]/2 +0.5
        q[1] = 0.5
    end
    """q = [0.5,-0.5,0.5,-0.5,
        1.0,-1.0,1.0,-1.0,
        -1.0,1.0,-1.0,1.0,
        -0.5,0.5,-0.5,0.5]"""

    for idefect = 1:id
        for i = 1:params.dimensions[1] 
            for j = 1:params.dimensions[2]
                phi = atan(j-yy[idefect],i-xx[idefect])
                th[i,j] += q[idefect]*phi + pi/4.0
            end
        end
    end
    k = 1 #for the bottom substrate
    for i = 1:params.dimensions[1]
        for j = 1:params.dimensions[2] 
            nx[i,j,k] = 0.5*cos(th[i,j])
            ny[i,j,k] = 0.5*sin(th[i,j])
            nz[i,j,k] = 0
        end
    end
    return nx,ny,nz
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

function run_sim_gpu(params::SimParams;save::Bool=true)
    @info "run_sim_gpu"
    setup_env(save) #setting up the data folder
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

    println(CUDA.memory_status())

    #Random.seed!(987654321)
    if !isempty(params.loadckpt) # loading from a checkpoint
        ckpt_params,nx,ny,nz,E = load_checkpoint(params.loadckpt)
        if nfields(params) != nfields(ckpt_params)
            println(propertynames(ckpt_params))
            @error "Checkpoint params and user-defined params mismatch"
        end
    else
        nx = rand(dtype_gpu, params.dimensions).-convert(dtype_gpu,0.5)
        ny = rand(dtype_gpu, params.dimensions).-convert(dtype_gpu,0.5)
        nz = rand(dtype_gpu, params.dimensions).-convert(dtype_gpu,0.5)
    end
    if total_defects > 0
        nx,ny,nz = setup_defects(params,nx,ny,nz)
        #nx,ny,nz = setup_lines(params,nx,ny,nz)
        s = sqrt.(nx.*nx .+ ny.*ny .+ nz.*nz)
        nx = nx./s
        ny = ny./s
        nz = nz./s
        nxb,nyb,nzb = nx[:,:,1],ny[:,:,1],nz[:,:,1] #setting original defect structure
        nxb_d,nyb_d,nzb_d = CuArray(nxb),CuArray(nyb),CuArray(nzb)
    else
        # if no defects specified, make bottom substrate planar aligned
        nyb = fill(1.0,(params.dimensions[1],params.dimensions[2]))
        nzb,nxb = zeros((params.dimensions[1],params.dimensions[2])),zeros((params.dimensions[1],params.dimensions[2]))
        nxb_d,nyb_d,nzb_d = CuArray(nxb),CuArray(nyb),CuArray(nzb)
    end
    
    tx = zeros(dtype_gpu,params.dimensions) 
    ty = deepcopy(tx)
    tz = deepcopy(tx)
    E = zeros(dtype_gpu,params.dimensions)
    KE = zeros(dtype_gpu,params.dimensions)

    # loading the data onto the GPU
    periodic_d = cu(params.periodic)
    #N_d = cu(N)
    CC_d = cu(CC)
    K_d = cu(params.K)
    dt_d = cu(dt)
    nx_d = CuArray(nx)
    ny_d = CuArray(ny)
    nz_d = CuArray(nz)
    tx_d = CuArray(tx)
    ty_d = CuArray(ty)
    tz_d = CuArray(tz)
    E_d = CuArray(E)
    KE_d = CuArray(KE)
  
    println(CUDA.memory_status())
    ctime = CUDA.@elapsed begin
        #compiling and optimizing GPU kernels
        # note, just passing dummy/empty data since launch = false
        pos_kernel = @cuda launch=false position_gpu!(CC_d,dt_d,nx_d,ny_d,nz_d,tx_d,ty_d,tz_d,KE_d)
        pos_config = launch_configuration(pos_kernel.fun)
        println("position_gpu! kernel compiled successfully ", pos_config)
        pos_threads = pos_config.threads
        pos_blocks = pos_config.blocks
        
        norm_kernel = @cuda launch=false normalize_gpu!(nx_d,ny_d,nz_d)
        norm_config = launch_configuration(norm_kernel.fun)
        println("normalize_gpu! kernel compiled successfully ", norm_config)
        norm_threads = norm_config.threads 
        norm_blocks = norm_config.blocks

        if params.thermobool == true
            fkick_d = cu(fkick[1])
            ttorque_kernel = @cuda launch=false torque_thermo_gpu!(tx_d,ty_d,tz_d,nx_d,ny_d,nz_d,E_d,fkick_d,periodic_d)
            ttorque_config = launch_configuration(ttorque_kernel.fun)
            println("torque_thermo_gpu! kernel compile successfully ", ttorque_config)
            ttorque_threads = ttorque_config.threads
            ttorque_blocks = ttorque_config.blocks
        else
            torque_kernel = @cuda launch=false torque_gpu!(tx_d,ty_d,tz_d,nx_d,ny_d,nz_d,E_d,periodic_d)
            #torque_kernel = @cuda launch=false torque_gpu!(tx_d,ty_d,tz_d,nx_d,ny_d,nz_d,E_d)
            torque_config = launch_configuration(torque_kernel.fun)
            println("torque_gpu! kernel compile successfully ", torque_config)
            torque_threads = torque_config.threads
            torque_blocks = torque_config.blocks
        end
    end
    println("Compilation time: $ctime s")
    CUDA.@sync @cuda threads=norm_config.threads blocks=norm_config.blocks normalize_gpu!(nx_d,ny_d,nz_d)
     
    pbar = [params.nsteps/100; [i*params.nsteps/10 for i in 1:10]] 
    pbar_counter = 0; elapsed_sum = 0; elapsed_counter = 0

    for istep in 1:params.nsteps
        elapsed_sum += CUDA.@elapsed begin
        elapsed_counter += 1

        #progress update and statistics
        if istep % pbar[pbar_counter+1] == 0
            elapsed_sum,pbar_counter = progress(params,istep,pbar_counter,elapsed_sum,elapsed_counter)
        end

        #updating torques
        if params.thermobool == true
            fkick_d = cu(fkick[istep])
            CUDA.@sync @cuda threads=ttorque_threads blocks=ttorque_blocks torque_thermo_gpu!(tx_d,ty_d,tz_d,nx_d,ny_d,nz_d,E_d,fkick_d,periodic_d)
            Lx,Ly,Lz = params.dimensions
            # costheta_inab_d,costheta_jnab_d,costheta_knab_d = CUDA.rand(Lx,Ly,Lz),CUDA.rand(Lx,Ly,Lz),CUDA.rand(Lx,Ly,Lz)
            # mag_inab_d,mag_jnab_d,mag_knab_d = CUDA.randn(Lx,Ly,Lz),CUDA.randn(Lx,Ly,Lz),CUDA.randn(Lx,Ly,Lz)
            # phi_inab_d,phi_jnab_d,phi_knab_d = CUDA.rand(Lx,Ly,Lz),CUDA.rand(Lx,Ly,Lz),CUDA.rand(Lx,Ly,Lz)
        else
            # running the torque gpu kernel without a thermostat
            CUDA.@sync @cuda threads=torque_threads blocks=torque_blocks torque_gpu!(tx_d,ty_d,tz_d,nx_d,ny_d,nz_d,E_d,periodic_d)
            #CUDA.@sync @cuda threads=torque_threads blocks=torque_blocks torque_gpu!(tx_d,ty_d,tz_d,nx_d,ny_d,nz_d,E_d,tx_inab_d,tx_jnab_d,tx_knab_d,ty_inab_d,ty_jnab_d,ty_knab_d,tz_inab_d,tz_jnab_d,tz_knab_d,periodic_d)
        end

        # reconstructing tx,ty,tz
        # tx_d .= tx_d .+ tx_inab_d .+ tx_jnab_d .+ tx_knab_d
        # ty_d .= ty_d .+ ty_inab_d .+ ty_jnab_d .+ ty_knab_d
        # tz_d .= tz_d .+ tz_inab_d .+ tz_jnab_d .+ tz_knab_d
        # updating positions
        CUDA.@sync @cuda threads=pos_threads blocks=pos_blocks position_gpu!(CC_d,dt_d,nx_d,ny_d,nz_d,tx_d,ty_d,tz_d,KE_d)
        # scaling potential energy
        E_d = 0.5*(K_d*CC_d)  .* E_d

        #if total_defects > 1
        # resetting the defects on the bottom substrate
            nx_d[:,:,1] = nxb_d
            ny_d[:,:,1] = nyb_d
            nz_d[:,:,1] = nzb_d
            # setting the top substrate with an anchoring conditions
            nx_d[:,:,params.dimensions[3]] .= cos(phi[istep])
            ny_d[:,:,params.dimensions[3]] .= sin(phi[istep])
            nz_d[:,:,params.dimensions[3]] .= 0
        #end 

        #renormalizing
        CUDA.@sync @cuda threads=norm_threads blocks=norm_blocks normalize_gpu!(nx_d,ny_d,nz_d)

        #writing out the data
        if save==true 
            if istep%esave == 0
                #writing energies to file
                e_filename = string("energy_"*params.simname*"_gpu.csv")
                write_E(e_filename,istep,sum(Array(E_d))/N3,sum(Array(KE_d))/N3) # TODO is summing on GPU faster?

                # writing torque on top substrate and angles out
                t_filename = string("torque_"*params.simname*"_gpu.csv")
                avg_T = sum(Array(tx_d[:,:,params.dimensions[3]]) .+ Array(ty_d[:,:,params.dimensions[3]]) .+ Array(tz_d[:,:,params.dimensions[3]]))/(params.dimensions[1]*params.dimensions[2])
                write_T(t_filename,istep,phi[istep],avg_T)
            end
            if istep%params.isave == 0
                write_xyzv(Float16,string("vectors_"*params.simname*"_gpu.xyzv"),params,istep,Array(nx_d),Array(ny_d),Array(nz_d),Array(E_d))
                #write_array_data(params,istep,Array(nx_d),Array(ny_d),Array(nz_d),Array(E_d))
            end
        end

        if !iszero(params.checkpoints) && istep in params.checkpoints
            create_checkpoint(params,istep,Array(nx_d),Array(ny_d),Array(nz_d),Array(E_d))
        end

        # zeroing out the torques
        tx_d .=0 # = CUDA.zeros(params.dimensions)
        ty_d .=0 # = CUDA.zeros(params.dimensions)
        tz_d .=0 # = CUDA.zeros(params.dimensions)
        E_d .=0 # = CUDA.zeros(params.dimensions)
        KE_d .=0
        # tx_inab_d .= 0
        # tx_jnab_d .= 0
        # tx_knab_d .= 0
        # ty_inab_d .= 0
        # ty_jnab_d .= 0
        # ty_knab_d .= 0
        # tz_inab_d .= 0
        # tz_jnab_d .= 0
        # tz_knab_d .= 0
        end #elapsed
    end #steps
    nx = Array(nx_d)
    ny = Array(ny_d)
    nz = Array(nz_d)

    return nx,ny,nz
end
end

