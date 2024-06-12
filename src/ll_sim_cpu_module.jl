__precompile__()
module ll_sim_cpu_module 
using Random #for seeding
import Distributed.procs
import ..SimParams
import ..create_checkpoint
import ..find_defects
import ..fill_vector
import ..write_E
import ..write_array_data
export run_sim_cpu 

const dtype_cpu = Float32

# FUNCTIONS
function __init__()
    @info string("CPUs : ",length(procs()))
end

function thermostat(fkick)
    # thermostat
    costheta = 2.0 * rand(Float64) -1.0 #random float between -1,1
    theta = acos(costheta)
    phi = 2.0*pi*rand(Float64) # random float between 0,2π
    mag = randn(Float64)
    # converting spherical to cartesian
 
    x = sin(theta)*cos(phi)*mag *fkick
    y = sin(theta)*sin(phi)*mag *fkick
    z = cos(theta)*mag * fkick
    return x,y,z
end

function torque_therm(params::SimParams,fkick::Array{Float64,1},istep::Int64,tx::Array{Float64,3},ty::Array{Float64,3},tz::Array{Float64,3},nx::Array{Float64,3},ny::Array{Float64,3},nz::Array{Float64,3},E::Array{Float64,3})::Tuple{Array{Float64, 3}, Array{Float64, 3}, Array{Float64, 3}, Array{Float64, 3}}
    #Edp = 0
    for index in CartesianIndices(nx)
        i,j,k = index[1],index[2],index[3]
        
        # i+1,j,k - note: by running neighbors individually, reduces time from ~60ms/it to ~30ms/it
        jnab,knab = j,k
        params.periodic ? inab = ifelse(i+1>params.dimensions[1],1,i+1) : inab = ifelse(i+1>params.dimensions[1],i,i+1)
        dp = nx[i,j,k]*nx[inab,jnab,knab] + ny[i,j,k]*ny[inab,jnab,knab]+nz[i,j,k]*nz[inab,jnab,knab]
        #println(i,j,k,", (",nx[i,j,k],",",ny[i,j,k],",",nz[i,j,k],") ",inab,jnab,knab,"(",nx[inab,jnab,knab],",",ny[inab,jnab,knab],",",nz[inab,jnab,knab],") : ",dp)

        E[i,j,k] = E[i,j,k] + (1-(dp*dp))
        #half_E = 0.5*(1-(dp*dp))
        #E[i,j,k] += half_E
        #E[inab,jnab,knab] += half_E

        #calculate the x,y,z components of the cross product between the same two vectors
        cx = ny[i,j,k]*nz[inab,jnab,knab]-nz[i,j,k]*ny[inab,jnab,knab]
        cy = nz[i,j,k]*nx[inab,jnab,knab]-nx[i,j,k]*nz[inab,jnab,knab]
        cz = nx[i,j,k]*ny[inab,jnab,knab]-ny[i,j,k]*nx[inab,jnab,knab]
        
        txkick,tykick,tzkick = thermostat(fkick[istep])
        txbond = dp*cx + txkick
        tybond = dp*cy + tykick
        tzbond = dp*cz + tzkick
        
        tx[i,j,k] += txbond
        tx[inab,jnab,knab] -= txbond

        ty[i,j,k] += tybond
        ty[inab,jnab,knab] -= tybond

        tz[i,j,k] += tzbond
        tz[inab,jnab,knab] -= tzbond

        #i,j+1,K
        inab,knab = i,k
        params.periodic ? jnab = ifelse(j+1>params.dimensions[2],1,j+1) : jnab = ifelse(j+1>params.dimensions[2],j,j+1)
        dp = nx[i,j,k]*nx[inab,jnab,knab] + ny[i,j,k]*ny[inab,jnab,knab] + nz[i,j,k]*nz[inab,jnab,knab]
        #println(i,j,k,", (",nx[i,j,k],",",ny[i,j,k],",",nz[i,j,k],") ",inab,jnab,knab,"(",nx[inab,jnab,knab],",",ny[inab,jnab,knab],",",nz[inab,jnab,knab],") : ",dp)

        E[i,j,k] = E[i,j,k] + (1-(dp*dp))
        #half_E = 0.5*(1-(dp*dp))
        #E[i,j,k] += half_E
        #E[inab,jnab,knab] += half_E

        #calculate the x,y,z components of the cross product between the same two vectors
        cx = ny[i,j,k]*nz[inab,jnab,knab]-nz[i,j,k]*ny[inab,jnab,knab]
        cy = nz[i,j,k]*nx[inab,jnab,knab]-nx[i,j,k]*nz[inab,jnab,knab]
        cz = nx[i,j,k]*ny[inab,jnab,knab]-ny[i,j,k]*nx[inab,jnab,knab]

        txkick,tykick,tzkick = thermostat(fkick[istep])
        txbond = dp*cx + txkick
        tybond = dp*cy + tykick
        tzbond = dp*cz + tzkick
        
        tx[i,j,k] += txbond
        tx[inab,jnab,knab] -= txbond

        ty[i,j,k] += tybond
        ty[inab,jnab,knab] -= tybond

        tz[i,j,k] += tzbond
        tz[inab,jnab,knab] -= tzbond

        #i,j,k+1
        inab,jnab,knab = i,j,ifelse(k+1>params.dimensions[3],1,k+1)
        
        if k == params.dimensions[3] 
            # top slice energy should not depend on bottom slice energy
            dp = nx[i,j,k]*nx[inab,jnab,k] + ny[i,j,k] * ny[inab,jnab,k] + nz[i,j,k]*nz[inab,jnab,k]
        else
            dp = nx[i,j,k]*nx[inab,jnab,knab]+ny[i,j,k]*ny[inab,jnab,knab]+nz[i,j,k]*nz[inab,jnab,knab]
        end
        
        #dp = nx[i,j,k]*nx[inab,jnab,knab]+ny[i,j,k]*ny[inab,jnab,knab]+nz[i,j,k]*nz[inab,jnab,knab]
        #println(i,j,k,", (",nx[i,j,k],",",ny[i,j,k],",",nz[i,j,k],") ",inab,jnab,knab,"(",nx[inab,jnab,knab],",",ny[inab,jnab,knab],",",nz[inab,jnab,knab],") : ",dp)
        E[i,j,k] = E[i,j,k] + (1-(dp*dp))

        #calculate the x,y,z components of the cross product between the same two vectors
        cx = ny[i,j,k]*nz[inab,jnab,knab]-nz[i,j,k]*ny[inab,jnab,knab]
        cy = nz[i,j,k]*nx[inab,jnab,knab]-nx[i,j,k]*nz[inab,jnab,knab]
        cz = nx[i,j,k]*ny[inab,jnab,knab]-ny[i,j,k]*nx[inab,jnab,knab]

        txkick,tykick,tzkick = thermostat(fkick[istep])
        txbond = dp*cx + txkick
        tybond = dp*cy + tykick
        tzbond = dp*cz + tzkick

        tx[i,j,k] += txbond
        tx[inab,jnab,knab] -= txbond

        ty[i,j,k] += tybond
        ty[inab,jnab,knab] -= tybond

        tz[i,j,k] += tzbond
        tz[inab,jnab,knab] -= tzbond
    end #index
    #E = (params.K*params.CC)/2 * Edp

    E = (params.K*params.CC)/2  .* E
    return tx,ty,tz,E
end

function torque(params::SimParams,tx::Array{Float64,3},ty::Array{Float64,3},tz::Array{Float64,3},nx::Array{Float64,3},ny::Array{Float64,3},nz::Array{Float64,3},E::Array{Float64,3})::Tuple{Array{Float64, 3}, Array{Float64, 3}, Array{Float64, 3}, Array{Float64, 3}}
    #Edp = 0
    for index in CartesianIndices(nx)
        i,j,k = index[1],index[2],index[3]
        
        # i+1,j,k - note: by running neighbors individually, reduces time from ~60ms/it to ~30ms/it
        jnab,knab = j,k
        params.periodic ? inab = ifelse(i+1>params.dimensions[1],1,i+1) : inab = ifelse(i+1>params.dimensions[1],i,i+1)
        dp = nx[i,j,k]*nx[inab,jnab,knab] + ny[i,j,k]*ny[inab,jnab,knab]+nz[i,j,k]*nz[inab,jnab,knab]
        #println(i,j,k,", (",nx[i,j,k],",",ny[i,j,k],",",nz[i,j,k],") ",inab,jnab,knab,"(",nx[inab,jnab,knab],",",ny[inab,jnab,knab],",",nz[inab,jnab,knab],") : ",dp)

        E[i,j,k] = E[i,j,k] + (1-(dp*dp))
        #half_E = 0.5*(1-(dp*dp))
        #E[i,j,k] += half_E
        #E[inab,jnab,knab] += half_E

        #calculate the x,y,z components of the cross product between the same two vectors
        cx = ny[i,j,k]*nz[inab,jnab,knab]-nz[i,j,k]*ny[inab,jnab,knab]
        cy = nz[i,j,k]*nx[inab,jnab,knab]-nx[i,j,k]*nz[inab,jnab,knab]
        cz = nx[i,j,k]*ny[inab,jnab,knab]-ny[i,j,k]*nx[inab,jnab,knab]

        txbond = dp*cx
        tybond = dp*cy
        tzbond = dp*cz
        
        tx[i,j,k] += txbond
        tx[inab,jnab,knab] -= txbond

        ty[i,j,k] += tybond
        ty[inab,jnab,knab] -= tybond

        tz[i,j,k] += tzbond
        tz[inab,jnab,knab] -= tzbond

        #i,j+1,K
        inab,knab = i,k
        params.periodic ? jnab = ifelse(j+1>params.dimensions[2],1,j+1) : jnab = ifelse(j+1>params.dimensions[2],j,j+1)
        dp = nx[i,j,k]*nx[inab,jnab,knab] + ny[i,j,k]*ny[inab,jnab,knab] + nz[i,j,k]*nz[inab,jnab,knab]
        #println(i,j,k,", (",nx[i,j,k],",",ny[i,j,k],",",nz[i,j,k],") ",inab,jnab,knab,"(",nx[inab,jnab,knab],",",ny[inab,jnab,knab],",",nz[inab,jnab,knab],") : ",dp)

        E[i,j,k] = E[i,j,k] + (1-(dp*dp))
        #half_E = 0.5*(1-(dp*dp))
        #E[i,j,k] += half_E
        #E[inab,jnab,knab] += half_E

        #calculate the x,y,z components of the cross product between the same two vectors
        cx = ny[i,j,k]*nz[inab,jnab,knab]-nz[i,j,k]*ny[inab,jnab,knab]
        cy = nz[i,j,k]*nx[inab,jnab,knab]-nx[i,j,k]*nz[inab,jnab,knab]
        cz = nx[i,j,k]*ny[inab,jnab,knab]-ny[i,j,k]*nx[inab,jnab,knab]


        txbond = dp*cx
        tybond = dp*cy
        tzbond = dp*cz

        
        tx[i,j,k] += txbond
        tx[inab,jnab,knab] -= txbond

        ty[i,j,k] += tybond
        ty[inab,jnab,knab] -= tybond

        tz[i,j,k] += tzbond
        tz[inab,jnab,knab] -= tzbond

        #i,j,k+1
        inab,jnab,knab = i,j,ifelse(k+1>params.dimensions[3],1,k+1)
        if k == params.dimensions[3] 
            # top slice energy should not depend on bottom slice energy
            dp = nx[i,j,k]*nx[inab,jnab,k] + ny[i,j,k] * ny[inab,jnab,k] + nz[i,j,k]*nz[inab,jnab,k]
        else
            dp = nx[i,j,k]*nx[inab,jnab,knab]+ny[i,j,k]*ny[inab,jnab,knab]+nz[i,j,k]*nz[inab,jnab,knab]
        end
        #println(i,j,k,", (",nx[i,j,k],",",ny[i,j,k],",",nz[i,j,k],") ",inab,jnab,knab,"(",nx[inab,jnab,knab],",",ny[inab,jnab,knab],",",nz[inab,jnab,knab],") : ",dp)
        E[i,j,k] = E[i,j,k] + (1-(dp*dp))

        #calculate the x,y,z components of the cross product between the same two vectors
        cx = ny[i,j,k]*nz[inab,jnab,knab]-nz[i,j,k]*ny[inab,jnab,knab]
        cy = nz[i,j,k]*nx[inab,jnab,knab]-nx[i,j,k]*nz[inab,jnab,knab]
        cz = nx[i,j,k]*ny[inab,jnab,knab]-ny[i,j,k]*nx[inab,jnab,knab]


        txbond = dp*cx
        tybond = dp*cy
        tzbond = dp*cz

        
        tx[i,j,k] += txbond
        tx[inab,jnab,knab] -= txbond

        ty[i,j,k] += tybond
        ty[inab,jnab,knab] -= tybond

        tz[i,j,k] += tzbond
        tz[inab,jnab,knab] -= tzbond
    end #index
    #E = (params.K*params.CC)/2 * Edp

    E = (params.K*params.CC)/2  .* E
    return tx,ty,tz,E
end

function position(params::SimParams,nx::Array{Float64,3},ny::Array{Float64,3},nz::Array{Float64,3},tx::Array{Float64,3},ty::Array{Float64,3},tz::Array{Float64,3})::Tuple{Array{Float64, 3}, Array{Float64, 3}, Array{Float64, 3}, Array{Float64,3}}
    wx = tx .* params.CC
    wy = ty .* params.CC
    wz = tz .* params.CC
    #KE = 0.5.*(wx.*wx + wy.*wy + wz.*wz)
    wXnx = wy.*nz
    wXny = wz.*nx
    wXnz = wx.*ny
    KE = 0.5.*(wXnx.*wXnx + wXny.*wXny + wXnz.*wXnz)
    nx += wXnx .* params.dt
    ny += wXny .* params.dt
    nz += wXnz .* params.dt
    return nx,ny,nz, KE
end



function normalize(nx,ny,nz)
    s = sqrt.(nx.*nx .+ ny.*ny .+ nz.*nz)
    nx = nx./s
    ny = ny./s
    nz = nz./s
    return nx,ny,nz
end

""" 
Function to set up defects on a single (bottom) substrate
"""
function setup_defects_1(params::SimParams,nx,ny,nz)
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
    spacing = params.defects["spacing"]
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


"""
Function to set up antagonistic anchoring for line defects on a single (bottom) substrate
"""
function setup_lines_1(params::SimParams,nx,ny,nz)
    @assert params.dimensions[1] == params.dimensions[2] string("xy dimension mismatch: ",params.dimensions[1],"!=",params.dimensions[2])
    k = 1 # for the bottom substrate
    L = 20 # period of the director stripe patter from Babakhanova et. al. 2020
    for i = 1:params.dimensions[1] # large dimension if i!=j
        for j in 1:params.dimensions[2]
            theta = (pi * i )/ L
            nx[i,j,k] = cos(theta)
            ny[i,j,k] = sin(theta)
            nz[i,j,k] = 0
        end
    end
    return nx,ny,nz
end
"""
Function to set up defects on both a top and bottom substrate
"""
function setup_defects_2(params::SimParams,nx,ny,nz)
    @assert params.dimensions[1] == params.dimensions[2] string("xy dimension mismatch: ",params.dimensions[1],"!=",params.dimensions[2])
    println("Defects:")
    ndefects = params.defects["ndefects"]
    spacing = params.defects["spacing"]
    th_bot = zeros(params.dimensions[1],params.dimensions[2])
    th_top = zeros(params.dimensions[1],params.dimensions[2])
    xx = zeros(ndefects)
    yy = zeros(ndefects)
    q_bot = zeros(ndefects) # bottom defect charges
    q_top = zeros(ndefects)

    if sum(ndefects) > 1
        id = 0
        for ii in 1:Int(ndefects[1]) 
            for jj in 1:Int(ndefects[2])
                id += 1
                if !isempty(spacing) # test if scaling parameters are specified
                    if spacing[1] != 0
                        xx[id] = ii*spacing[1] + 0.5
                        xx[id] = xx[id] +(1-(spacing[1]*(ndefects[1]+1))/params.dimensions[1])*(params.dimensions[1]/2)
                    else
                        xx[id] = ii*(params.dimensions[1]/(ndefects[1]+1))+0.5
                    end
                    if spacing[2] != 0 
                        yy[id] = jj*spacing[2] + 0.5
                        yy[id] = yy[id] + (1-(spacing[2](ndefects[2]+1))/params.dimensions[2])*(params.dimensions[2]/2)
                    else
                        yy[id] = jj*(params.dimensions[2]/(ndefects[2]+1))+0.5
                    end
                else 
                    xx[id] = ii*(params.dimensions[1]/(ndefects[1]+1))+0.5
                    yy[id] = jj*(params.dimensions[2]/(ndefects[2]+1))+0.5
                end
                q_bot[id] = 0.5
                if sum(ndefects) == 4
                    if id == 2 || id == 3
                        q_bot[id] = -0.5
                    end
                elseif sum(ndefects) != 4
                    if id%2 == 0
                        q_bot[id] = -0.5
                    end
                end
                println("\t",xx[id],"\t",yy[id],"\t",q_bot[id])
            end
        end

    else
        xx[1] = params.dimensions[1]/2 +0.5
        yy[1] = params.dimensions[1]/2 +0.5
        q_bot[1] = 0.5
    end
    #q_bot[2] = 0.5 # for opposite defects on top and bottom
    #q_top .= -q_bot
    q_top .= q_bot
    for idefect = 1:id # theta for bottom defects
        for i = 1:params.dimensions[1] 
            for j = 1:params.dimensions[2]
                phi = atan(j-yy[idefect],i-xx[idefect])
                th_bot[i,j] += q_bot[idefect]*phi + pi/4.0 + pi/4.0
            end
        end
    end
    for idefect = 1:id # theta for top defects
        for i = 1:params.dimensions[1] 
            for j = 1:params.dimensions[2]
                phi = atan(j-yy[idefect],i-xx[idefect])
                th_top[i,j] += q_top[idefect]*phi + pi/4.0 + pi/4.0
            end
        end
    end
    k = 1 #for the bottom substrate
    for i = 1:params.dimensions[1]
        for j = 1:params.dimensions[2] 
            nx[i,j,k] = 0.5*cos(th_bot[i,j])
            ny[i,j,k] = 0.5*sin(th_bot[i,j])
            nz[i,j,k] = 0
        end
    end
    k = params.dimensions[3] #for the top substrate
    for i = 1:params.dimensions[1]
        for j = 1:params.dimensions[2] 
            nx[i,j,k] = 0.5*cos(th_top[i,j])
            ny[i,j,k] = 0.5*sin(th_top[i,j])
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
    # nx,ny,nz are read + written (2*3), tx,ty,tz are read + written twice + written 4 times (2*3+4), E,KE are written to (1+1), tx,tz,tz,E,KE are written to (5) (zeros)
    T_eff = lpad(round(((2*3+2*3+4+1+1+5)*1/1e9*params.dimensions[1]*params.dimensions[2]*params.dimensions[3]*sizeof(Float64))/(elapsed_sum/elapsed_counter), digits = 4),6)
    # calculating RAM usage
    used_mem = Base.format_bytes(Sys.total_memory()-Sys.free_memory())
    total_mem = Base.format_bytes(Sys.total_memory())
    mem_ratio = round(100*((Sys.total_memory()-Sys.free_memory())/Sys.total_memory()), digits=3)
    println(istep, ' '^(length(digits(params.nsteps))-length(digits(istep))), " (",' '^(3-length(digits(floor(Int,istep/params.nsteps*100)))), istep/params.nsteps * 100,"%) ", '='^(2*pbar_counter), ' '^(20-2*pbar_counter), " Avg. iter: $avg_it s, ETA: $eta min, Mem. throughput: $T_eff GB/s , Mem. usage: $used_mem/$total_mem ($mem_ratio%)")
    pbar_counter += 1
    elapsed_sum = 0
    return elapsed_sum,pbar_counter
end

function run_sim_cpu(params::SimParams;save::Bool=true,ckpsave::Bool=false,ckpload::Tuple=(),save_vectors::Bool=false)
    @info "run_sim_cpu"
    # SETUP
    
    Nx,Ny,Nz = params.dimensions
    N3 = Nx*Ny*Nz
    total_defects = sum(size(params.defects["ndefects"])) #
    phi = fill_vector(params.phi)
    kbt = fill_vector(params.kbt)
    @assert length(phi) == params.nsteps "phi vector not defined over all timesteps"
    @assert length(kbt) == params.nsteps "kbt vector not defined over all timesteps"

    fkick = sqrt.(8*kbt.*params.dt)
    # randomize spins
    #Random.seed!(987654321)

    println("Memory usage: ",Base.format_bytes(Sys.total_memory()-Sys.free_memory()),"/",Base.format_bytes(Sys.total_memory()), " (",round(100*((Sys.total_memory()-Sys.free_memory())/Sys.total_memory()),digits=3),"%)")
    if !isempty(ckpload)
        @info "loading nx,ny,nz,E from checkpoint data"
        # if loading from a checkpoint
        nx = ckpload[1]
        ny = ckpload[2]
        nz = ckpload[3]
        E = ckpload[4]
    else
        # starting a new simulation
        nx = rand(Float64, params.dimensions).-0.5
        ny = rand(Float64, params.dimensions).-0.5
        nz = rand(Float64, params.dimensions).-0.5
        E = zeros((params.dimensions))
    end

    # creating uninitialized torque and kinetic energy arrays
    tx = similar(nx)
    ty = similar(ny)
    tz = similar(nz)
    KE = similar(E)

    if total_defects > 0
        # setting up the defect structure
        #nx,ny,nz = setup_defects_1(params,nx,ny,nz)
        nx,ny,nz = setup_lines_1(params,nx,ny,nz)
        nx,ny,nz = normalize(nx,ny,nz)
         #setting original defect structure
        nx1 = nx[:,:,1]
        ny1 = ny[:,:,1]
        nz1 = nz[:,:,1]
        # setting top defect structure
        #nx_top = nx[:,:,params.dimensions[3]] 
        #ny_top = ny[:,:,params.dimensions[3]]
        #nz_top = nz[:,:,params.dimensions[3]] 
    else
        #normalizing directors
        nx .= 0.0
        ny .= 1.0
        nz .= 0.0
        total_defects = 1
        nx,ny,nz = normalize(nx,ny,nz)
    end
    
    # for k in 1:params.dimensions[3] # for two disclination lines
    #     nx[:,:,k] = nx1
    #     ny[:,:,k] = ny1
    #     nz[:,:,k] = nz1
    # end
    println("Memory usage: ",Base.format_bytes(Sys.total_memory()-Sys.free_memory()),"/",Base.format_bytes(Sys.total_memory()), " (",round(100*((Sys.total_memory()-Sys.free_memory())/Sys.total_memory()),digits=3),"%)")
    
    pbar = [params.nsteps/100; [i*params.nsteps/10 for i in 1:10]] 
    pbar_counter = 0; elapsed_sum = 0; elapsed_counter = 0

    # MAIN LOOP
    for istep in 1:params.nsteps
        elapsed_sum += @elapsed begin
        elapsed_counter += 1
        # progress update and statistics
        if istep % pbar[pbar_counter+1] == 0
            elapsed_sum,pbar_counter = progress(params,istep,pbar_counter,elapsed_sum,elapsed_counter)
        end

        # zeroing out the torques
        tx .= 0.;ty .= 0.; tz .=0.; E.=0.;KE.=0.;

        # calculate torques
        if params.thermobool == true
            tx,ty,tz,E = torque_therm(params,fkick,istep,tx,ty,tz,nx,ny,nz,E)
        else
            tx,ty,tz,E = torque(params,tx,ty,tz,nx,ny,nz,E)
        end

        # update positions
        nx,ny,nz,KE = position(params,nx,ny,nz,tx,ty,tz)
        
        if total_defects > 0
            # resetting the defects on the bottom substrate for strong anchoring
            nx[:,:,1],ny[:,:,1],nz[:,:,1] = nx1,ny1,nz1 
            # setting the top substrate with an anchoring conditions
            nx[:,:,params.dimensions[3]] .= cos(phi[istep])
            ny[:,:,params.dimensions[3]] .= sin(phi[istep])
            nz[:,:,params.dimensions[3]] .= 0.
        end
        
        # normalizing again
        nx,ny,nz = normalize(nx,ny,nz)
        #println(sum(sqrt.(nx.*nx + ny.*ny + nz.*nz))/(N^3))
        
        if save==true
            if istep%params.isave == 0
                if params.defectfinding == true
                    find_defects(params,nx,ny,nz)
                end
                #write_array_data(params,istep,nx,ny,nz,E)
                write_xyzv(Float32,string("vectors_"*params.simname*".csv"),params,istep,nx,ny,nz,E)
            end
            write_E(string("energy_"*params.simname*".csv"),istep,sum(E)/N3,sum(KE)/N3) #energy per particle be saved
        end 
        end
    end #steps
    println(string("Finished ",params.simname))
    if ckpsave == true
        _ = create_checkpoint(params,nx,ny,nz,E)
    end
    return nx,ny,nz,E
end
end #module