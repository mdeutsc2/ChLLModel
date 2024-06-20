module ll_setup_module
import ..SimParams
import ..dtype_gpu
export setup_lines
export setup_defects

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
end