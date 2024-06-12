__precompile__()
module ll_analysis_module
    using DelimitedFiles
    import ..SimParams
    export find_defects
    """
    Analysis code to find defects
    """

    function load_data(filename)
        data = readdlm(filename)
        N = cbrt(data[1]) # gets number of particles
        steps = data[2:1002:end] # selects steps
        filter!(x->isa(x,Number),steps) # removes errant strings from steps
        frames = [i for i in 1:length(steps)]
        vecs = Any[] # stores vectors for each frame
        for i in frames
            #a[3:1002,:]
            start = (i-1)*1002+3
            stop = start+999
            push!(vecs, data[start:stop,:]) 
        end
        return vecs,frames,N
    end

    function find_defects(params,nx,ny,nz;write_out=true)
        # Selinger Implementation
        #vectors,_,N = load_data("vectors.xyzv") #loading in the data from the .xyzv file
        #N = Int(N) # getting the dimension
        #data = vectors[end] # selecting the last saved timestep
        #nx = convert(Array{Float64,3},reshape(data[:,4], (N,N,N))) # converting the data from 1D-3D
        #ny = convert(Array{Float64,3},reshape(data[:,5], (N,N,N)))
        #nz = convert(Array{Float64,3},reshape(data[:,6], (N,N,N)))
        #N = params.N
        defects = [] #empty array to store the defects

        for i in 1:params.dimensions[1]
            for j in 1:params.dimensions[2]
                for k in 1:params.dimensions[3]
                    #      vx vy vz
                    nxy = [i j k;
                            i+1 j k;
                            i+2 j k;
                            i+2 j+1 k;
                            i+2 j+2 k;
                            i+1 j+2 k;
                            i j+2 k;
                            i j+1 k]
                    for row in eachrow(nxy) # boundary condition check
                        if params.periodic == true
                            if row[1] > params.dimensions[1]
                                row[1] = 1
                            end
                            if row[2] > params.dimensions[2]
                                row[2] = 1
                            end
                        elseif params.periodic == false
                            if row[1] > params.dimensions[1]
                                row[1] = params.dimensions[1]
                            end
                            if row[2] > params.dimensions[2]
                                row[2] = params.dimensions[2]
                            end
                        else
                            @error "periodicity error in neighbor array xy in finding defects code"
                        end
                    end
                    nyz = [i j k;
                            i j+1 k;
                            i j+2 k;
                            i j+2 k+1;
                            i j+2 k+2;
                            i j+1 k+2;
                            i j k+2;
                            i j k+1]
                    for row in eachrow(nyz) # boundary condition check
                        if params.periodic == true
                            if row[2] > params.dimensions[2]
                                row[2] = 1
                            end
                            if row[3] > params.dimensions[3]
                                row[3] = 1
                            end
                        elseif params.periodic == false
                            if row[2] > params.dimensions[2]
                                row[2] = params.dimensions[2]
                            end
                            if row[3] > params.dimensions[3]
                                row[3] = params.dimensions[3]
                            end
                        else
                            @error "periodicity error neighbor array yz in finding defects code"
                        end
                    end
                    nxz = [i j k;
                            i+1 j k;
                            i+2 j k;
                            i+2 j k+1;
                            i+2 j k+2;
                            i+1 j k+2;
                            i j k+2;
                            i j k+1]
                    for row in eachrow(nxz) # boundary condition check
                        if params.periodic == true
                            if row[1] > params.dimensions[1]
                                row[1] = 1
                            end
                            if row[3] > params.dimensions[3]
                                row[3] = 1
                            end
                        elseif params.periodic == false
                            if row[1] > params.dimensions[1]
                                row[1] = params.dimensions[1]
                            end
                            if row[3] > params.dimensions[3]
                                row[3] = params.dimensions[3]
                            end
                        else
                            @error "periodicity error neighbor array xz in finding defects code"
                        end
                    end

                    # XY defects
                    vx = [nx[row[1],row[2],row[3]] for row in eachrow(nxy)]
                    vy = [ny[row[1],row[2],row[3]] for row in eachrow(nxy)]
                    vz = [nz[row[1],row[2],row[3]] for row in eachrow(nxy)]
                         
                    for kk in 1:7
                        kkp1 = kk+1
                        dp = vx[kk]*vx[kkp1] + vy[kk]*vy[kkp1] + vz[kk]*vz[kkp1]
                        if dp < 0
                            vx[kkp1],vy[kkp1],vz[kkp1] = -vx[kkp1],-vy[kkp1],-vz[kkp1]
                        end
                    end
                    dp = vx[8]*vx[1]+vy[8]*vy[1] + vz[8]*vz[1]
                    if dp < 0
                        if i+1 > params.dimensions[1]
                            push!(defects,[i+1-params.dimensions[1],j+1,k])
                        elseif j+1 > params.dimensions[2]
                            push!(defects,[i+1,j+1-params.dimensions[2],k])
                        else
                            push!(defects,[i+1,j+1,k])
                        end
                    end

                    # YZ defects
                    vx = [nx[row[1],row[2],row[3]] for row in eachrow(nyz)]
                    vy = [ny[row[1],row[2],row[3]] for row in eachrow(nyz)]
                    vz = [nz[row[1],row[2],row[3]] for row in eachrow(nyz)]

                    for kk in 1:7
                        kkp1 = kk+1
                        dp = vx[kk]*vx[kkp1] + vy[kk]*vy[kkp1] + vz[kk]*vz[kkp1]
                        if dp < 0
                            vx[kkp1],vy[kkp1],vz[kkp1] = -vx[kkp1],-vy[kkp1],-vz[kkp1]
                        end
                    end
                    dp = vx[8]*vx[1] + vy[8]*vy[1] + vz[8]*vz[1]
                    if dp < 0
                        if j+1 > params.dimensions[2]
                            push!(defects,[i,j+1-params.dimensions[2],k+1])
                        elseif k+1 > params.dimensions[3]
                            push!(defects,[i,j+1,k+1-params.dimensions[3]])
                        else
                            push!(defects,[i,j+1,k+1])
                        end
                    end

                    # XZ defects
                    vx = [nx[row[1],row[2],row[3]] for row in eachrow(nxz)]
                    vy = [ny[row[1],row[2],row[3]] for row in eachrow(nxz)]
                    vz = [nz[row[1],row[2],row[3]] for row in eachrow(nxz)]
                    
                    for kk in 1:7
                        kkp1 = kk+1
                        dp = vx[kk]*vx[kkp1] + vy[kk]*vy[kkp1] + vz[kk]*vz[kkp1]
                        if dp < 0
                            vx[kkp1],vx[kkp1],vz[kkp1] = -vx[kkp1],-vy[kkp1],-vz[kkp1]
                        end
                    end
                    dp = vx[8]*vx[1] + vy[8]*vy[1] + vz[8]*vz[1]
                    if dp < 0
                        if i+1 > params.dimensions[1]
                            push!(defects,[i+1-params.dimensions[1],j,k+1])
                        elseif k+1 > params.dimensions[3]
                            push!(defects,[i+1,j,k+1-params.dimensions[3]])
                        else 
                            push!(defects,[i+1,j,k+1])
                        end
                    end
                end
            end
        end

        # count the number of unique defects
        u = unique(defects)
        for i in 1:length(u)
            c = count(ii->(ii==u[i]),defects) # counting the instance of unique defects
            x,y,z = u[i][1],u[i][2],u[i][3] # get coords
            u[i] = [x,y,z,c] # create unique list of coords and number of instances
        end
        #exit()
        #println(size(collect(defects)))
        if write_out == true
            filename = string("defects_",params.simname,".csv")
            f = open(filename, "a+")
            write(f, "#####\n") # write a frame delimiter
            for row in eachrow(u)
                row = row[1]
                datastring = string(row[1],",",row[2],",",row[3],",",row[4],"\n")
                write(f, datastring)
            end
            close(f)
        end
        return nothing
    end

end #module
