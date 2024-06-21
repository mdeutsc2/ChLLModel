__precompile__()
module ll_io_module 

using JLD2
using YAML
using Dates
using OrderedCollections
using WriteVTK,Printf

import ..SimParams

export create_checkpoint
export load_checkpoint
export save_params
export parse_args
export fill_vector
export write_E
export write_T
export write_xyzv
export setup_env
export write_vtk

"""
    fill_vector(::Vector{Tuple{Float64, Float64, Int64}})::Vector{Float64}

Function to return a vector filled with values from SimParams for the top substrate twist and 
the thermostat, kbt. Accepts a list in the form of [(1.0,1000),(1.0,2.0,1000)]
"""
function fill_vector(list_of_tuples::Vector{Tuple})::Vector{Float64}
    vec = Float64[]
    for step in list_of_tuples
        if length(step) == 3
           start = step[1]
           stop = step[2]
           len = Int(step[3])
           append!(vec,collect(LinRange(start,stop,len)))
       elseif length(step) == 2
           append!(vec,fill(step[1],Int(step[2])))
       else
           @error "step size undefined"
       end
   end
   return vec
end

function write_E(filename,istep,E,KE)
    f = open(filename,"a+")
    if istep == 1
        write(f,"Step,PE,KE\n")
        write(f,string(istep,",",E,",",KE,"\n"))
    else
        write(f,string(istep,",",E,",",KE,"\n"))
    end
    close(f)
    return nothing
end

function write_T(filename,istep,dtheta,summedT)
    f = open(filename, "a+")
    if istep == 1
        write(f,"Step, dtheta, torque")
        write(f, string(istep,",",dtheta,",",summedT,"\n"))
    else
        write(f, string(istep,",",dtheta,",",summedT,"\n"))
    end
    close(f)
    return nothing
end

function write_vtk(dtype,filename,params,istep,nx,ny,nz,E)
    nx = convert.(dtype,nx)
    ny = convert.(dtype,ny)
    nz = convert.(dtype,nz)
    E = convert.(dtype,E)
    fmt_str = string(split(filename,".")[1]*"_%06i")
    fmt = Printf.Format(fmt_str)
    filename = Printf.format(fmt,istep)
    vtk_grid(filename, 1:params.dimensions[1], 1:params.dimensions[2],1:params.dimensions[3], ascii=true) do vtk
        vtk["director", VTKPointData()] = (nx,ny,nz)
        vtk["Energy",VTKPointData()] = E
    end
end

function write_xyzv(dtype,filename,params,istep,nx,ny,nz,E)
    nx = convert.(dtype,nx)
    ny = convert.(dtype,ny)
    nz = convert.(dtype,nz)
    E = convert.(dtype,E)
    f = open(filename, "a+")
    write(f, string(params.dimensions,"\n"))
    write(f,string(istep,"\n"))
    for i in 1:params.dimensions[1]
        for j in 1:params.dimensions[2]
            for k in 1:params.dimensions[3]
                write(f,string(i," ",j," ",k," ",nx[i,j,k]," ",ny[i,j,k]," ",nz[i,j,k]," ",E[i,j,k],"\n"))
            end
        end
    end
    close(f)
    return nothing
end

function write_array_data(params::SimParams,istep::Int,nx,ny,nz,E)
    filename = "data_"*params.simname*".h5"
    if params.savedata == "legacy"
        filename = string("vectors_"*params.simname*".xyzv")
        write_xyzv(filename,params,istep,convert.(Float16,nx),convert.(Float16,ny),convert.(Float16,nz),convert.(Float16,E))
    end
    if params.savedata == "both"
        jldopen(filename,"a+") do file
            # convert to FP16 to save space
            file["nx/$istep"] = convert.(Float16,nx)
            file["ny/$istep"] = convert.(Float16,ny)
            file["nz/$istep"] = convert.(Float16,nz)
            file["E/$istep"] = convert.(Float16,E)
        end
    end
    if params.savedata == "vectors"
        jldopen(filename,"a+") do file
            file["nx/$istep"] = convert.(Float16,nx)
            file["ny/$istep"] = convert.(Float16,ny)
            file["nz/$istep"] = convert.(Float16,nz)
        end
    end
    if params.savedata == "energies"
        jldopen(filename,"a+") do file
            file["E/$istep"] = convert.(Float16,E)
        end
    end
    return nothing
end

function parse_args()
    # getting params from arguments
    if !isempty(ARGS) && occursin(".yml",ARGS[end])
        params_filename = ARGS[end]
        @assert occursin(".yml",params_filename) "SimParams file not a YAML (.yml) file or does not exist"
        raw_params = YAML.load_file(params_filename) # load raw Dict from YAML file
        println("$params_filename loaded")
        params = SimParams(raw_params)
    else
        params = nothing # will use default parameters in ll_main.jl
		@error "no parameters specified"
		exit()
    end

    # getting computation target
    if !isempty(ARGS) && (ARGS[begin] == "--target" || ARGS[begin] == "-t")
        target = uppercase(ARGS[begin+1])
    else
        println("Specify a computation target with \"--target {CPU,GPU}\" or \"-t {CPU,GPU}\" ")
        @error "Target not specified"
    end
    return params,target
end

function save_params(params)
    # saving params as a YAML file
    params_filename = "params_"*params.simname*".yml"
    println("Saved parameters to $params_filename")
    params_dict = OrderedDict(key=>getfield(params, key) for key ∈ fieldnames(SimParams))
    YAML.write_file(params_filename,params_dict)
end

function create_checkpoint(params::SimParams,istep,nx,ny,nz,E)
    filename = params.simname*"_"*string(istep)*".ckp.jld2"
    println(string("Saving checkpoint... ",filename))
    # creating two groups, params and data
    # converting params into dict to loop over and save
    params_dict = Dict(key=>getfield(params, key) for key ∈ fieldnames(SimParams))
    jldopen(filename,"w") do file
        for (key,value) in params_dict
            file[string("params/",key)] = value
        end
        file["data/nx"] = nx
        file["data/ny"] = ny
        file["data/nz"] = nz
        file["data/E"] = E
    end # close file
    return filename
end

function load_checkpoint(filename::String)
    file = jldopen(filename, "r")

    ckptparams = SimParams(file["params/nsteps"], # nsteps
                       file["params/progress"], # ProgressMeter
                       file["params/simname"], #simname
                       file["params/dimensions"],
                       file["params/dt"],
                       file["params/CC"],
                       file["params/K"],
                       file["params/isave"],
                       file["params/savedata"],
                       file["params/checkpoints"],
                       file["params/loadckpt"],
                       file["params/defectfinding"],
                       file["params/periodic"],
                       file["params/defects"],
                       file["params/phi"],
                       file["params/thermobool"],
                       file["params/kbt"]
                      )
    nx = file["data/nx"]
    ny = file["data/ny"]
    nz = file["data/nz"]
    E = file["data/E"]
    @warn "Params from checkpoint loaded, modify params before running simulation"
    return ckptparams,nx,ny,nz,E
end

function setup_env(save)
    if save
        sim_folder = Dates.format(Dates.today(), "mm-dd-yyyy")
        if !isdir(sim_folder)
            mkdir(sim_folder)
        end
        cd(sim_folder)
        println("Working directory is now ",pwd())
    end
end
end #module