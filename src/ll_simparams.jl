#ll_simparams.jl
__precompile__()
module ll_simparams
export SimParams

# PARAMETERS
mutable struct SimParams 
    nsteps::Int64 # total number of steps for simulation to run
    progress::Bool # boolean for a progress meter
    simname::String
    dimensions::Tuple{Int64,Int64,Int64} # number of particles per row
    dt::Float64 # time step
    CC::Float64 # motility, C
    K::Float64# Frank constant equivalent
    isave::Int64 #save interval for saving the vectors
    savedata::String
    checkpoints::Vector{Int64} # vector of when to save a checkpoint
    loadckpt::String # filename for the checkpoint to load
    defectfinding::Bool # find and record defects at the save intervals
    periodic::Bool # boolean for x and y boundary conditions
    defects::Dict #defect parameters, specify ndefects" => [array of charges], "spacing" => (x spacing, y spacing)
    phi::Vector{Tuple} # top anchoring angle relative to the x axis in degrees
    thermobool::Bool # turn thermostat on and off
    kbt::Vector{Tuple} #temperature 
    flow :: Bool # turn Lattice-Boltzmann dynmaics 
end 

default_params = SimParams(50_000, # nsteps
                            true, # ProgressMeter
                            "80x80x10_400_000steps_0.1kbt_rot_cont", #simname
                            (80,80,10), # Nx,Ny,Nz
                            0.1, # dt
                            1.0, # CC
                            1.0, # K
                            2000, # isave
                            "both", # savedata (does nothing if not saving to hdf5 files)
                            [0], # checkpoints, if 0 do not save any checkpoints
                            "", # load checkpoint
                            false, # defectfinding
                            false, # periodic
                            Dict("ndefects" => [0.5,-0.5], "spacing" => (8,0)), # defects
                            [(pi/2,deg2rad(90+4*122),400_000)],#LinRange(pi/2,-pi/2,2_000),
                            true, # thermobool
                            [(0.5,0.1,15_000),(0.1,400_000)]) #kbt

function SimParams(d::Dict)
    function parse_defects(d)
        defects = Dict()
        for (key,value) in d
            if key == "ndefects"
                #print(typeof(value))
                push!(defects,"ndefects"=>value)
            end
            if key == "spacing"
                spacing = Tuple(parse.(Int64,split(strip(value,['(',')']),',')))
                push!(defects,"spacing"=>spacing)
            end
        end
        return defects
    end

    SimParams(d["nsteps"],
              d["progress"],
              d["simname"],
              Tuple(parse.(Int64,split(strip(d["dimensions"],['(',')']),','))),
              d["dt"],
              d["CC"],
              d["K"],
              d["isave"],
              d["savedata"],
              d["checkpoints"],
              d["loadckpt"],
              d["defectfinding"],
              d["periodic"],
              #parse_defects(d["defects"]),
              d["defects"],
              #Dict([key,Tuple(parse.(Int64,split(strip(value,['(',')']),',')))] for (key,value) in d["defects"]),
              [Tuple(parse.(Float64,split(strip(row,['(',')']),','))) for row in d["phi"]],
              d["thermobool"],
              [Tuple(parse.(Float64,split(strip(row,['(',')']),','))) for row in d["kbt"]]
              )
end


end #module