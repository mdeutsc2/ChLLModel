# ALLPS - Applied Lebwohl-Lasher Parallel Simulation
# ll_main.jl 
# Implementation of the Lebwohl-Lasher Rotor Model in 3D

# julia --math-mode=fast --check-bounds=no test.jl -t {CPU|GPU} simparams.yml
# ./localrun.sh -t CPU simparams.yml
const dtype_gpu = Float32

# MODULES 
include("./ll_simparams.jl")
using .ll_simparams

include("./ll_setup_module.jl")
using .ll_setup_module

include("./ll_io_module.jl")
using .ll_io_module

include("./ll_analysis_module.jl")
using .ll_analysis_module

params,target = parse_args()

if target == "MGPU"
    include("./ll_sim_mgpu_module.jl")
    using .ll_sim_mgpu_module
elseif target == "GPU"
    include("./ll_sim_gpu_module.jl")
    using .ll_sim_gpu_module
elseif target == "CPU"
    include("./ll_sim_cpu_module.jl")
    using .ll_sim_cpu_module
end

# ----- SIMULATIONS
if target == "CPU"
    nx,ny,nz = @time run_sim_cpu(params,save=true)
elseif target == "GPU"
    nx,ny,nz = @time run_sim_gpu(params,save=false)
elseif target == "MGPU"
    @time run_sim_mgpu(params,save=false)
else
    println("target:$(target)")
    @error "Target not available, specify CPU, GPU, or MGPU (multi-gpu)"
end

@info "Done!"
