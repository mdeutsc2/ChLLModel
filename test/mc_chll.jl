using WriteVTK
using Rotations

# Define the parameters
L = 20                # Lattice size (LxL)
J = 1.0               # Interaction strength
T = 1.0               # Temperature
num_steps = 100000    # Number of Monte Carlo steps
plot_interval = 10000 # Interval for plotting

function normalize(nx,ny,nz)
    s = sqrt.(nx.*nx .+ ny.*ny .+ nz.*nz)
    nx = nx./s
    ny = ny./s
    nz = nz./s
    return nx,ny,nz
end

function random_unit_vector()
    return rotation_axis(RotXYZ(rand(),rand(),rand()))
end

function init(L)
    nx = zeros(L,L,L)
    ny = zeros(L,L,L)
    nz = zeros(L,L,L)
    for i in 1:L
        for j in 1:L
            for k in 1:L
                r = random_unit_vector()
                nx[i,j,k] = r[1]
                ny[i,j,k] = r[2]
                nz[i,j,k] = r[3]
            end
        end
    end
    return nx,ny,nz
end

# Periodic boundary conditions
function pbc(i, L)
    return (i - 1) % L + 1
end

# Compute the energy change for a trial orientation
function delta_E(nx,ny,nz, x, y, z, trial_spin)
    dE = 0.0
    current_spin = lattice[x, y]
    
    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)]
        neighbor_spin = lattice[pbc(x + dx, L), pbc(y + dy, L)]
        dE += J * ((dot(trial_spin, neighbor_spin))^2 - (dot(current_spin, neighbor_spin))^2)
    end
    
    return dE
end

# Metropolis algorithm
function run_sim(T,num_steps)
    nx,ny,nz = init(L)
    
    for step in 1:num_steps
        x = rand(1:L)
        y = rand(1:L)
        z = rand(1:L)
        trial_spin = random_unit_vector()
        dE = ΔE(nx,ny,nz, x, y, z, trial_spin)
        
        if dE <= 0 || rand() < exp(-dE / T)
            lattice[x, y] = trial_spin
        end
        
        # if step % plot_interval == 0
        #     write_out()
        # end
    end
end

# Run the simulation
runsim(T,num_steps)


