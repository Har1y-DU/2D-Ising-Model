using Distributed
addprocs(31 - nprocs())  # Add workers if needed
@everywhere using Random, PyCall, HDF5
@everywhere @pyimport matplotlib.pyplot as plt


# Define exchange constants
@everywhere const Jx = 1.0     # Coupling constant in x-direction
@everywhere const Jy = 1.0     # Coupling constant in y-direction

# Define parameters
@everywhere const nt = 2000
@everywhere const eqSteps = 100000
@everywhere const mcSteps = 10000
@everywhere Nx = 20
@everywhere Ny = 20
@everywhere T = range(3.5, stop=1.3, length=nt)

@everywhere function initialstate(seed=100)
    Random.seed!(seed)  # Set a fixed seed for repeatability
    return 2 * rand(Bool, Nx, Ny) .- 1
end

# Energy contribution of a spin with its neighbors
@everywhere function E_ising_model(S, neighbors)
    nb_left, nb_right, nb_up, nb_down = neighbors
    energy = -S * (Jx * (nb_left + nb_right) + Jy * (nb_up + nb_down))
    return energy
end

# Neighbor list with explicit neighbors
@everywhere function nb_list_2D(config, a, b)
    # Nearest neighbors with periodic boundary conditions
    nb_right = config[mod1(a+1, Nx), b]
    nb_left  = config[mod1(a-1, Nx), b]
    nb_up    = config[a, mod1(b+1, Ny)]
    nb_down  = config[a, mod1(b-1, Ny)]

    return nb_left, nb_right, nb_up, nb_down
end

@everywhere function mcmove(config, beta)
    """Monte Carlo move using Metropolis algorithm"""

    for _ in 1:(Nx * Ny)
        a = rand(1:Nx)
        b = rand(1:Ny)
        S = config[a, b]  # Current spin at position (a, b)

        neighbors = nb_list_2D(config, a, b)

        dE = E_ising_model(-S, neighbors) - E_ising_model(S, neighbors)

        # Metropolis criterion
        if (dE < 0)
            config[a, b] = -S
        elseif (rand() < exp(-dE * beta))
            config[a, b] = -S
        end
    end
    return config
end

@everywhere function calcEnergy(config)
    """Energy of a given configuration"""
    energy = 0.0

    for i in 1:Nx
        for j in 1:Ny
            S = config[i, j]  # Spin at position (i, j)
            neighbors = nb_list_2D(config, i, j)  # Get the list of neighbors
            energy += E_ising_model(S, neighbors)
        end
    end
    return energy * 0.5  # Factor of 1/2 to account for double counting of pairs
end

@everywhere function calcMag(config)
    """Magnetization of a given configuration"""
    return sum(config)
end

# Parallelized main function: each worker processes a different temperature
function main()
    E_vals, M_vals, C_vals, X_vals = Float64[], Float64[], Float64[], Float64[]  # Store Energy and Magnetization

    total_start_time = time()

    # Parallel processing for each temperature
    results = @distributed (vcat) for ti in 1:nt
        T_val = T[ti]
        beta = 1 / T_val
        norm = 1 / mcSteps

        println("Working on temperature: T=$(T_val)")

        # Step 1: Equilibrate
        config = initialstate()  # Initial random state
        for _ in 1:eqSteps
            config = mcmove(config, beta)
        end

        # Step 2: Monte Carlo sampling
        E_local, M_local, E2_local, M2_local = 0.0, 0.0, 0.0, 0.0
        for _ in 1:mcSteps
            config = mcmove(config, beta)
            E = calcEnergy(config)
            M = abs(calcMag(config))
            E_local += E
            M_local += M
            E2_local += E^2
            M2_local += M^2
        end

        # Return averages and squared averages for this temperature
        E_avg = E_local * norm
        M_avg = M_local * norm
        E2_avg = E2_local * norm
        M2_avg = M2_local * norm

        C = (E2_avg - E_avg^2) * beta^2
        X = (M2_avg - M_avg^2) * beta

        [E_avg, M_avg, C, X]
    end

    for i in 1:4:length(results)
        push!(E_vals, results[i])          # Energy
        push!(M_vals, results[i + 1])      # Magnetization
        push!(C_vals, results[i + 2])     # Specific Heat
        push!(X_vals, results[i + 3])     # Susceptibility
    end

    total_end_time = time()  # End timing here
    total_time = total_end_time - total_start_time  # Compute total time for this lattice size
    println("Total time: $(total_time) seconds")

    return E_vals, M_vals, C_vals, X_vals, total_time
end

# Function to plot results for E, M, C, X
function plot_results(E_values, M_values, C_values, X_values, T, dot_size=2)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    plt.figure(figsize=(12, 8))

    # Plot Energy
    plt.subplot(2, 2, 1)
    plt.scatter(T, E_values, s=dot_size)
    plt.xlabel("Temperature")
    plt.ylabel("Energy")

    # Plot Magnetization
    plt.subplot(2, 2, 2)
    plt.scatter(T, M_values, s=dot_size)
    plt.xlabel("Temperature")
    plt.ylabel("Magnetization")

    # Plot Specific Heat
    plt.subplot(2, 2, 3)
    plt.scatter(T, C_values, s=dot_size)
    plt.xlabel("Temperature")
    plt.ylabel("Specific Heat")

    # Plot Susceptibility
    plt.subplot(2, 2, 4)
    plt.scatter(T, X_values, s=dot_size)
    plt.xlabel("Temperature")
    plt.ylabel("Susceptibility")

    plt.tight_layout()
    plt.savefig("2D_ising_N20_plot.png")
    plt.show()
end

# Function to save results to an HDF5 file
function save_results(E_vals, M_vals, C_vals, X_vals, total_time, filename="2D_ising_results_N20.h5")
    h5open(filename, "w") do file
        file["Energy"] = E_vals
        file["Magnetization"] = M_vals
        file["Specific_Heat"] = C_vals
        file["Susceptibility"] = X_vals
        file["Total_Time"] = total_time
    end
end

# Example usage
E_vals, M_vals, C_vals, X_vals, total_time = main()
plot_results(E_vals, M_vals, C_vals, X_vals, T)
save_results(E_vals, M_vals, C_vals, X_vals, total_time)