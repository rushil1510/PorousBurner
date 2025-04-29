import os
import numpy as np
import cantera as ct
from deap import base, creator, tools, algorithms
import random
import PorousMediaBurner as pmb
import csv
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

# Constants and parameters
A_VALUES = [1, 100, 1000, 10000, 100000, 1000000]  # Different weights for NOx penalty
RESULTS_FILE = "results.txt"

def compute_heating_value(outlet, inlet_h):
    """Compute heating value [J/kg] from outlet and inlet enthalpy."""
    return outlet.enthalpy_mass - inlet_h

def calculate_nox(temperature, C1, C2):
    """Calculate NOx concentration using the exponential model."""
    return C1 * np.exp(C2 * temperature)

def load_species_data(dataset_path, species_name):
    """Loads species data from a CSV file."""
    filepath = os.path.join(dataset_path, f"{species_name}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    positions, temperatures, values = [], [], []
    with open(filepath, 'r') as csvfile:
        lines = csvfile.readlines()
        # Find the NO section
        no_section_start = -1
        for i, line in enumerate(lines):
            if "NO" in line:
                no_section_start = i
                break
        
        if no_section_start == -1:
            raise ValueError("NO section not found in the data file")
        
        # Skip the header line after NO section
        no_section_start += 2
        
        # Parse the NO data
        for line in lines[no_section_start:]:
            try:
                parts = line.strip().split(',')
                if len(parts) != 3:
                    continue
                positions.append(float(parts[0]))
                temperatures.append(float(parts[1]))
                values.append(float(parts[2]))
            except (ValueError, IndexError):
                continue
                
    return np.array(positions), np.array(temperatures), np.array(values)

def fit_nox_model(dataset_path, species_name):
    """Fit NOx model to data and return C1 and C2 parameters."""
    positions, temperatures, nox_values = load_species_data(dataset_path, species_name)
    # Linear regression of log(NOx) vs Temperature
    log_nox = np.log(nox_values)
    A = np.vstack([temperatures, np.ones(len(temperatures))]).T
    C2, log_C1 = np.linalg.lstsq(A, log_nox, rcond=None)[0]
    C1 = np.exp(log_C1)
    return C1, C2

def simulate(params):
    """Run burner simulation with given parameters and return heating value and NOx."""
    sic3_porosity, sic10_porosity, preheat_length = params
    # override porosity
    pmb.SiC3PPI.porosity = sic3_porosity
    pmb.SiC10PPI.porosity = sic10_porosity
    # build reactor cascade with variable preheat length
    lengths = pmb.lengths  # list of section lengths
    total_length = sum(lengths)
    reactors = []
    sum_len = 0.0
    for L in lengths:
        midpoint = sum_len + 0.5 * L
        sum_len += L
        if midpoint < preheat_length:
            props = pmb.ReactorProperties(
                diameter=2 * pmb.inch,
                length=L,
                midpoint=midpoint,
                chemistry=False,
                TsInit=pmb.TsInit,
                solid=pmb.YZA40PPI)
            gas = pmb.gas_init
        elif midpoint < total_length * 0.75:
            props = pmb.ReactorProperties(
                diameter=2 * pmb.inch,
                length=L,
                midpoint=midpoint,
                chemistry=True,
                TsInit=pmb.gas_hot.T,
                solid=pmb.SiC3PPI)
            gas = pmb.gas_hot
        else:
            props = pmb.ReactorProperties(
                diameter=2 * pmb.inch,
                length=L,
                midpoint=midpoint,
                chemistry=True,
                TsInit=pmb.gas_hot.T,
                solid=pmb.SiC10PPI)
            gas = pmb.gas_hot
        reactors.append(pmb.PMReactor(gas, props=props))
    # set neighbors
    for i, r in enumerate(reactors):
        if i > 0:
            r.neighbor_left = reactors[i-1]
        if i < len(reactors)-1:
            r.neighbor_right = reactors[i+1]
    # run net
    net = ct.ReactorNet(reactors)
    net.max_steps = 100000
    net.atol = 1e-9
    net.rtol = 1e-4
    net.atol_sensitivity = 1e-3
    net.rtol_sensitivity = 0.05
    net.advance_to_steady_state()
    # outlet gas and metrics
    outlet = reactors[-1].thermo
    heating = compute_heating_value(outlet, pmb.h_in_inlet)
    T_out = outlet.T

    # Use the fitted NOx model
    C1, C2 = fit_nox_model("Datasets", "Ch4_80_mass")  # Using available dataset
    nox = calculate_nox(T_out, C1, C2)

    return heating, nox

def fitness_function_1(individual, a):
    """Fitness function 1: F = T - a*NOx"""
    heating, nox = simulate(individual)
    return (heating - a * nox,)

def fitness_function_2(individual, a):
    """Fitness function 2: F = T/(1 + a*NOx)"""
    heating, nox = simulate(individual)
    return (heating / (1 + a * nox),)

def fitness_function_3(individual, a):
    """Fitness function 3: F = T - a*log(NOx)"""
    heating, nox = simulate(individual)
    return (heating - a * np.log(nox),)

def run_genetic_algorithm(a, fitness_func, pop_size=50, ngen=40):
    """Run genetic algorithm optimization with given parameters."""
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    
    total_length = sum(pmb.lengths)
    toolbox.register('attr_sic3', random.uniform, 0.5, 0.95)
    toolbox.register('attr_sic10', random.uniform, 0.5, 0.95)
    toolbox.register('attr_preheat', random.uniform, 0.0, total_length*0.5)
    toolbox.register('individual', tools.initCycle, creator.Individual,
                     (toolbox.attr_sic3, toolbox.attr_sic10, toolbox.attr_preheat), n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', fitness_func, a=a)
    toolbox.register('mate', tools.cxBlend, alpha=0.5)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=0.05, indpb=0.1)
    toolbox.register('select', tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    cxpb, mutpb = 0.5, 0.2
    
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb, mutpb, ngen, halloffame=hof, verbose=True)
    return hof[0]

def run_gradient_descent(a, fitness_func, initial_guess=None):
    """Run gradient descent optimization with given parameters."""
    if initial_guess is None:
        initial_guess = [0.7, 0.7, sum(pmb.lengths)*0.25]  # Default initial guess
    
    def objective(params):
        return -fitness_func(params, a)[0]  # Negative because we want to maximize
    
    bounds = [(0.5, 0.95), (0.5, 0.95), (0.0, sum(pmb.lengths)*0.5)]
    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
    return result.x

def save_results(method, a, params, fitness_func_name):
    """Save optimization results to file."""
    heating, nox = simulate(params)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(RESULTS_FILE, 'a') as f:
        f.write(f"{timestamp} | {method} | a={a} | {fitness_func_name} | "
                f"SiC3={params[0]:.4f}, SiC10={params[1]:.4f}, "
                f"Preheat={params[2]:.4f} | Heating={heating:.2f}, NOx={nox:.3e}\n")

def main():
    # Clear results file
    with open(RESULTS_FILE, 'w') as f:
        f.write("Timestamp | Method | a | Fitness Function | Parameters | Results\n")
    
    fitness_functions = [
        (fitness_function_1, "F = T - a*NOx"),
        (fitness_function_2, "F = T/(1 + a*NOx)"),
        (fitness_function_3, "F = T - a*log(NOx)")
    ]
    
    for a in A_VALUES:
        for fitness_func, func_name in fitness_functions:
            print(f"\nRunning optimization with a={a} and {func_name}")
            
            # Run genetic algorithm
            print("\nRunning Genetic Algorithm...")
            ga_params = run_genetic_algorithm(a, fitness_func)
            save_results("GA", a, ga_params, func_name)
            
            # Run gradient descent
            print("\nRunning Gradient Descent...")
            gd_params = run_gradient_descent(a, fitness_func, initial_guess=ga_params)
            save_results("GD", a, gd_params, func_name)
            
            # Compare results
            ga_heating, ga_nox = simulate(ga_params)
            gd_heating, gd_nox = simulate(gd_params)
            
            print(f"\nComparison for a={a} and {func_name}:")
            print(f"GA: Heating={ga_heating:.2f}, NOx={ga_nox:.3e}")
            print(f"GD: Heating={gd_heating:.2f}, NOx={gd_nox:.3e}")

if __name__ == '__main__':
    main() 