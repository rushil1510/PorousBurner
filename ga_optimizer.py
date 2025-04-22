import os
import numpy as np
import cantera as ct
from deap import base, creator, tools, algorithms
import random
import PorousMediaBurner as pmb
import csv

# Compute heating value [J/kg]
def compute_heating_value(outlet, inlet_h):
    return outlet.enthalpy_mass - inlet_h

def calculate_nox(temperature, C1, C2):
    """Calculates NOx concentration using the provided equation."""
    return C1 * np.exp(C2 * temperature)

def load_species_data(dataset_path, species_name):
    """Loads species data from a CSV file."""
    filepath = os.path.join(dataset_path, f"{species_name}.csv")
    positions, temperatures, values = [], [], []
    with open(filepath, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            positions.append(float(row['position']))
            temperatures.append(float(row['temperature']))
            values.append(float(row['value']))
    return np.array(positions), np.array(temperatures), np.array(values)

def fit_nox_model(dataset_path, species_name):
    """Fits the NOx model to the data and returns C1 and C2."""
    positions, temperatures, nox_values = load_species_data(dataset_path, species_name)
    # Linear regression of log(NOx) vs Temperature
    log_nox = np.log(nox_values)
    A = np.vstack([temperatures, np.ones(len(temperatures))]).T
    C2, log_C1 = np.linalg.lstsq(A, log_nox, rcond=None)[0]
    C1 = np.exp(log_C1)
    return C1, C2

# Simulation wrapper: returns heating value and NOx mass fraction (NO + NO2)
def simulate(params):
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
    C1, C2 = fit_nox_model("datasets/SpeciesData", "NO")  # Example: Fit to NO data
    nox = calculate_nox(T_out, C1, C2)

    return heating, nox

# Fitness: maximize heating, minimize NOx mass fraction
def fitness_function(individual):
    heating, nox = simulate(individual)
    # penalize NOx, weight factor
    score = heating - 1e5 * nox
    return (score,)

def main():
    # run GA without external NOx dataset
    # GA setup
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    # parameter ranges: porosities [0.5, 0.95], preheat_length [0, total_length*0.5]
    total_length = sum(pmb.lengths)
    toolbox.register('attr_sic3', random.uniform, 0.5, 0.95)
    toolbox.register('attr_sic10', random.uniform, 0.5, 0.95)
    toolbox.register('attr_preheat', random.uniform, 0.0, total_length*0.5)
    toolbox.register('individual', tools.initCycle, creator.Individual,
                     (toolbox.attr_sic3, toolbox.attr_sic10, toolbox.attr_preheat), n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', fitness_function)
    toolbox.register('mate', tools.cxBlend, alpha=0.5)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=0.05, indpb=0.1)
    toolbox.register('select', tools.selTournament, tournsize=3)

    pop = toolbox.population(n=50)
    ngen = 40
    cxpb, mutpb = 0.5, 0.2
    # run GA
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb, mutpb, ngen, halloffame=hof, verbose=True)
    best = hof[0]
    best_heating, best_nox = simulate(best)
    print('Best parameters: SiC3 porosity={:.4f}, SiC10 porosity={:.4f}, Preheat length={:.4f} m'.format(*best))
    print('Heating value: {:.2f} J/kg, NOx emission: {:.3e}'.format(best_heating, best_nox))
    # compute and print fitness score
    fitness_score = best_heating - 1e5 * best_nox
    print(f'Fitness score: {fitness_score:.3f}')

if __name__ == '__main__':
    main()