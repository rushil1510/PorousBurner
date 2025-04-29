# Porous Burner Optimization

This project implements optimization algorithms to find optimal parameters for a porous media burner, specifically focusing on:
- SiC3 porosity
- SiC10 porosity
- Preheating length

## Files

- **PorousMediaBurner.py**
  Defines the burner model and simulation of heat and species transport through a cascade of porous reactors using Cantera.

- **optimizer_comparison.py**
  Implements both genetic algorithm and gradient descent optimization with multiple fitness functions:
  - F = T - a*NOx
  - F = T/(1 + a*NOx)
  - F = T - a*log(NOx)
  where T is the heating value and NOx is the emissions.
  
  The optimization is run for different values of a (NOx penalty weight):
  - a = [1, 100, 1000, 10000, 100000, 1000000]

- **parse_dataset_to_csv.py**
  Converts the dataset from Excel format to CSV for easier processing.

- **results.txt**
  Stores the optimization results with timestamps, including:
  - Optimization method (GA or GD)
  - a value
  - Fitness function used
  - Optimal parameters
  - Heating value and NOx emissions

## Dependencies

Install required Python packages:
```bash
pip install numpy cantera deap matplotlib scipy
```

## Usage

1. Make sure Cantera mechanism file (e.g. gri30.yaml) is accessible.
2. Convert the dataset to CSV format:
   ```bash
   python parse_dataset_to_csv.py
   ```
3. Run the optimization comparison:
   ```bash
   python optimizer_comparison.py
   ```
4. View results in `results.txt`.

## Optimization Methods

### Genetic Algorithm (GA)
- Population size: 50
- Generations: 40
- Crossover probability: 0.5
- Mutation probability: 0.2
- Tournament selection with size 3
- Blend crossover with alpha=0.5
- Gaussian mutation with sigma=0.05

### Gradient Descent (GD)
- Uses L-BFGS-B method
- Bounded optimization
- Initial guess from GA results
- Maximum iterations: 1000

## Results Format

Each line in `results.txt` follows this format:
```
Timestamp | Method | a | Fitness Function | Parameters | Results
```

Where:
- Timestamp: When the optimization was run
- Method: GA (Genetic Algorithm) or GD (Gradient Descent)
- a: Weight factor for NOx penalty
- Fitness Function: Which fitness function was used
- Parameters: Optimal values for SiC3 porosity, SiC10 porosity, and preheat length
- Results: Heating value and NOx emissions
