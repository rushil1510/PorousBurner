# PorousBurner GA Optimizer

This project uses a genetic algorithm to optimize the operating parameters of a porous media burner.

## Files

- **PorousMediaBurner.py**
  Defines the burner model and simulation of heat and species transport through a cascade of porous reactors using Cantera.

- **ga_optimizer.py**
  Implements a genetic algorithm with DEAP to search for optimal values of:
  - SiC3 porosity
  - SiC10 porosity
  - Preheating length
  The fitness function maximizes heating value and minimizes NOx emissions based on a fitted NOx model.

- **nox_data.csv**
  CSV dataset containing measured NOx emissions vs temperature (T in K, NOx in appropriate units) used to fit coefficients of the NOx model.

- **run.sh**
  Bash script to execute the optimizer and append the best parameters and fitness score to `log.txt`.

- **log.txt**
  Stores the appended results from each run (ignored by Git).

## Dependencies

Install required Python packages:
```bash
pip install numpy pandas scipy cantera deap matplotlib
```

## Usage

1. Prepare your NOx dataset as `nox_data.csv` in this directory.
2. Run the optimizer:
   ```bash
   ./run.sh
   ```
3. View results in `log.txt`.
