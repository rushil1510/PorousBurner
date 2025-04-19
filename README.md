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

<!-- no external NOx dataset is required, NO and NO2 are read directly from the simulation -->

- **run.sh**
  Bash script to execute the optimizer and append the best parameters and fitness score to `log.txt`.

- **log.txt**
  Stores the appended results from each run (ignored by Git).

## Dependencies

Install required Python packages:
```bash
pip install numpy cantera deap matplotlib
```

## Usage

1. Make sure Cantera mechanism file (e.g. gri30.yaml) is accessible.
2. Run the optimizer:
   ```bash
   ./run.sh
   ```
3. View results in `log.txt`.
