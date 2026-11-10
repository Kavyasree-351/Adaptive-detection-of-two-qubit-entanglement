# Adaptive detection of two-qubit entanglement

Open Simulation.html locally in a browser to run the interactive demonstration. Choose a state, noise level, and shot count, then click Start run. The held-out benchmark and chart appear below the controls.

See [the short report](Short%20Report.md) for the method and results. The data are generated from simulated quantum states.

## Reproduce the experiment

Install Python 3 and NumPy, then run:

```sh
python -m pip install -r requirements.txt
python experiment.py
```

The script regenerates the dataset, trains and evaluates the models, and writes the standalone simulation and report.
