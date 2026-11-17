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

## Review fixes

The supplied `entanglement_review_tests.zip` is integrated in `entanglement_review/`.
The original review assertions are preserved; only their default repository path
was made portable. The initial run reproduced 25 passes and 3 failures. After the
fixes, all 28 supplied tests and 6 additional regressions pass.

- The logistic baseline now uses quadratic features, matching the neural input
  representation. It is explicitly named **quadratic logistic regression** rather
  than presented as the original raw-linear baseline.
- Training and validation include pure/mixed product states and local Pauli-axis
  preparations. The axis family includes both signs and all axes, so this is not
  a special-case override for the `|00>` demonstration. The classifier still
  receives only measured features; reference labels are not consulted at inference.
- Empty metric subsets return `null` for undefined statistics. Both the report
  and HTML display these safely.
- The standalone demo and report were regenerated from the corrected pipeline.
  `experiment_results.json` contains machine-readable current metrics.
- Review utilities no longer depend on `/home/claude` or `/tmp/h.js`.

At 10% depolarizing noise and 500 shots per observable, the updated run uses
7.38 observables on average, with 93.19% coverage, 87.81% answered-case accuracy
and 3.59% false positives over all separable cases. The independent product-family
stress test has a 7.83% false-positive rate. These are simulated results from one
seed, not certificates or universal robustness guarantees. The augmentation was
motivated by a known regression, so passing that regression is not independent
evidence of generalization. The main test state mixture remains unchanged, but
training/validation composition and random measurement draws changed; this is not
a paired causal ablation against the earlier report.

Known remaining demo limitation: the Werner mixture with p=0.36 becomes separable
after 10% depolarization, but the current neural policy still predicts entanglement
for ideal expectations and the demo's 500-shot seed-7 run. Passing regression
tests does not imply every demo or physical state is classified correctly. The
separate negativity regressor can also disagree with the classifier near zero.

## Run the review

Install Node.js for the headless HTML/Python parity tests, then run from this folder:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q entanglement_review
python entanglement_review/demo_dump.py Simulation.html
```

No `REPO` environment variable is needed for the bundled tests. It can still
override the target for an external checkout. `analysis_out.json` and
`improve_out.json` in the review directory are **historical supplied review
snapshots**, not regenerated results for this revision. See that directory's README
before running the optional exploratory analyses.
