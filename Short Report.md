# Adaptive detection of two-qubit entanglement

## Research question

Can a classifier use fewer Pauli observables per state by requesting additional measurements only when its confidence is low?

## Method

We simulated 10,800 training, 3,600 validation and 3,600 main test density matrices. The original pure/mixed/locally rotated Werner test mixture is retained. Training and validation additionally include random pure products, random mixed products, and products prepared along local Pauli axes to cover separable states absent from the original mixture. Independent draws do not imply unseen families: Pauli-axis pure endpoints can recur across splits. A separate product-family stress test uses independent draws. Werner-family sampling was concentrated near the transition. For every noise condition, global depolarization is applied before computing the PPT label and negativity. Each Pauli observable has independent binomial shots. Budgets count distinct observables, not distinct apparatus settings.

Logistic regression and the one-hidden-layer neural classifier both use the same quadratic Pauli features; the random forest uses raw Pauli means. This replaces the degenerate raw-linear logistic baseline and is explicitly a different feature representation. The adaptive policy uses the neural classifier. The negativity estimator uses ridge regression with quadratic features, clipped to the physical range. One threshold is selected on validation data by minimizing mean budget subject to at least 90% coverage and answered-case accuracy within two percentage points of fixed 15. These constraints do not guarantee a small false-positive rate or calibrated scores.

The selected threshold was **75.0%**.

## Results at 10% depolarization and 500 shots per observable

| Policy | Mean observables | Answered accuracy | Coverage | False-positive rate | Negativity MAE |
|---|---:|---:|---:|---:|---:|
| Fixed neural (3) | 3.00 | 65.9% | 100.0% | 7.6% | 0.0745 |
| Fixed neural (6) | 6.00 | 76.0% | 100.0% | 1.7% | 0.0409 |
| Fixed neural (9) | 9.00 | 77.2% | 100.0% | 0.5% | 0.0245 |
| Fixed neural (12) | 12.00 | 87.4% | 100.0% | 11.8% | 0.0251 |
| Fixed neural (15) | 15.00 | 87.2% | 100.0% | 16.0% | 0.0258 |
| Adaptive neural | 7.38 | 87.8% | 93.2% | 3.6% | 0.0421 |
| Quadratic logistic regression (3) | 3.00 | 58.0% | 100.0% | 0.7% | — |
| Quadratic logistic regression (6) | 6.00 | 69.0% | 100.0% | 0.4% | — |
| Quadratic logistic regression (9) | 9.00 | 73.0% | 100.0% | 0.3% | — |
| Quadratic logistic regression (12) | 12.00 | 79.4% | 100.0% | 3.0% | — |
| Quadratic logistic regression (15) | 15.00 | 82.1% | 100.0% | 5.9% | — |
| Random forest (3) | 3.00 | 68.3% | 100.0% | 15.0% | — |
| Random forest (6) | 6.00 | 80.3% | 100.0% | 8.5% | — |
| Random forest (9) | 9.00 | 86.9% | 100.0% | 6.5% | — |
| Random forest (12) | 12.00 | 86.5% | 100.0% | 6.3% | — |
| Random forest (15) | 15.00 | 86.8% | 100.0% | 6.7% | — |

The adaptive policy used **7.38 observables per state**, a **50.8% reduction** against fixed 15. It answered 93.2% of cases, with 87.8% accuracy among those cases versus 87.2% for fixed 15. False-positive rates use all separable states as the denominator, including abstentions. An operational comparison therefore needs coverage alongside accuracy and false positives.

In the 683 near-boundary test states (absolute minimum partial-transpose eigenvalue at most 0.025), adaptive accuracy was 63.6%, coverage was 83.6%, and the average budget was 9.48. Empty subsets are reported as undefined, not zero.

## Noise and shot sweep

| Depolarization / shots | Adaptive observables | Answered accuracy | Coverage | False-positive rate | Fixed-15 accuracy |
|---|---:|---:|---:|---:|---:|
| Ideal | 7.30 | 87.3% | 94.9% | 2.4% | 88.5% |
| 0% / 100 | 7.40 | 86.5% | 95.2% | 14.3% | 85.1% |
| 0% / 500 | 7.27 | 87.0% | 95.1% | 3.7% | 87.2% |
| 0% / 1,000 | 7.34 | 87.3% | 94.6% | 3.4% | 88.0% |
| 0% / 5,000 | 7.34 | 87.3% | 94.2% | 2.8% | 88.1% |
| 2% / 100 | 7.43 | 85.6% | 94.4% | 15.6% | 84.9% |
| 2% / 500 | 7.43 | 87.0% | 94.4% | 4.1% | 86.9% |
| 2% / 1,000 | 7.35 | 87.2% | 94.5% | 3.2% | 87.4% |
| 2% / 5,000 | 7.32 | 87.2% | 94.6% | 2.3% | 87.6% |
| 5% / 100 | 7.46 | 85.1% | 95.1% | 15.7% | 83.8% |
| 5% / 500 | 7.39 | 87.4% | 94.1% | 4.6% | 86.7% |
| 5% / 1,000 | 7.40 | 87.4% | 93.6% | 3.2% | 87.4% |
| 5% / 5,000 | 7.34 | 87.5% | 94.2% | 2.4% | 87.8% |
| 10% / 100 | 7.62 | 85.1% | 94.1% | 12.8% | 82.5% |
| 10% / 500 | 7.38 | 87.8% | 93.2% | 3.6% | 87.2% |
| 10% / 1,000 | 7.36 | 88.1% | 93.7% | 3.0% | 87.7% |
| 10% / 5,000 | 7.31 | 87.9% | 93.6% | 1.7% | 87.9% |
| 15% / 100 | 7.56 | 86.2% | 93.5% | 11.5% | 82.2% |
| 15% / 500 | 7.36 | 87.4% | 93.2% | 3.4% | 86.4% |
| 15% / 1,000 | 7.32 | 88.7% | 92.9% | 1.7% | 87.8% |
| 15% / 5,000 | 7.30 | 88.4% | 92.6% | 1.3% | 87.8% |
| 20% / 100 | 7.60 | 84.8% | 93.3% | 12.6% | 79.2% |
| 20% / 500 | 7.29 | 88.8% | 93.2% | 2.2% | 87.1% |
| 20% / 1,000 | 7.25 | 89.3% | 92.6% | 1.4% | 87.7% |
| 20% / 5,000 | 7.31 | 89.9% | 92.2% | 0.8% | 88.4% |

## Held-out product-family stress test

On 1,800 independent separable preparations at 10% noise / 500 shots, false-positive rate was 7.8%, coverage 95.8%, and mean observable count 7.93. These families are now represented in training; this is not a claim of general out-of-distribution robustness.


## Interpretation and limits

At 10% depolarization and 500 shots, accepted predictions stopped at 3 observables (950 states), 6 observables (1,221 states), 9 observables (601 states), 12 observables (479 states), 15 observables (104 states). The policy abstained on 245 states.

At the main test condition, negativity mean absolute error changed from 0.0745 with three observables to 0.0258 with fifteen. The adaptive MAE covers answered cases only. Fixed-budget MAE covers all test cases. The adaptive decision is driven by classification confidence, so the demo’s negativity estimate at stopping does not have a separate uncertainty guarantee.

The model scores were thresholded on validation data without a separate probability calibration step. The reported accuracy is specific to the simulated state mixture, noise channel and measurement order. The experiment uses one split and seed, and all results are from simulation.

For this measurement order, budgets 3/6/9/12/15 correspond to 3/6/9/9/9 local product settings if compatible marginals share outcomes. The simulator does not implement that shared-shot protocol, so observable savings are not experimental shot or runtime savings. Review scripts include exploratory witness/PPT comparisons with heuristic finite-shot margins; those margins are not certified error bounds. Neural predictions are not entanglement certificates.

## Reproduce

With Python and NumPy installed, run `python experiment.py` in the source folder. The run regenerates the offline simulation and this report. Open `Simulation.html` in a browser to step through a state and inspect the held-out benchmark.
