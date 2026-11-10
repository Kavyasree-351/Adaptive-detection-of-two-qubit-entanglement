# Adaptive detection of two-qubit entanglement

## Research question

Can a classifier use fewer Pauli observables per state by requesting additional measurements only when its confidence is low?

## Method

We simulated 7,200 training, 2,400 validation and 3,600 test density matrices. Each split contained independent random pure, random mixed and locally rotated Werner-family draws. Werner-family sampling was concentrated near the transition. The demonstration also includes an exact Bell state and a product state. For every noise condition, we applied global depolarization before computing the PPT label and negativity, so the target is the state actually measured. We simulated each Pauli observable with independent binomial shot counts. The measurement budgets count distinct observables and do not account for sharing shots across commuting observables.

We compared logistic regression, a random forest and a one-hidden-layer neural classifier using quadratic Pauli features. The adaptive policy uses the neural classifier at each budget. The negativity estimator uses ridge regression with quadratic features, clipped to the physical range. We selected one confidence threshold on the validation mixture by minimizing the mean budget subject to at least 90% coverage and answered-case accuracy within two percentage points of the fixed 15-observable neural classifier.

The selected threshold was **69.0%**.

## Results at 10% depolarization and 500 shots per observable

| Policy | Mean observables | Answered accuracy | Coverage | False-positive rate | Negativity MAE |
|---|---:|---:|---:|---:|---:|
| Fixed neural (3) | 3.00 | 79.1% | 100.0% | 34.1% | 0.0746 |
| Fixed neural (6) | 6.00 | 88.0% | 100.0% | 19.8% | 0.0366 |
| Fixed neural (9) | 9.00 | 92.0% | 100.0% | 12.8% | 0.0180 |
| Fixed neural (12) | 12.00 | 91.3% | 100.0% | 11.0% | 0.0177 |
| Fixed neural (15) | 15.00 | 91.0% | 100.0% | 15.2% | 0.0179 |
| Adaptive neural | 4.80 | 89.8% | 96.6% | 17.9% | — |
| Logistic regression (3) | 3.00 | 69.1% | 100.0% | 100.0% | — |
| Logistic regression (6) | 6.00 | 69.1% | 100.0% | 100.0% | — |
| Logistic regression (9) | 9.00 | 69.1% | 100.0% | 100.0% | — |
| Logistic regression (12) | 12.00 | 69.1% | 100.0% | 100.0% | — |
| Logistic regression (15) | 15.00 | 69.1% | 100.0% | 100.0% | — |
| Random forest (3) | 3.00 | 79.0% | 100.0% | 39.5% | — |
| Random forest (6) | 6.00 | 86.4% | 100.0% | 29.7% | — |
| Random forest (9) | 9.00 | 89.2% | 100.0% | 26.7% | — |
| Random forest (12) | 12.00 | 88.2% | 100.0% | 31.7% | — |
| Random forest (15) | 15.00 | 87.6% | 100.0% | 34.1% | — |

The adaptive policy used **4.80 observables per state**, a **68.0% reduction** against fixed 15. It answered 96.6% of cases, with 89.8% accuracy among those cases versus 91.0% for fixed 15. False-positive rates use all separable states as the denominator, including abstentions. An operational comparison therefore needs coverage alongside accuracy and false positives.

In the 683 near-boundary test states (absolute minimum partial-transpose eigenvalue at most 0.025), adaptive accuracy was 66.0%, coverage was 89.5%, and the average budget was 6.69. This margin includes separable states close to the boundary as well as weakly entangled states.

## Noise and shot sweep

| Depolarization / shots | Adaptive observables | Answered accuracy | Coverage | False-positive rate | Fixed-15 accuracy |
|---|---:|---:|---:|---:|---:|
| Ideal | 4.57 | 91.3% | 97.3% | 15.5% | 92.8% |
| 0% / 100 | 4.42 | 87.6% | 97.7% | 34.2% | 89.1% |
| 0% / 500 | 4.51 | 90.2% | 97.6% | 19.6% | 92.0% |
| 0% / 1,000 | 4.54 | 90.9% | 97.2% | 17.3% | 92.5% |
| 0% / 5,000 | 4.58 | 91.4% | 97.3% | 16.2% | 92.6% |
| 2% / 100 | 4.44 | 86.3% | 97.6% | 37.8% | 89.1% |
| 2% / 500 | 4.62 | 89.7% | 97.0% | 20.0% | 91.7% |
| 2% / 1,000 | 4.64 | 90.9% | 96.7% | 16.5% | 92.7% |
| 2% / 5,000 | 4.62 | 91.0% | 97.1% | 16.3% | 92.4% |
| 5% / 100 | 4.53 | 86.0% | 97.3% | 34.8% | 87.7% |
| 5% / 500 | 4.69 | 89.7% | 96.9% | 20.0% | 91.3% |
| 5% / 1,000 | 4.72 | 90.5% | 96.4% | 17.6% | 92.1% |
| 5% / 5,000 | 4.71 | 91.1% | 96.6% | 16.1% | 92.2% |
| 10% / 100 | 4.66 | 84.7% | 97.2% | 34.1% | 86.6% |
| 10% / 500 | 4.80 | 89.8% | 96.6% | 17.9% | 91.0% |
| 10% / 1,000 | 4.78 | 90.0% | 96.4% | 17.0% | 91.9% |
| 10% / 5,000 | 4.87 | 90.5% | 96.0% | 15.7% | 92.1% |
| 15% / 100 | 4.77 | 83.4% | 97.1% | 33.4% | 84.6% |
| 15% / 500 | 4.89 | 89.1% | 96.3% | 17.0% | 90.8% |
| 15% / 1,000 | 4.91 | 89.5% | 96.3% | 16.0% | 90.8% |
| 15% / 5,000 | 4.96 | 90.2% | 95.8% | 14.4% | 91.5% |
| 20% / 100 | 4.87 | 82.5% | 96.7% | 32.0% | 83.6% |
| 20% / 500 | 5.05 | 88.6% | 95.9% | 16.5% | 90.3% |
| 20% / 1,000 | 4.98 | 89.5% | 96.4% | 14.2% | 91.8% |
| 20% / 5,000 | 5.02 | 90.3% | 96.5% | 12.8% | 92.3% |

## Interpretation and limits

At the main test condition, negativity mean absolute error changed from 0.0746 with three observables to 0.0179 with fifteen. Quantification is reported at fixed budgets. The adaptive decision is driven by classification confidence, so the demo’s negativity estimate at stopping does not have a separate uncertainty guarantee.

The model scores were thresholded on validation data without a separate probability calibration step. The reported accuracy is specific to the simulated state mixture, noise channel and measurement order. The experiment uses one split and seed, and all results are from simulation.

## Reproduce

With Python and NumPy installed, run `python experiment.py` in the source folder. The run regenerates the offline simulation and this report. Open `Simulation.html` in a browser to step through a state and inspect the held-out benchmark.
