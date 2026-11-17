# Repository audit — 17 November 2026

## Verdict

A coherent, reproducible simulation prototype, not yet a rigorous demonstration
of general adaptive entanglement certification or a hardware-ready detector.
The implementation, saved benchmark, generated report and browser computation
agree on the checked cases. Passing tests is not a claim that all states are
classified correctly.

## Requested checks

1. **Hand calculation:** five reference density matrices have negativities
   0.5, 0, 0, 0.2, 0. A deliberately imperfect prediction vector gives exactly
   4/5 accuracy, 1/3 false-positive rate, and 2 true + 1 false = 3 positive
   admissions. All 15 expectation values are recorded in `audit_results.json`.
   Queries, distance caps, clusters and retrieval recall do not apply here.
2. **Shuffled-label control:** independently generated 900 training and 900 test
   states per seed; shuffle training labels only, retaining their prevalence.
   Train the actual quadratic logistic, neural and forest implementations over
   three seeds. Their held-out Brier errors range approximately 0.211–0.251,
   compared with constant-prior errors 0.210–0.213. No suspicious predictive
   advantage was observed. This small control does not certify absence of every
   leakage mechanism or rerun the complete adaptive threshold-selection pipeline.
   The project uses its own random forest, not ExtraTrees.
3. **Invariants:** physics tests cover normalization, positivity, partial
   transpose, known negativity, observable reconstruction and shot statistics.
   Added randomized count checks, finite/ranged adaptive input checks, matching
   metric shapes and valid costs. Saved scenario counts sum to the test size.
   Continuous-state control splits have disjoint density-matrix fingerprints.
4. **File-backed reporting:** compare every saved result with embedded HTML data,
   then regenerate the report from `experiment_results.json` and require exact
   text equality. Production report generation now explicitly reads that file.
   No separate thesis manuscript was supplied; its numbers were not audited.

## Remaining research weaknesses

- **Split claims:** training and validation use independent random draws, but
  discrete pure Pauli-axis endpoints recur. Disjoint row IDs would not solve
  identical-state leakage. These examples are known preparation families;
  unseen-family generalization needs a separately designed holdout.
- **Limited classifier:** the documented noisy near-boundary Werner example is
  still misclassified. Product stress-test FPR is 7.83%. Predictions are not
  physics certificates and scores have not been probability-calibrated.
- **Limited statistical evidence:** production results use one seed and one
  simulated mixture. Repeated seeds, uncertainty intervals, and independently
  held-out state families are still required for strong performance claims.
- **Cost interpretation:** stages use a fixed measurement ordering. This is
  adaptive stopping, not learned choice of the next observable. Observable
  reductions are not automatically reductions in apparatus settings or time.
- **Baseline comparability:** the neural/logistic models use quadratic features;
  the forest uses raw inputs. Feature parity is not complete across all models.
- **Artifacts:** review `analysis_out.json` and `improve_out.json` are historical
  snapshots, clearly documented as such. README summary numbers are hand-written
  and may become stale after retraining; the generated report is authoritative.
- **Reproducibility:** broad dependency requirements, no automated CI and no
  saved per-observation production split manifest or bootstrap intervals.
- **Presentation:** headless Node checks validate controls, output logic and
  Python parity; they do not establish visual layout quality across browsers.

## Reproduce

```sh
python -m pip install -r requirements-dev.txt
python entanglement_review/test_audit.py
python -m pytest -q entanglement_review
```

The first command after installation writes the numerical audit artifact.
The tests run all original checks plus the added audit and regression checks.
