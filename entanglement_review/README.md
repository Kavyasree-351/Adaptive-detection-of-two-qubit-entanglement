# Review tests for Adaptive-detection-of-two-qubit-entanglement

These are the supplied review files, now bundled in the target repository.
Install `requirements-dev.txt` and Node.js. From the repository root:

    python -m pytest -q entanglement_review
    python entanglement_review/demo_dump.py Simulation.html

The 28 original assertions are unchanged. Six extra regression tests check
augmented state validity, all Pauli-eigenbasis products, complete product-state
demo traces, legacy model loading, and empty-subset rendering in Python and JS.
All 34 pass after the fixes. The tests automatically locate the parent repository.

Optional larger exploratory runs (needs scikit-learn; may take substantially longer):

    python entanglement_review/analysis.py . > analysis_current.json
    python entanglement_review/improve.py . > improve_current.json

Historical baseline: 25 pass, 3 fail (reproduced before edits). Those failures
identified a degenerate raw-linear logistic baseline, an overconfident product-state
prediction, and non-JSON-safe empty metrics. They are now fixed.

`analysis_out.json` and `improve_out.json` are retained exactly as supplied historical
evidence. They do not describe the corrected committed model. The exploratory
scripts deliberately retain their original training-mixture experiments and do not
reproduce the augmented production run in `experiment.py`. In particular, their
product augmentation differs from the added Pauli-axis family. Use
`experiment_results.json` and `Short Report.md` for current production metrics.

The supplied function named `rigorous_adaptive` has heuristic finite-shot margins,
not a proved finite-sample error guarantee. Unconstrained noisy linear tomography
may produce nonphysical estimates. Hybrid early stopping cannot undo an incorrect
earlier neural decision. Observable counts differ from shared local measurement
settings; the optional settings counts are resource estimates, not timings from
a shared-shot implementation. Those exploratory alternatives were not silently
adopted as guaranteed improvements.
