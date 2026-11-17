# Review tests for Adaptive-detection-of-two-qubit-entanglement

Put these files anywhere outside the repo, then (needs numpy, pytest, scikit-learn, Node.js for the HTML test):

    REPO=/path/to/repo python -m pytest -q test_project.py   # 28 tests: physics, policy, report, HTML/Python parity
    python analysis.py /path/to/repo > analysis_out.json      # baselines, OOD, seeds, calibration, settings cost
    python improve.py  /path/to/repo > improve_out.json       # prototype fixes (augmentation, hybrid, asymmetric thresholds)
    python demo_dump.py /path/to/repo/Simulation.html         # drives the HTML demo headlessly for every demo state

Expected on the current repo: 25 pass, 3 fail. The 3 failures are deliberate checks for real defects:
degenerate logistic baseline, |00> called entangled by the demo, NaN crash on an empty subset.
