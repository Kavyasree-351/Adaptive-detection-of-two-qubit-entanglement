"""Render the short report from held-out experiment metrics."""

def write_report(path,data):
    pct = lambda x: '—' if x is None else f'{x:.1%}'
    result = data['results']['10% / 500']
    a = result['adaptive']
    fixed = result['fixed'][-1]
    saved = (15-a['cost'])/15
    lines = [
        '# Adaptive detection of two-qubit entanglement',
        '',
        '## Research question',
        '',
        'Can a classifier use fewer Pauli observables per state by requesting additional measurements only when its confidence is low?',
        '',
        '## Method',
        '',
        'We simulated 7,200 training, 2,400 validation and 3,600 test density matrices. Each split contained independent random pure, random mixed and locally rotated Werner-family draws. Werner-family sampling was concentrated near the transition. The demonstration also includes an exact Bell state and a product state. For every noise condition, we applied global depolarization before computing the PPT label and negativity, so the target is the state actually measured. We simulated each Pauli observable with independent binomial shot counts. The measurement budgets count distinct observables and do not account for sharing shots across commuting observables.',
        '',
        'We compared logistic regression, a random forest and a one-hidden-layer neural classifier using quadratic Pauli features. The adaptive policy uses the neural classifier at each budget. The negativity estimator uses ridge regression with quadratic features, clipped to the physical range. We selected one confidence threshold on the validation mixture by minimizing the mean budget subject to at least 90% coverage and answered-case accuracy within two percentage points of the fixed 15-observable neural classifier.',
        '',
        f"The selected threshold was **{pct(data['threshold'])}**." if data['constrained'] else 'No scanned threshold met both validation constraints. The reported fallback threshold was 50%, which accepts every first-stage prediction.',
        '',
        '## Results at 10% depolarization and 500 shots per observable',
        '',
        '| Policy | Mean observables | Answered accuracy | Coverage | False-positive rate | Negativity MAE |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    def row(name,m,mae=None):
        err = '—' if mae is None else f'{mae:.4f}'
        return f"| {name} | {m['cost']:.2f} | {pct(m['accuracy'])} | {pct(m['coverage'])} | {pct(m['fpr'])} | {err} |"
    lines += [row(f'Fixed neural ({b})',m,r) for b,m,r in zip(data['budgets'],result['fixed'],result['mae'])]
    lines += [row('Adaptive neural',a)]
    for name, rows in result['baselines'].items():
        lines += [row(f'{name} ({b})',m) for b,m in zip(data['budgets'],rows)]
    lines += [
        '',
        f"The adaptive policy used **{a['cost']:.2f} observables per state**, a **{saved:.1%} reduction** against fixed 15. It answered {pct(a['coverage'])} of cases, with {pct(a['accuracy'])} accuracy among those cases versus {pct(fixed['accuracy'])} for fixed 15. False-positive rates use all separable states as the denominator, including abstentions. An operational comparison therefore needs coverage alongside accuracy and false positives.",
        '',
        f"In the {result['near_n']} near-boundary test states (absolute minimum partial-transpose eigenvalue at most 0.025), adaptive accuracy was {pct(result['near']['accuracy'])}, coverage was {pct(result['near']['coverage'])}, and the average budget was {result['near']['cost']:.2f}. This margin includes separable states close to the boundary as well as weakly entangled states.",
        '',
        '## Noise and shot sweep',
        '',
        '| Depolarization / shots | Adaptive observables | Answered accuracy | Coverage | False-positive rate | Fixed-15 accuracy |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for name,r in data['results'].items():
        m = r['adaptive']
        lines.append(f"| {name} | {m['cost']:.2f} | {pct(m['accuracy'])} | {pct(m['coverage'])} | {pct(m['fpr'])} | {pct(r['fixed'][-1]['accuracy'])} |")
    lines += [
        '',
        '## Interpretation and limits',
        '',
        f"At the main test condition, negativity mean absolute error changed from {result['mae'][0]:.4f} with three observables to {result['mae'][-1]:.4f} with fifteen. Quantification is reported at fixed budgets. The adaptive decision is driven by classification confidence, so the demo\u2019s negativity estimate at stopping does not have a separate uncertainty guarantee.",
        '',
        'The model scores were thresholded on validation data without a separate probability calibration step. The reported accuracy is specific to the simulated state mixture, noise channel and measurement order. The experiment uses one split and seed, and all results are from simulation.',
        '',
        '## Reproduce',
        '',
        'With Python and NumPy installed, run `python experiment.py` in the source folder. The run regenerates the offline simulation and this report. Open `Simulation.html` in a browser to step through a state and inspect the held-out benchmark.',
    ]
    (path/'Short Report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

