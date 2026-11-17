"""Render the short report from held-out experiment metrics."""

def write_report(path,data):
    pct = lambda x: '—' if x is None else f'{x:.1%}'
    number = lambda x: '—' if x is None else f'{x:.2f}'
    sizes = data.get('split_sizes',dict(train=7200,validation=2400,test=3600))
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
        f"We simulated {sizes['train']:,} training, {sizes['validation']:,} validation and {sizes['test']:,} main test density matrices. The original pure/mixed/locally rotated Werner test mixture is retained. Training and validation additionally include random pure products, random mixed products, and products prepared along local Pauli axes to cover separable states absent from the original mixture. Independent draws do not imply unseen families: Pauli-axis pure endpoints can recur across splits. A separate product-family stress test uses independent draws. Werner-family sampling was concentrated near the transition. For every noise condition, global depolarization is applied before computing the PPT label and negativity. Each Pauli observable has independent binomial shots. Budgets count distinct observables, not distinct apparatus settings.",
        '',
        'Logistic regression and the one-hidden-layer neural classifier both use the same quadratic Pauli features; the random forest uses raw Pauli means. This replaces the degenerate raw-linear logistic baseline and is explicitly a different feature representation. The adaptive policy uses the neural classifier. The negativity estimator uses ridge regression with quadratic features, clipped to the physical range. One threshold is selected on validation data by minimizing mean budget subject to at least 90% coverage and answered-case accuracy within two percentage points of fixed 15. These constraints do not guarantee a small false-positive rate or calibrated scores.',
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
        return f"| {name} | {number(m['cost'])} | {pct(m['accuracy'])} | {pct(m['coverage'])} | {pct(m['fpr'])} | {err} |"
    lines += [row(f'Fixed neural ({b})',m,r) for b,m,r in zip(data['budgets'],result['fixed'],result['mae'])]
    lines += [row('Adaptive neural',a,result['adaptive_mae'])]
    for name, rows in result['baselines'].items():
        lines += [row(f'{name} ({b})',m) for b,m in zip(data['budgets'],rows)]
    lines += [
        '',
        f"The adaptive policy used **{a['cost']:.2f} observables per state**, a **{saved:.1%} reduction** against fixed 15. It answered {pct(a['coverage'])} of cases, with {pct(a['accuracy'])} accuracy among those cases versus {pct(fixed['accuracy'])} for fixed 15. False-positive rates use all separable states as the denominator, including abstentions. An operational comparison therefore needs coverage alongside accuracy and false positives.",
        '',
        f"In the {result['near_n']} near-boundary test states (absolute minimum partial-transpose eigenvalue at most 0.025), adaptive accuracy was {pct(result['near']['accuracy'])}, coverage was {pct(result['near']['coverage'])}, and the average budget was {number(result['near']['cost'])}. Empty subsets are reported as undefined, not zero.",
        '',
        '## Noise and shot sweep',
        '',
        '| Depolarization / shots | Adaptive observables | Answered accuracy | Coverage | False-positive rate | Fixed-15 accuracy |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for name,r in data['results'].items():
        m = r['adaptive']
        lines.append(f"| {name} | {m['cost']:.2f} | {pct(m['accuracy'])} | {pct(m['coverage'])} | {pct(m['fpr'])} | {pct(r['fixed'][-1]['accuracy'])} |")
    stops = ', '.join(f'{b} observables ({n:,} states)' for b,n in zip(data['budgets'],result['stop_counts']))
    if 'product_stress' in data:
        s = data['product_stress']
        lines += ['', '## Held-out product-family stress test', '',
                  f"On {data['product_stress_n']:,} independent separable preparations at 10% noise / 500 shots, false-positive rate was {pct(s['fpr'])}, coverage {pct(s['coverage'])}, and mean observable count {number(s['cost'])}. These families are now represented in training; this is not a claim of general out-of-distribution robustness.", '']
    lines += [
        '',
        '## Interpretation and limits',
        '',
        f'At 10% depolarization and 500 shots, accepted predictions stopped at {stops}. The policy abstained on {result["abstained"]:,} states.',
        '',
        f"At the main test condition, negativity mean absolute error changed from {result['mae'][0]:.4f} with three observables to {result['mae'][-1]:.4f} with fifteen. The adaptive MAE covers answered cases only. Fixed-budget MAE covers all test cases. The adaptive decision is driven by classification confidence, so the demo\u2019s negativity estimate at stopping does not have a separate uncertainty guarantee.",
        '',
        'The model scores were thresholded on validation data without a separate probability calibration step. The reported accuracy is specific to the simulated state mixture, noise channel and measurement order. The experiment uses one split and seed, and all results are from simulation.',
        '',
        'For this measurement order, budgets 3/6/9/12/15 correspond to 3/6/9/9/9 local product settings if compatible marginals share outcomes. The simulator does not implement that shared-shot protocol, so observable savings are not experimental shot or runtime savings. Review scripts include exploratory witness/PPT comparisons with heuristic finite-shot margins; those margins are not certified error bounds. Neural predictions are not entanglement certificates.',
        '',
        '## Reproduce',
        '',
        'With Python and NumPy installed, run `python experiment.py` in the source folder. The run regenerates the offline simulation and this report. Open `Simulation.html` in a browser to step through a state and inspect the held-out benchmark.',
    ]
    (path/'Short Report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
