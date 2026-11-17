"""Prototype fixes. Usage: python improve.py REPO_DIR"""
import sys, json
import numpy as np
sys.path.insert(0, sys.argv[1])
import experiment as ex
import analysis as an

B = ex.BUDGETS


def separable_states(n, R):
    u = lambda k: R.normal(size=(k, 2)) + 1j * R.normal(size=(k, 2))
    a, b = u(n), u(n)
    a /= np.linalg.norm(a, axis=1, keepdims=True); b /= np.linalg.norm(b, axis=1, keepdims=True)
    v = np.einsum('ni,nj->nij', a, b).reshape(n, 4)
    pure = np.einsum('ni,nj->nij', v, v.conj())
    g1 = R.normal(size=(n, 2, 2)) + 1j * R.normal(size=(n, 2, 2))
    g2 = R.normal(size=(n, 2, 2)) + 1j * R.normal(size=(n, 2, 2))
    r1 = g1 @ g1.conj().swapaxes(1, 2); r1 /= np.trace(r1, axis1=1, axis2=2)[:, None, None]
    r2 = g2 @ g2.conj().swapaxes(1, 2); r2 /= np.trace(r2, axis1=1, axis2=2)[:, None, None]
    mixed = np.einsum('nij,nkl->nikjl', r1, r2).reshape(n, 4, 4)
    return np.concatenate([pure, mixed])


def pipeline(augment, seed=42):
    ex.RNG = np.random.default_rng(seed); R = ex.RNG
    tr, va, te = (ex.states(n) for n in (2400, 800, 1200))
    if augment:
        tr = np.concatenate([tr, separable_states(1200, R)])
        va = np.concatenate([va, separable_states(400, R)])
    x, neg, _ = ex.physical(tr, R.choice(ex.NOISE, len(tr)), R.choice(ex.SHOTS, len(tr)), R)
    y = neg > 1e-9
    models = [ex.fit(x[:, :b], y, 32, 40 + b) for b in B]
    xv, nv, _ = ex.physical(va, R.choice(ex.NOISE, len(va)), R.choice(ex.SHOTS, len(va)), R)
    vy = nv > 1e-9
    vp = [ex.probability(m, xv[:, :b]) for b, m in zip(B, models)]
    f15 = ((vp[-1] >= .5) == vy).mean()
    cands = []
    for t in np.arange(.51, .971, .01):
        pr, co = ex.adaptive(vp, t); s = ex.metrics(pr, vy, co)
        if s['coverage'] >= .9 and s['accuracy'] >= f15 - .02:
            cands.append((s['cost'], round(t, 2)))
    thr = min(cands)[1] if cands else .5
    return dict(models=models, thr=thr, te=te)


def asym_adaptive(probs, t_ent, t_sep, stages=B):
    n = len(probs[0]); pred = np.full(n, -1); cost = np.full(n, 15)
    for b, p in zip(stages, probs):
        und = pred < 0
        e = und & (p >= t_ent); s = und & (1 - p >= t_sep)
        pred[e] = 1; pred[s] = 0; cost[e | s] = b
    return pred, cost


def evaluate(M, rho, noise=.1, shots=500, seed=9):
    x, n, _ = ex.physical(rho, noise, shots, np.random.default_rng(seed))
    y = n > 1e-9
    pp = [ex.probability(m, x[:, :b]) for b, m in zip(B, M['models'])]
    return x, y, pp


def S(pred, y, cost, sett=None):
    m = ex.metrics(pred, y, cost)
    o = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k != 'wrong'}
    if sett is not None: o['settings'] = round(float(np.mean(sett)), 2)
    return o


def main():
    out = {}
    R = np.random.default_rng(123)
    sep_ood = separable_states(600, R)
    prod00 = np.diag([1., 0, 0, 0])[None]
    for name, aug in [('original_training_mix', False), ('augmented_with_product_states', True)]:
        M = pipeline(aug)
        x, y, pp = evaluate(M, M['te'])
        pr, co = ex.adaptive(pp, M['thr'])
        xs, ys, pps = evaluate(M, sep_ood)
        prs, cos = ex.adaptive(pps, M['thr'])
        p00 = [float(ex.probability(m, (ex.physical(prod00, 0, None, R)[0])[:, :b])[0]) for b, m in zip(B, M['models'])]
        out[name] = dict(threshold=M['thr'], in_distribution=S(pr, y, co),
                         ood_product_states_FPR=round(float((prs == 1).mean()), 4),
                         P_entangled_for_00_by_budget=[round(p, 3) for p in p00])
        if not aug:
            base = M
    # hybrid: neural early stop at 3/6/9, otherwise tomography + PPT on all 15 (9 settings)
    M = base
    hy = {}
    for noise, shots in [(0, None), (.1, 500), (.1, 5000)]:
        x, y, pp = evaluate(M, M['te'], noise, shots)
        pr, co = ex.adaptive(pp[:3], M['thr'])
        lam = an.ppt_from_tomography(x)
        rest = pr < 0
        pr[rest] = (lam[rest] < 0).astype(int); co[rest] = 15
        sett = np.array([an.SETTINGS[c] for c in co])
        npr, nco = ex.adaptive(pp, M['thr'])
        hy[f"{noise:.0%}/{shots or 'ideal'}"] = dict(neural_adaptive=S(npr, y, nco, [an.SETTINGS[c] for c in nco]),
                                                    hybrid=S(pr, y, co, sett))
    out['hybrid_neural_then_tomography'] = hy
    # asymmetric thresholds: stricter bar for declaring 'entangled'
    x, y, pp = evaluate(M, M['te'])
    out['asymmetric_thresholds_10_500'] = {}
    for te in (.69, .8, .9, .95):
        pr, co = asym_adaptive(pp, te, M['thr'])
        out['asymmetric_thresholds_10_500'][f"ent>={te}, sep>={M['thr']}"] = S(pr, y, co)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
