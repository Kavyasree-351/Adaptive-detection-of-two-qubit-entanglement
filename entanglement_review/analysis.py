"""Extra experiments on top of the repo's pipeline. Usage: python analysis.py REPO_DIR"""
import sys, json
import numpy as np

sys.path.insert(0, sys.argv[1])
import experiment as ex
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

B = ex.BUDGETS
SETTINGS = {3: 3, 6: 6, 9: 9, 12: 9, 15: 9}   # local product settings needed (XI etc. come free)


def train(seed):
    ex.RNG = np.random.default_rng(seed)
    R = ex.RNG
    tr, va, te = (ex.states(n) for n in (2400, 800, 1200))
    x, neg, _ = ex.physical(tr, R.choice(ex.NOISE, len(tr)), R.choice(ex.SHOTS, len(tr)), R)
    y = neg > 1e-9
    models = [ex.fit(x[:, :b], y, 32, 40 + b) for b in B]
    reg = [ex.regressor(x[:, :b], neg) for b in B]
    xv, nv, _ = ex.physical(va, R.choice(ex.NOISE, len(va)), R.choice(ex.SHOTS, len(va)), R)
    vy, vp, vfixed, _ = ex.assess(models, reg, xv, nv)
    cands = []
    for t in np.arange(.51, .971, .01):
        pred, cost = ex.adaptive(vp, t)
        s = ex.metrics(pred, vy, cost)
        if s['coverage'] >= .9 and s['accuracy'] >= vfixed[-1]['accuracy'] - .02:
            cands.append((s['cost'], round(t, 2)))
    thr = min(cands)[1] if cands else .5
    return dict(tr=tr, te=te, x=x, y=y, models=models, reg=reg, thr=thr, R=R)


def neural_eval(T, xx, nn):
    yy = nn > 1e-9
    pp = [ex.probability(m, xx[:, :b]) for b, m in zip(B, T['models'])]
    pred, cost = ex.adaptive(pp, T['thr'])
    return yy, pp, pred, cost


# ---------------- rigorous (zero-false-positive in the ideal limit) detectors
def corr_matrix(x):          # 3x3 correlation matrix T_ij = <sigma_i sigma_j>
    idx = {n: k for k, n in enumerate(ex.NAMES)}
    return x[:, [idx[a + b] for a in "XYZ" for b in "XYZ"]].reshape(-1, 3, 3)


def witness_scores(x):
    """Returns per-stage statistic s with 'entangled' certified when s > 1.
    3 obs: |T_xx|+|T_yy|+|T_zz| <= ||T||_1 <= 1 for separable states (de Vicente).
    9 obs: trace norm of the full correlation matrix (de Vicente criterion)."""
    s3 = np.abs(x[:, :3]).sum(1)
    s9 = np.linalg.svd(corr_matrix(x), compute_uv=False).sum(1)
    return s3, s9


def ppt_from_tomography(x):
    rho = ex.IDENTITY + np.einsum('nk,kij->nij', x, ex.OPS) / 4
    return np.linalg.eigvalsh(rho.reshape(-1, 2, 2, 2, 2).transpose(0, 1, 4, 3, 2).reshape(-1, 4, 4))[:, 0]


def rigorous_adaptive(x, shots, margin_sigma=3.0):
    """Stop at 3 or 9 observables if a witness certifies entanglement with a
    shot-noise margin; otherwise do full tomography and apply PPT with margin.
    Separable is only declared after full tomography; otherwise 'abstain' if
    the PPT eigenvalue is within the noise margin."""
    n = len(x)
    s3, s9 = witness_scores(x)
    sig = 0 if shots is None else 1 / np.sqrt(shots)          # per-observable std upper bound
    pred = np.full(n, -1); cost = np.full(n, 15); sett = np.full(n, 9)
    a = s3 > 1 + margin_sigma * sig * np.sqrt(3)
    pred[a] = 1; cost[a] = 3; sett[a] = 3
    b = (pred < 0) & (s9 > 1 + margin_sigma * sig * 3)
    pred[b] = 1; cost[b] = 9; sett[b] = 9
    lam = ppt_from_tomography(x)
    m = margin_sigma * sig * np.sqrt(15) / 4                   # crude eigenvalue error bound
    rest = pred < 0
    pred[rest & (lam < -m)] = 1
    pred[rest & (lam > m)] = 0
    return pred, cost, sett


def summary(pred, y, cost, sett=None):
    m = ex.metrics(pred, y, cost)
    out = {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items() if k != 'wrong'}
    if sett is not None:
        out['settings'] = round(float(np.mean(sett)), 2)
    return out


def ece(p, y, bins=10):
    conf = np.maximum(p, 1 - p); hat = p >= .5
    e = 0
    for lo in np.linspace(.5, 1, bins, endpoint=False):
        s = (conf >= lo) & (conf < lo + .5 / bins + (lo + .5 / bins >= 1))
        if s.any():
            e += s.mean() * abs((hat[s] == y[s]).mean() - conf[s].mean())
    return e


def main():
    out = {}
    T = train(42)
    R = np.random.default_rng(7)
    xx, nn, margin = ex.physical(T['te'], .10, 500, R)
    yy, pp, pred, cost = neural_eval(T, xx, nn)
    sett = np.array([SETTINGS[c] for c in cost])
    out['threshold'] = T['thr']
    out['neural_adaptive_10_500'] = summary(pred, yy, cost, sett)
    out['neural_fixed15_10_500'] = summary((pp[-1] >= .5).astype(int), yy, np.full(len(yy), 15))

    # per family (test = 1200 pure, 1200 mixed, 1200 Werner)
    fam = {}
    for name, sl in [('pure', slice(0, 1200)), ('mixed', slice(1200, 2400)), ('werner', slice(2400, 3600))]:
        fam[name] = dict(entangled_frac=round(float(yy[sl].mean()), 3), **summary(pred[sl], yy[sl], cost[sl]))
    out['per_family_10_500'] = fam

    # rigorous baselines across scenarios
    rig = {}
    for noise, shots in [(0, None), (0, 500), (.1, 500), (.1, 100), (.1, 5000), (.2, 500)]:
        x2, n2, _ = ex.physical(T['te'], noise, shots, np.random.default_rng(11))
        y2 = n2 > 1e-9
        _, pp2, pr2, c2 = neural_eval(T, x2, n2)
        rp, rc, rs = rigorous_adaptive(x2, shots)
        lam = ppt_from_tomography(x2)
        name = f"{noise:.0%}/{shots or 'ideal'}"
        rig[name] = dict(
            neural_adaptive=summary(pr2, y2, c2, np.array([SETTINGS[c] for c in c2])),
            tomography_PPT_no_margin=summary((lam < 0).astype(int), y2, np.full(len(y2), 15), np.full(len(y2), 9)),
            rigorous_adaptive=summary(rp, y2, rc, rs),
            witness3_alone=summary((witness_scores(x2)[0] > 1).astype(int), y2, np.full(len(y2), 3)))
    out['rigorous_vs_neural'] = rig

    # fair logistic regression + a properly trained MLP (same quadratic features)
    xt, yt = T['x'], T['y']
    fair = {}
    for b in B:
        f_tr, f_te = ex.features(xt[:, :b])[:, 1:], ex.features(xx[:, :b])[:, 1:]
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000)).fit(f_tr, yt)
        mlp = make_pipeline(StandardScaler(), MLPClassifier((64, 64), alpha=1e-3, max_iter=400,
                            early_stopping=True, random_state=0)).fit(f_tr, yt)
        repo_lin = ex.probability(ex.fit(xt[:, :b], yt, 0, 60 + b), xx[:, :b]) >= .5
        fair[b] = dict(repo_logistic_raw=summary(repo_lin.astype(int), yy, np.full(len(yy), b)),
                       logistic_quadratic=summary(lr.predict(f_te).astype(int), yy, np.full(len(yy), b)),
                       mlp_64x64=summary(mlp.predict(f_te).astype(int), yy, np.full(len(yy), b)),
                       repo_neural=summary((pp[B.index(b)] >= .5).astype(int), yy, np.full(len(yy), b)))
    out['fair_baselines_10_500'] = fair

    # calibration
    out['ECE_per_budget_10_500'] = [round(ece(p, yy), 4) for p in pp]

    # out of distribution
    ood = {}
    n = 1200
    u = lambda: np.linalg.qr(ex.complex_normal((n, 2, 2)))[0]
    UA, UB = u(), u()
    rot = np.array([np.kron(a, b) for a, b in zip(UA, UB)])
    p = R.uniform(0, 1, n)
    wer2 = rot @ (p[:, None, None] * ex.BELL + (1 - p[:, None, None]) * ex.IDENTITY) @ rot.conj().swapaxes(1, 2)
    psi = ex.complex_normal((n, 2)); phi = ex.complex_normal((n, 2))
    psi /= np.linalg.norm(psi, axis=1, keepdims=True); phi /= np.linalg.norm(phi, axis=1, keepdims=True)
    prod = np.einsum('ni,nj->nij', psi, phi).reshape(n, 4)
    prod = np.einsum('ni,nj->nij', prod, prod.conj())
    a, b2 = ex.complex_normal((n, 2, 2)), ex.complex_normal((n, 2, 2))
    ra = a @ a.conj().swapaxes(1, 2); ra /= np.trace(ra, axis1=1, axis2=2)[:, None, None]
    rb = b2 @ b2.conj().swapaxes(1, 2); rb /= np.trace(rb, axis1=1, axis2=2)[:, None, None]
    prodmix = np.einsum('nij,nkl->nikjl', ra, rb).reshape(n, 4, 4)
    classical = np.zeros((n, 4, 4)); classical[:, range(4), range(4)] = R.dirichlet(np.ones(4), n)
    for name, rho in [('Werner, both qubits rotated', wer2), ('random pure product states', prod),
                      ('random mixed product states', prodmix), ('classical (diagonal) states', classical)]:
        x3, n3, _ = ex.physical(rho, .1, 500, np.random.default_rng(5))
        y3, pp3, pr3, c3 = neural_eval(T, x3, n3)
        rp, rc, rs = rigorous_adaptive(x3, 500)
        ood[name] = dict(entangled_frac=round(float(y3.mean()), 3), neural_adaptive=summary(pr3, y3, c3),
                         rigorous_adaptive=summary(rp, y3, rc, rs))
    out['out_of_distribution_10_500'] = ood

    # seed variance
    seeds = {}
    for s in (1, 2, 3, 4, 5):
        T2 = train(s)
        x4, n4, _ = ex.physical(T2['te'], .1, 500, T2['R'])
        y4, pp4, pr4, c4 = neural_eval(T2, x4, n4)
        seeds[s] = dict(threshold=T2['thr'], adaptive=summary(pr4, y4, c4),
                        fixed15_acc=round(float(((pp4[-1] >= .5) == y4).mean()), 4))
    out['seed_variance_10_500'] = seeds
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
