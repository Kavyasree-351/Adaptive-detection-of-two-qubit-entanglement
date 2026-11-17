"""Reproduce the two-qubit experiment. Run with Python 3 and NumPy."""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(42)
BUDGETS = (3, 6, 9, 12, 15)
NAMES = "XX YY ZZ XY XZ YX YZ ZX ZY XI YI ZI IX IY IZ".split()
PAULI = dict(I=np.eye(2), X=np.array([[0,1],[1,0]]),
             Y=np.array([[0,-1j],[1j,0]]), Z=np.diag([1,-1]))
OPS = np.array([np.kron(PAULI[a], PAULI[b]) for a,b in NAMES])
IDENTITY = np.eye(4)/4
BELL = np.outer([1,0,0,1], [1,0,0,1])/2
NOISE = (0,.02,.05,.10,.15,.20)
SHOTS = (100,500,1000,5000)

def complex_normal(shape):
    return RNG.normal(size=shape) + 1j*RNG.normal(size=shape)

def states(n):
    # Independent draws within each split. Exact Bell states are demo-only.
    pure = complex_normal((n,4,1))
    mixed = complex_normal((n,4,4))
    both = np.concatenate([pure @ pure.conj().swapaxes(1,2),
                           mixed @ mixed.conj().swapaxes(1,2)])
    both /= np.trace(both, axis1=1, axis2=2)[:,None,None]
    # Locally rotate the Bell component to avoid a single privileged orientation.
    u, _ = np.linalg.qr(complex_normal((n,2,2)))
    rotation = np.array([np.kron(np.eye(2), item) for item in u])
    bell = rotation @ BELL @ rotation.conj().swapaxes(1,2)
    p = np.concatenate([RNG.uniform(0,1,n//2), RNG.uniform(.22,.46,n-n//2)])
    return np.concatenate([both, p[:,None,None]*bell + (1-p[:,None,None])*IDENTITY])

def separable_states(n, rng):
    """Independent pure/mixed products and Pauli-axis product preparations.

    Include all local Pauli eigenbasis families, not a special-case demo label.
    The original pure/mixed/Werner generator remains unchanged for comparisons.
    """
    a = rng.normal(size=(n,2)) + 1j*rng.normal(size=(n,2))
    b = rng.normal(size=(n,2)) + 1j*rng.normal(size=(n,2))
    a /= np.linalg.norm(a,axis=1,keepdims=True)
    b /= np.linalg.norm(b,axis=1,keepdims=True)
    psi = np.einsum('ni,nj->nij',a,b).reshape(n,4)
    pure = psi[:,:,None]*psi[:,None,:].conj()
    local = []
    for _ in range(2):
        g = rng.normal(size=(n,2,2)) + 1j*rng.normal(size=(n,2,2))
        r = g @ g.conj().swapaxes(1,2)
        local.append(r/np.trace(r,axis1=1,axis2=2)[:,None,None])
    mixed = np.einsum('nij,nkl->nikjl',*local).reshape(n,4,4)
    # Random local polarization includes pure endpoints and interior mixed states.
    axis = np.array([PAULI[k] for k in 'XYZ'])
    shared_axis = rng.integers(3,size=n//2)
    local = []
    for _ in range(2):
        length = rng.uniform(0,1,n)
        length[:n//2] = 1
        direction = rng.choice([-1,1],n)*length
        choices = rng.integers(3,size=n)
        # Cover the extrema of each measured diagonal correlation, including
        # both signs and all three axes; noisy random products rarely reach them.
        choices[:n//2] = shared_axis
        local.append((PAULI['I']+direction[:,None,None]*axis[choices])/2)
    aligned = np.einsum('nij,nkl->nikjl',*local).reshape(n,4,4)
    return np.concatenate([pure,mixed,aligned])

def physical(rho, noise, shots, rng):
    noise = np.broadcast_to(noise, len(rho))
    rho = (1-noise[:,None,None])*rho + noise[:,None,None]*IDENTITY
    eigen = np.linalg.eigvalsh(rho.reshape(-1,2,2,2,2).transpose(0,1,4,3,2).reshape(-1,4,4))
    neg = -np.minimum(eigen,0).sum(axis=1)
    means = np.einsum('nij,kji->nk',rho,OPS).real.clip(-1,1)
    if shots is None:
        return means, neg, eigen[:,0]
    shots = np.broadcast_to(shots, len(rho))[:,None]
    observed = 2*rng.binomial(shots,(1+means)/2)/shots-1
    return observed, neg, eigen[:,0]

def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-40,40)))

def fit(x, y, hidden, seed):
    rng = np.random.default_rng(seed)
    # Both baselines see the same nonlinear feature map. Linear raw Pauli
    # means cannot capture the sign-symmetric correlations in this mixture.
    x = features(x)[:,1:]
    if hidden:
        w = rng.normal(0,.5,(x.shape[1],hidden))
        b = np.zeros(hidden)
        v = rng.normal(0,.2,hidden)
    else:
        v = np.zeros(x.shape[1])
    c = 0.
    for _ in range(1600):
        idx = rng.integers(len(x),size=512)
        batch = x[idx]
        h = np.tanh(batch @ w + b) if hidden else batch
        err = sigmoid(h @ v+c)-y[idx]
        if hidden:
            delta = err[:,None]*v*(1-h*h)
            w -= .24*(batch.T @ delta/len(idx)+.0002*w)
            b -= .24*delta.mean(axis=0)
        v -= .24*(h.T @ err/len(idx)+.0002*v)
        c -= .24*err.mean()
    return dict(v=v.tolist(), c=float(c), w=w.tolist() if hidden else None,
                b=b.tolist() if hidden else None, feature_map='quadratic')

def probability(model, x):
    if model.get('feature_map') == 'quadratic' or model['w'] is not None:
        x = features(x)[:,1:]
    if model['w'] is not None:
        x = np.tanh(x @ np.array(model['w']) + model['b'])
    return sigmoid(x @ np.array(model['v']) + model['c'])

def forest(x,y,seed):
    rng = np.random.default_rng(seed)
    def grow(a,b,depth):
        avg = float(b.mean())
        if depth == 0 or len(b)<24 or avg in (0,1):
            return avg
        best = None
        for j in rng.choice(a.shape[1],max(1,int(a.shape[1]**.5)),replace=False):
            for t in np.quantile(a[:,j],[.2,.4,.6,.8]):
                left = a[:,j]<=t
                nl = int(left.sum())
                nr = len(b)-nl
                if min(nl,nr)<8:
                    continue
                pl, pr = b[left].mean(), b[~left].mean()
                score = nl*pl*(1-pl)+nr*pr*(1-pr)
                if best is None or score<best[0]:
                    best = (score,int(j),float(t),left)
        if best is None:
            return avg
        _,j,t,left = best
        return [j,t,grow(a[left],b[left],depth-1),grow(a[~left],b[~left],depth-1)]
    trees = []
    for _ in range(24):
        ix = rng.integers(len(x),size=min(3000,len(x)))
        trees.append(grow(x[ix],y[ix],8))
    return trees

def forest_probability(trees,x):
    def predict(tree, rows):
        if isinstance(tree,float):
            return np.full(len(rows),tree)
        j,t,left,right = tree
        mask = rows[:,j]<=t
        out = np.empty(len(rows))
        out[mask] = predict(left,rows[mask])
        out[~mask] = predict(right,rows[~mask])
        return out
    return np.mean([predict(tree,x) for tree in trees],axis=0)

def features(x):
    return np.column_stack([np.ones(len(x)),x,x*x,
                            *[x[:,i]*x[:,j] for i in range(x.shape[1]) for j in range(i)]])

def regressor(x,y):
    f = features(x)
    return np.linalg.solve(f.T @ f + np.eye(f.shape[1])*.5,f.T @ y)

def metrics(pred,y,cost):
    pred,y = np.asarray(pred),np.asarray(y,dtype=bool)
    if pred.shape != y.shape or pred.ndim != 1 or not np.isin(pred,[-1,0,1]).all():
        raise ValueError('Predictions and labels must be matching vectors; predictions are -1/0/1.')
    costs=np.asarray(cost)
    if costs.ndim > 1 or (costs.ndim == 1 and costs.shape != y.shape) or not np.isfinite(costs).all() or np.any(costs<0):
        raise ValueError('Costs must be finite nonnegative scalars or matching vectors.')
    if not len(y):
        return dict(cost=None,coverage=None,accuracy=None,fpr=None,wrong=None)
    answered = pred>=0
    return dict(cost=float(np.mean(cost)), coverage=float(answered.mean()),
                accuracy=float((pred[answered]==y[answered]).mean()) if answered.any() else None,
                fpr=float((pred[~y]==1).mean()) if (~y).any() else None,
                wrong=float(((pred!=y)&answered).mean()))

def adaptive(probs, threshold):
    if not .5 <= threshold <= 1 or not 1 <= len(probs) <= len(BUDGETS):
        raise ValueError('Invalid threshold or stage count.')
    n = len(probs[0])
    if any(np.asarray(p).shape != (n,) or not np.isfinite(p).all() or np.any((p<0)|(p>1)) for p in probs):
        raise ValueError('Every stage needs matching finite probability vectors in [0,1].')
    pred = np.full(n,-1)
    cost = np.full(n,BUDGETS[-1])
    for b,p in zip(BUDGETS,probs):
        accept = (pred<0)&(np.maximum(p,1-p)>=threshold)
        pred[accept] = p[accept]>=.5
        cost[accept] = b
    return pred,cost

def assess(models,reg,x,neg):
    y = neg>1e-9
    prob = [probability(m,x[:,:b]) for b,m in zip(BUDGETS,models)]
    fixed = [metrics(p>=.5,y,b) for b,p in zip(BUDGETS,prob)]
    mae = [float(np.mean(abs(np.clip(features(x[:,:b])@r,0,.5)-neg)))
           for b,r in zip(BUDGETS,reg)]
    return y,prob,fixed,mae

def main():
    global RNG
    RNG = np.random.default_rng(42)
    train,val,test = (states(n) for n in (2400,800,1200))
    train = np.concatenate([train,separable_states(1200,RNG)])
    val = np.concatenate([val,separable_states(400,RNG)])
    x,neg,_ = physical(train,RNG.choice(NOISE,len(train)),RNG.choice(SHOTS,len(train)),RNG)
    y = neg>1e-9
    models = [fit(x[:,:b],y,32,40+b) for b in BUDGETS]
    logistic = [fit(x[:,:b],y,0,60+b) for b in BUDGETS]
    forests = [forest(x[:,:b],y,80+b) for b in BUDGETS]
    reg = [regressor(x[:,:b],neg) for b in BUDGETS]
    xv,nv,_ = physical(val,RNG.choice(NOISE,len(val)),RNG.choice(SHOTS,len(val)),RNG)
    vy,vp,vfixed,_ = assess(models,reg,xv,nv)
    candidates = []
    for t in np.arange(.51,.971,.01):
        pred,cost = adaptive(vp,t)
        score = metrics(pred,vy,cost)
        if score['coverage']>=.90 and score['accuracy']>=vfixed[-1]['accuracy']-.02:
            candidates.append((score['cost'],float(round(t,2))))
    threshold = min(candidates)[1] if candidates else .5
    scenarios = [('Ideal',0,None)] + [(f'{noise:.0%} / {shots:,}',noise,shots)
                                     for noise in NOISE for shots in SHOTS]
    results = {}
    for name,noise,shots in scenarios:
        xx,nn,margin = physical(test,noise,shots,RNG)
        yy,pp,fixed,mae = assess(models,reg,xx,nn)
        pred,cost = adaptive(pp,threshold)
        answered = pred>=0
        estimates = np.empty(len(nn))
        for budget, model in zip(BUDGETS, reg):
            stop = cost == budget
            estimates[stop] = np.clip(features(xx[stop,:budget]) @ model, 0, .5)
        adaptive_mae = float(np.mean(abs(estimates[answered]-nn[answered]))) if answered.any() else None
        stop_counts = [int(((cost == b) & answered).sum()) for b in BUDGETS]
        near = abs(margin)<=.025
        results[name] = dict(fixed=fixed, mae=mae, adaptive=metrics(pred,yy,cost),
                             near=metrics(pred[near],yy[near],cost[near]), near_n=int(near.sum()), adaptive_mae=adaptive_mae,
                             stop_counts=stop_counts, abstained=int((~answered).sum()))
        if name=='10% / 500':
            results[name]['baselines'] = {
                'Quadratic logistic regression':[metrics(probability(m,xx[:,:b])>=.5,yy,b)
                                       for b,m in zip(BUDGETS,logistic)],
                'Random forest':[metrics(forest_probability(m,xx[:,:b])>=.5,yy,b)
                                 for b,m in zip(BUDGETS,forests)]}
    demo = [('Bell state',BELL),('Product state |00>',np.diag([1,0,0,0])),
            ('Werner mixture, p = 0.36',.36*BELL+.64*IDENTITY),
            ('Werner mixture, p = 0.25',.25*BELL+.75*IDENTITY),
            ('Held-out random pure state',test[12]),('Held-out random mixed state',test[1212])]
    # Adding white noise shifts each partial-transpose eigenvalue affinely.
    demo_rows = []
    for label,rho in demo:
        base,_,min_eig = physical(rho[None],0,None,RNG)
        eig = np.linalg.eigvalsh(rho.reshape(2,2,2,2).transpose(0,3,2,1).reshape(4,4))
        demo_rows.append(dict(label=label, means=base[0].tolist(), eig=eig.tolist()))
    # Separate held-out product-family stress test; never used to select threshold.
    stress = separable_states(600,np.random.default_rng(2026))
    sx,sn,_ = physical(stress,.1,500,np.random.default_rng(2027))
    sp = [probability(m,sx[:,:b]) for b,m in zip(BUDGETS,models)]
    spr,sco = adaptive(sp,threshold)
    stress_metrics = metrics(spr,sn>1e-9,sco)
    data = dict(budgets=BUDGETS,names=NAMES,models=models,reg=[r.tolist() for r in reg],
                split_sizes=dict(train=len(train),validation=len(val),test=len(test)),
                product_stress=stress_metrics,product_stress_n=len(stress),
                threshold=threshold,results=results,demo=demo_rows, constrained=bool(candidates))
    template = (HERE/'simulation-template.html').read_text(encoding='utf-8')
    (HERE/'Simulation.html').write_text(template.replace('__EXPERIMENT__',json.dumps(data,allow_nan=False)),encoding='utf-8')
    (HERE/'experiment_results.json').write_text(json.dumps(
        {k:v for k,v in data.items() if k not in ('models','reg')},indent=2,allow_nan=False),encoding='utf-8')
    from report import write_report
    # The published report is rendered from the actual saved result artifact.
    write_report(HERE,json.loads((HERE/'experiment_results.json').read_text(encoding='utf-8')))

if __name__ == '__main__':
    main()
