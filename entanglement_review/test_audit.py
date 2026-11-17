"""Hand-counted checks, shuffled-label controls and artifact provenance."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import experiment as ex
import report

RESULTS = {}

def test_five_state_toy():
    rho = np.array([ex.BELL,np.diag([1.,0,0,0]),ex.IDENTITY,
                    .6*ex.BELL+.4*ex.IDENTITY,.2*ex.BELL+.8*ex.IDENTITY])
    x,n,_ = ex.physical(rho,0,None,np.random.default_rng(0))
    np.testing.assert_allclose(n,[.5,0,0,.2,0],atol=1e-12)
    y = n>1e-9
    pred = np.array([1,0,1,1,0])
    m = ex.metrics(pred,y,np.full(5,3))
    assert m['accuracy']==4/5
    assert m['fpr']==1/3
    assert m['coverage']==1
    RESULTS['toy'] = dict(negativity=n.tolist(),expectations=x.tolist(),metrics=m,
                          true_admissions=2,false_admissions=1,total_admissions=3)

def test_shuffled_labels():
    rows=[]
    for seed in (31,32,33):
        ex.RNG=np.random.default_rng(seed)
        tr,te=ex.states(300),ex.states(300)
        fingerprints=lambda a:{hashlib.sha256(np.round(r,12).tobytes()).hexdigest() for r in a}
        assert fingerprints(tr).isdisjoint(fingerprints(te))
        rng=np.random.default_rng(seed+100)
        x,n,_=ex.physical(tr,.1,500,rng)
        xt,nt,_=ex.physical(te,.1,500,rng)
        yt=nt>1e-9
        shuffled=rng.permutation(n>1e-9)
        for hidden in (0,32):
            model=ex.fit(x,shuffled,hidden,seed)
            p=ex.probability(model,xt)
            brier=float(np.mean((p-yt)**2))
            prior=float(np.mean((shuffled.mean()-yt)**2))
            assert brier >= prior-.03, 'Suspicious predictive skill with shuffled training labels'
            rows.append(dict(seed=seed,hidden=hidden,brier=brier,constant_prior_brier=prior))
        trees=ex.forest(x,shuffled,seed)
        p=ex.forest_probability(trees,xt)
        brier=float(np.mean((p-yt)**2))
        assert brier >= prior-.03
        rows.append(dict(seed=seed,model='forest',brier=brier,constant_prior_brier=prior))
    RESULTS['shuffled_labels']=rows

def test_artifacts_and_report_match_saved_results():
    import re
    saved=json.loads((ROOT/'experiment_results.json').read_text())
    html=json.loads(re.search(r'const data = (\{.*?\});\n',
        (ROOT/'Simulation.html').read_text(encoding='utf-8'),re.S).group(1))
    assert saved == {k:v for k,v in html.items() if k not in ('models','reg')}
    with tempfile.TemporaryDirectory() as folder:
        report.write_report(Path(folder),saved)
        assert (Path(folder)/'Short Report.md').read_text(encoding='utf-8') == (ROOT/'Short Report.md').read_text(encoding='utf-8')
    for r in saved['results'].values():
        assert sum(r['stop_counts'])+r['abstained']==saved['split_sizes']['test']
        for m in [r['adaptive'],r['near'],*r['fixed']]:
            for k in ('accuracy','coverage','fpr','wrong'):
                assert m[k] is None or 0<=m[k]<=1
    RESULTS['artifact_parity']=True

def test_randomized_count_invariants():
    rng=np.random.default_rng(5)
    for _ in range(100):
        y=rng.integers(2,size=50).astype(bool)
        pred=rng.integers(-1,2,size=50)
        m=ex.metrics(pred,y,np.full(50,3))
        tp=int(np.sum((pred==1)&y)); fp=int(np.sum((pred==1)&~y))
        assert tp+fp==int(np.sum(pred==1))
        assert m['coverage']==float(np.mean(pred>=0))
        assert m['wrong']==float(np.sum((pred>=0)&(pred!=y))/50)

if __name__=='__main__':
    test_five_state_toy()
    test_shuffled_labels()
    test_artifacts_and_report_match_saved_results()
    test_randomized_count_invariants()
    (ROOT/'audit_results.json').write_text(json.dumps(RESULTS,indent=2,allow_nan=False))
    print(json.dumps(RESULTS,indent=2))
