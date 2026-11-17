"""Additional regression coverage for review fixes (no weakened review tests)."""
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(os.environ.get('REPO',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT))
import experiment as ex
import report
from test_project import JS_HARNESS


def data():
    return json.loads(re.search(r'const data = (\{.*?\});\n',
        (ROOT/'Simulation.html').read_text(encoding='utf-8'),re.S).group(1))


def test_augmented_families_are_physical_and_separable():
    rho = ex.separable_states(100,np.random.default_rng(2028))
    assert rho.shape == (300,4,4)
    assert np.allclose(np.trace(rho,axis1=1,axis2=2),1)
    assert np.linalg.eigvalsh(rho).min() > -1e-12
    _,neg,_ = ex.physical(rho,0,None,np.random.default_rng(4))
    assert neg.max() < 1e-12


def test_legacy_linear_model_remains_loadable():
    model = dict(w=None,b=None,v=[1.,-1.,.5],c=.2)
    x = np.array([[.1,.2,.3]])
    assert np.allclose(ex.probability(model,x),ex.sigmoid(x@np.array(model['v'])+.2))


def test_empty_subset_is_undefined_not_perfect():
    assert all(v is None for v in ex.metrics(np.array([],int),np.array([],bool),np.array([])).values())


def test_report_and_html_handle_empty_near_subset(tmp_path):
    d = copy.deepcopy(data())
    for scenario in d['results'].values():
        scenario['near_n'] = 0
        scenario['near'] = ex.metrics(np.array([],int),np.array([],bool),np.array([]))
    report.write_report(tmp_path,d)
    html = (ROOT/'simulation-template.html').read_text(encoding='utf-8')
    (tmp_path/'demo.html').write_text(html.replace('__EXPERIMENT__',json.dumps(d,allow_nan=False)),encoding='utf-8')
    (tmp_path/'h.js').write_text(JS_HARNESS,encoding='utf-8')
    run = subprocess.run(['node',str(tmp_path/'h.js'),str(tmp_path/'demo.html'),'[]'],capture_output=True,text=True,check=True)
    assert json.loads(run.stdout) == []
    assert 'nan' not in (tmp_path/'Short Report.md').read_text(encoding='utf-8').lower()


def test_all_pauli_eigenbasis_products_do_not_stop_as_entangled():
    d = data()
    local = [(np.eye(2)+sign*ex.PAULI[axis])/2 for axis in 'XYZ' for sign in (-1,1)]
    rho = np.array([np.kron(a,b) for a in local for b in local])
    x,_,_ = ex.physical(rho,0,None,np.random.default_rng(4))
    probabilities = [ex.probability(m,x[:,:budget]) for m,budget in zip(d['models'],d['budgets'])]
    pred,_ = ex.adaptive(probabilities,d['threshold'])
    assert not np.any(pred == 1)


def test_product_demo_full_trace_has_no_false_entangled_decision(tmp_path):
    d = data()
    i = next(i for i,s in enumerate(d['demo']) if s['label']=='Product state |00>')
    cases = [dict(state=i,noise=n,shots=s) for n in (0,.1,.2) for s in (0,500)]
    harness = tmp_path/'h.js'
    harness.write_text(JS_HARNESS,encoding='utf-8')
    result = subprocess.run(['node',str(harness),str(ROOT/'Simulation.html'),json.dumps(cases)],
        capture_output=True,text=True,check=True)
    for output in json.loads(result.stdout):
        assert not output['status'].startswith('Entangled')
