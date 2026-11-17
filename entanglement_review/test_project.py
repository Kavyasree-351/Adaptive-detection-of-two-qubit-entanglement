"""Unit and integration tests for the adaptive entanglement-detection project.

Run:  REPO=/path/to/repo python -m pytest -q test_project.py
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(os.environ.get("REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
import experiment as ex  # noqa: E402
import report  # noqa: E402

RNG = np.random.default_rng(0)


def pt_min_eig(rho):
    return np.linalg.eigvalsh(rho.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4))[0]


# ---------------------------------------------------------------- physics
def test_pauli_basis_is_orthogonal_hermitian_traceless():
    for P in ex.OPS:
        assert np.allclose(P, P.conj().T)
        assert abs(np.trace(P)) < 1e-12
    gram = np.einsum("aij,bji->ab", ex.OPS, ex.OPS).real
    assert np.allclose(gram, 4 * np.eye(15))


def test_fifteen_expectations_reconstruct_the_state():
    rho = ex.states(20)
    means, _, _ = ex.physical(rho, 0, None, RNG)
    recon = ex.IDENTITY + np.einsum("nk,kij->nij", means, ex.OPS) / 4
    assert np.allclose(recon, rho, atol=1e-12)


def test_generated_states_are_valid_density_matrices():
    rho = ex.states(50)
    assert rho.shape == (150, 4, 4)
    assert np.allclose(np.trace(rho, axis1=1, axis2=2), 1)
    assert np.allclose(rho, rho.conj().swapaxes(1, 2))
    assert np.linalg.eigvalsh(rho).min() > -1e-12


@pytest.mark.parametrize("rho,expected", [
    (ex.BELL, 0.5),
    (np.diag([1.0, 0, 0, 0]), 0.0),
    (ex.IDENTITY, 0.0),
])
def test_negativity_of_reference_states(rho, expected):
    _, neg, _ = ex.physical(rho[None], 0, None, RNG)
    assert neg[0] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("p", [0.0, 0.2, 0.33, 0.34, 0.5, 0.9, 1.0])
def test_werner_negativity_and_one_third_threshold(p):
    rho = p * ex.BELL + (1 - p) * ex.IDENTITY
    _, neg, _ = ex.physical(rho[None], 0, None, RNG)
    assert neg[0] == pytest.approx(max(0, (3 * p - 1) / 4), abs=1e-12)
    assert (neg[0] > 1e-9) == (p > 1 / 3)


@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.7])
def test_depolarising_noise_applied_before_labelling(q):
    _, neg, _ = ex.physical(ex.BELL[None], q, None, RNG)
    assert neg[0] == pytest.approx(max(0, (3 * (1 - q) - 1) / 4), abs=1e-12)


def test_partial_transpose_matches_independent_implementation():
    rho = ex.states(30)
    _, _, min_eig = ex.physical(rho, 0, None, RNG)
    assert np.allclose(min_eig, [pt_min_eig(r) for r in rho])
    # PT is basis-independent in spectrum: transpose on A gives the same spectrum
    pta = rho.reshape(-1, 2, 2, 2, 2).transpose(0, 3, 2, 1, 4).reshape(-1, 4, 4)
    assert np.allclose(np.linalg.eigvalsh(pta)[:, 0], min_eig)


def test_shot_noise_is_unbiased_with_binomial_variance():
    rho = np.repeat((0.6 * ex.BELL + 0.4 * ex.IDENTITY)[None], 20000, axis=0)
    exact, _, _ = ex.physical(rho[:1], 0, None, RNG)
    obs, _, _ = ex.physical(rho, 0, 500, np.random.default_rng(1))
    assert np.allclose(obs.mean(0), exact[0], atol=0.004)
    assert np.allclose(obs.var(0), (1 - exact[0] ** 2) / 500, rtol=0.06)


# ---------------------------------------------------------- policy/metrics
def test_adaptive_policy_extremes():
    probs = [np.array([0.9, 0.2, 0.55])] * 5
    pred, cost = ex.adaptive(probs, 0.5)
    assert pred.tolist() == [1, 0, 1] and cost.tolist() == [3, 3, 3]
    pred, cost = ex.adaptive(probs, 0.999)
    assert pred.tolist() == [-1, -1, -1] and cost.tolist() == [15, 15, 15]


def test_adaptive_policy_escalates_until_confident():
    probs = [np.array([0.6]), np.array([0.65]), np.array([0.95]), np.array([0.5]), np.array([0.5])]
    pred, cost = ex.adaptive(probs, 0.9)
    assert pred.tolist() == [1] and cost.tolist() == [9]


def test_metrics_counts_abstentions_in_fpr_denominator():
    y = np.array([False, False, True, True])
    pred = np.array([1, -1, 1, -1])
    m = ex.metrics(pred, y, np.array([3, 15, 3, 15]))
    assert m["coverage"] == 0.5 and m["accuracy"] == 0.5 and m["fpr"] == 0.5


def test_metrics_on_empty_subset_is_json_safe():
    """Exposes a latent crash: an empty near-boundary subset produces NaN, and
    json.dumps(..., allow_nan=False) in experiment.main() would then raise."""
    m = ex.metrics(np.array([], int), np.array([], bool), np.array([]))
    json.dumps(m, allow_nan=False)


# --------------------------------------------------------------- models
def test_feature_count_is_quadratic():
    x = RNG.normal(size=(4, 15))
    assert ex.features(x).shape[1] == 1 + 15 + 15 + 105


def test_logistic_baseline_is_not_degenerate():
    """The committed report shows logistic regression predicting 'entangled' for
    every state (FPR 100%, accuracy = base rate). This test documents that."""
    train = ex.states(400)
    x, neg, _ = ex.physical(train, 0.1, 500, np.random.default_rng(3))
    y = neg > 1e-9
    m = ex.fit(x, y, 0, 0)
    p = ex.probability(m, x)
    assert 0.02 < (p >= 0.5).mean() < 0.98, "logistic regression collapsed to one class"


# ----------------------------------------------- report and HTML end-to-end
@pytest.fixture(scope="module")
def sim_data():
    html = (REPO / "Simulation.html").read_text(encoding="utf-8")
    m = re.search(r"const data = (\{.*?\});\n", html, re.S)
    return json.loads(m.group(1))


def test_report_handles_unconstrained_threshold(tmp_path, sim_data):
    d = dict(sim_data, constrained=False)
    report.write_report(tmp_path, d)
    assert "No scanned threshold" in (tmp_path / "Short Report.md").read_text()


JS_HARNESS = r"""
const fs=require('fs');
const html=fs.readFileSync(process.argv[2],'utf8');
const script=html.split('<script>')[1].split('</script>')[0];
const els={};
function mk(id){return els[id]||(els[id]={id,value:'',textContent:'',disabled:false,innerHTML:'',
 rows:[],listeners:{},options:[],
 add(o){this.options.push(o)},addEventListener(t,f){this.listeners[t]=f},
 replaceChildren(){this.rows=[]},
 insertRow(){const r=[];this.rows.push(r);return{insertCell(){const c={};r.push(c);return c}}}});}
global.document={getElementById:mk};
global.Option=function(l,v){this.label=l;this.value=v};
eval(script);
const cases=JSON.parse(process.argv[3]);const out=[];
for(const c of cases){
  mk('state').value=String(c.state);mk('noise').value=String(c.noise);
  mk('shots').value=String(c.shots);mk('seed').value='7';
  els.controls.listeners.submit({preventDefault(){}});
  while(!els.next.disabled) els.next.listeners.click();
  out.push({p:els.trace.rows.map(r=>parseFloat(r[2].textContent)/100),status:els.status.textContent});
}
console.log(JSON.stringify(out));
"""


def test_html_demo_matches_python_models(tmp_path, sim_data):
    """Drive the real Simulation.html script under Node with a DOM stub, in the
    ideal (no-shot) setting, and compare its probabilities with Python's."""
    harness = tmp_path / "h.js"
    harness.write_text(JS_HARNESS)
    cases = [dict(state=i, noise=n, shots=0) for i in range(len(sim_data["demo"])) for n in (0, 0.1)]
    res = subprocess.run(["node", str(harness), str(REPO / "Simulation.html"), json.dumps(cases)],
                         capture_output=True, text=True, check=True)
    out = json.loads(res.stdout)
    for c, o in zip(cases, out):
        means = np.array(sim_data["demo"][c["state"]]["means"]) * (1 - c["noise"])
        for k, p_js in enumerate(o["p"]):
            b = sim_data["budgets"][k]
            p_py = ex.probability(sim_data["models"][k], means[None, :b])[0]
            assert p_js == pytest.approx(p_py, abs=6e-4)  # JS prints 1 decimal of %
        eig = np.array(sim_data["demo"][c["state"]]["eig"])
        truth = abs(-np.minimum(0, (1 - c["noise"]) * eig + c["noise"] / 4).sum())
        assert f"true {truth:.4f}" in o["status"]


def test_demo_does_not_confidently_call_product_state_entangled(sim_data):
    """|00> is separable. The demo should never stop and declare it entangled."""
    i = [d["label"] for d in sim_data["demo"]].index("Product state |00>")
    means = np.array(sim_data["demo"][i]["means"])
    p3 = ex.probability(sim_data["models"][0], means[None, :3])[0]
    assert p3 < sim_data["threshold"], f"P(entangled)={p3:.3f} at 3 observables"
