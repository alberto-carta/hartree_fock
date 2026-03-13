"""
Test: does targeting work with the TRUE PM bath (no spin_kick)?
This is the critical missing experiment from the previous diagnostics.
"""
import os, sys, warnings
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats import ortho_group as _ortho
from triqs_hartree_fock import ImpuritySolver
from triqs_hartree_fock.incoherent_ensemble_solver import (
    _make_target_density_matrix, _targeting_step,
    _impurity_free_energy, _impurity_free_energy_components
)
from dmft_driver import make_h_int_kanamori_simple, dmft_loop_bethe_hf

norb=3; beta=40.0; w_max=20.0; eps_dlr=1e-9; U=8.0; J=1.0; t=1.0; n_target=2.0
gf_struct = [('up',norb),('down',norb)]
h_int = make_h_int_kanamori_simple(U, U-2*J, J, 0, norb=norb)

# ── 1. PM bath (NO spin_kick) ──────────────────────────────────────────────
pm = ImpuritySolver(gf_struct=gf_struct, beta=beta, w_max=w_max, eps=eps_dlr, dc='cFLL')
r = dmft_loop_bethe_hf(pm, t=t, h_int=h_int, mu_init=U, n_target=n_target,
    max_iter=100, mix=0.1, eps=1e-10, verbose=False, adjust_mu=True,
    mu_bracket=75.0, with_fock=True, one_shot=True, method='hybr', tol=1e-14)
G0_pm = pm.G0_iw
F_pm, F_pm_int, F_pm_kin = _impurity_free_energy_components(pm)
print(f"PM bath: mu={r['mu']:.4f}  F={F_pm:.4f}  (E_int={F_pm_int:.4f}  E_kin={F_pm_kin:.4f})")
print(f"n_up={np.round(np.diag(r['density']['up'].real),4)}")
print(f"n_dn={np.round(np.diag(r['density']['down'].real),4)}")

# ── 2. Targeting with PM bath ────────────────────────────────────────────
def run_t(rho_t, alpha, n_steps, label):
    sigma = {'up': np.zeros((norb,norb)), 'down': np.zeros((norb,norb))}
    for _ in range(n_steps):
        sigma, rho_now = _targeting_step(G0_pm, sigma, rho_t, alpha, True)
    eu = np.linalg.norm(rho_now['up']  - rho_t['up'])
    ed = np.linalg.norm(rho_now['down'] - rho_t['down'])
    # HF solve
    s = ImpuritySolver(gf_struct=gf_struct, beta=beta, w_max=w_max, eps=eps_dlr,
                       dc='cFLL', force_real=True)
    for bl in ['up','down']: s.G0_iw[bl].data[:] = G0_pm[bl].data
    s.dc_fixed_value = 0.0
    s.Sigma_HF['up'] = sigma['up'].copy(); s.Sigma_HF['down'] = sigma['down'].copy()
    s.solve(h_int, with_fock=True, one_shot=False, method='hybr', tol=1e-8)
    rho_up = np.diag(s.G_iw['up'].density().real)
    rho_dn = np.diag(s.G_iw['down'].density().real)
    m = rho_up.sum()-rho_dn.sum()
    F, F_int, F_kin = _impurity_free_energy_components(s)
    print(f"\n  {label}")
    print(f"  targeting ‖err‖ up={eu:.4f}  dn={ed:.4f}")
    print(f"  reached n_up={np.round(np.diag(rho_now['up']),3)}  "
          f"n_dn={np.round(np.diag(rho_now['down']),3)}")
    print(f"  Σ_up={np.round(np.diag(sigma['up']),3)}  "
          f"Σ_dn={np.round(np.diag(sigma['down']),3)}")
    F, F_int, F_kin = _impurity_free_energy_components(s)
    print(f"  → HF conv={s.hf_converged}  m={m:.3f}  F={F:.4f}  "
          f"E_int={F_int:.4f}  E_kin={F_kin:.4f}  ΔF={F-F_pm:+.4f}")
    return sigma, m

rho_diag = {'up': np.diag([1.,1.,0.]), 'down': np.zeros((norb,norb))}
R = _ortho.rvs(norb, random_state=7)
rho_rot  = {'up': R@np.diag([1.,1.,0.])@R.T, 'down': np.zeros((norb,norb))}

print("\n" + "="*60)
print("  All targeting tests use the PM bath (no spin_kick)")
print("="*60)
run_t(rho_diag, 0.5,  30, "diag [1,1,0], alpha=0.5,  30 steps  ← current default")
run_t(rho_diag, 2.0,  30, "diag [1,1,0], alpha=2.0,  30 steps")
run_t(rho_diag, 2.0, 100, "diag [1,1,0], alpha=2.0, 100 steps")
run_t(rho_rot,  0.5,  30, "SO(3) rotated, alpha=0.5,  30 steps")
run_t(rho_rot,  2.0,  30, "SO(3) rotated, alpha=2.0,  30 steps")

# ── 3. Sweep: fraction of random proposals finding m>1.5 ────────────────
print("\n" + "="*60)
print("  Sweep: random proposals, PM bath, count m>1.5 out of 50")
print("="*60)

def sweep(alpha, steps, n_trials=50, seed=999):
    rng = np.random.default_rng(seed)
    n_found = 0; n_conv = 0
    for _ in range(n_trials):
        nu = int(rng.integers(0, 3))   # 0,1,2
        nd = 2 - nu
        rho_t = {
            'up':   _make_target_density_matrix(nu, norb, rng, 0.0),
            'down': _make_target_density_matrix(nd, norb, rng, 0.0),
        }
        sigma = {'up': np.zeros((norb,norb)), 'down': np.zeros((norb,norb))}
        for _ in range(steps):
            sigma, _ = _targeting_step(G0_pm, sigma, rho_t, alpha, True)
        s = ImpuritySolver(gf_struct=gf_struct, beta=beta, w_max=w_max,
                           eps=eps_dlr, dc='cFLL', force_real=True)
        for bl in ['up','down']: s.G0_iw[bl].data[:] = G0_pm[bl].data
        s.dc_fixed_value = 0.0
        s.Sigma_HF['up'] = sigma['up']; s.Sigma_HF['down'] = sigma['down']
        s.solve(h_int, with_fock=True, one_shot=False, method='hybr', tol=1e-7)
        if s.hf_converged:
            n_conv += 1
            m_val = (np.diag(s.G_iw['up'].density().real).sum()
                   - np.diag(s.G_iw['down'].density().real).sum())
            if m_val > 1.5: n_found += 1
    return n_found, n_conv

for a, s in [(0.5,30),(2.0,30),(2.0,100)]:
    nf, nc = sweep(a, s)
    print(f"  alpha={a}, n_steps={s:3d}: m>1.5 in {nf}/50 proposals  "
          f"(converged: {nc}/50)")

print("\nDone.")
