"""
Debug: test HF convergence from several starting points including Hund's m=2 state.
Compare free energies to understand why the magnetic state sits above PM in free energy.
"""


#%%
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from triqs.gf import BlockGf, Gf
from triqs.gf.meshes import MeshDLRImFreq
from triqs_hartree_fock import ImpuritySolver
from triqs_hartree_fock import IncoherentEnsembleSolver, DensityMatrixProposals
from dmft_driver import make_h_int_kanamori_simple, dmft_loop_bethe_hf
from triqs_hartree_fock.incoherent_ensemble_solver import Solution, _impurity_free_energy, _impurity_free_energy_components

#%%
norb      = 3
beta      = 40.0
w_max     = 20.0
eps_dlr   = 1e-9
U         = 8.0
J         = 1.0
t         = 1.0
n_target  = 2.0
gf_struct = [('up', norb), ('down', norb)]


# give a kick  [1, 1, 0] to break symmetry and converge Hund's m=2 state; then test stability of PM and Hund's states by seeding with their respective Σ_HF



spin_kick = {'up': np.diag([1.0, 1.0, 0]),
               'down': np.diag([0, 0, 0])}


h_int = make_h_int_kanamori_simple(U, U - 2*J, J, 0, norb=norb)

# ── Step 1: converge PM Bethe lattice DMFT ──────────────────────────────────
print("Running PM DMFT loop (Bethe, one-shot)...")
pm = ImpuritySolver(gf_struct=gf_struct, beta=beta, w_max=w_max, eps=eps_dlr, dc='cFLL')
pm_result = dmft_loop_bethe_hf(
    pm, t=t, h_int=h_int, mu_init=U,
    n_target=n_target, max_iter=100, mix=0.1, eps=1e-10,
    verbose=False, adjust_mu=True, mu_bracket=75.0,
    with_fock=True, one_shot=True, method='hybr', tol=1e-14, 
    #   spin_kick=spin_kick
      )    
mu = pm_result['mu']
n_pm = sum(pm_result['density'][bl].diagonal().real.sum() for bl in ['up', 'down'])
F_pm, F_pm_int, F_pm_kin = _impurity_free_energy_components(pm)
print(f"  mu     = {mu:.6f}")
print(f"  n_tot  = {n_pm:.6f}")
print(f"  F_PM   = {F_pm:.6f}  (E_int={F_pm_int:.6f}  E_kin={F_pm_kin:.6f})")
G0_conv = pm.G0_iw

#%%
# ── Helper ──────────────────────────────────────────────────────────────────
def solve_from(sigma_up, sigma_down, label, max_iter=500, tol=1e-10):
    s = ImpuritySolver(gf_struct=gf_struct, beta=beta, w_max=w_max,
                       eps=eps_dlr, dc='cFLL', force_real=True)
    for bl in ['up', 'down']:
        s.G0_iw[bl].data[:] = G0_conv[bl].data
    s.dc_fixed_value = 0.0
    s.Sigma_HF['up']   = sigma_up.astype(float)
    s.Sigma_HF['down'] = sigma_down.astype(float)
    s.solve(h_int, with_fock=True, one_shot=False, method='hybr', tol=tol)
    rho_up   = s.G_iw['up'].density().real
    rho_down = s.G_iw['down'].density().real
    n_up   = np.diag(rho_up)
    n_down = np.diag(rho_down)
    m      = n_up.sum() - n_down.sum()
    n_tot  = n_up.sum() + n_down.sum()
    F, F_int, F_kin = _impurity_free_energy_components(s)
    dF     = F - F_pm
    print(f"\n{'─'*62}")
    print(f"  {label}")
    print(f"  converged : {s.hf_converged}")
    print(f"  F         : {F:.6f}   (ΔF = {dF:+.6f} vs PM)")
    print(f"  F_int     : {F_int:.6f}   F_kin : {F_kin:.6f}")
    print(f"  n_tot     : {n_tot:.4f}   m = {m:.4f}")
    print(f"  n_up      : {np.round(n_up, 4)}")
    print(f"  n_down    : {np.round(n_down, 4)}")
    print(f"  Σ_up (diag): {np.round(np.diag(s.Sigma_HF['up']), 4)}")
    print(f"  Σ_dn (diag): {np.round(np.diag(s.Sigma_HF['down']), 4)}")
    return s, F


# ══════════════════════════════════════════════════════════════════════════════
#%%
# ── A: Targeting approach (original test) ────────────────────────────────────
print("\n" + "="*62)
print("  TEST A: EnsembleSolver, custom_proposals targeting [1,1,0]/[0,0,0]")
print("="*62)

rho_up_target   = np.diag([1., 1., 0.]).astype(float)
rho_down_target = np.diag([0., 0., 0.]).astype(float)

custom_dm = DensityMatrixProposals(
    n_proposals       = 1,
    n_targeting_steps = 100,
    targeting_alpha   = 2.0,
    half_occ_prob     = 0.0,
    force_real        = True,
    custom_proposals  = [{'up': rho_up_target, 'down': rho_down_target}],
)

ens = IncoherentEnsembleSolver(
    gf_struct            = gf_struct,
    beta                 = beta,
    w_max                = w_max,
    eps                  = eps_dlr,
    dc_fixed_value       = 0.0,
    force_real           = True,
    enforce_paramagnetic = True,
    prune_tol            = 0.1,
    # dc_fixed_value       = 0.0,
    verbosity            = 'debug',
)
ens.G0_iw['up'].data[:]   = G0_conv['up'].data
ens.G0_iw['down'].data[:] = G0_conv['down'].data

ens.solve(
    h_int               = h_int,
    proposal_generators = [custom_dm],
    mu                  = mu,
    n_elec_total        = n_target,
    with_fock           = True,
    hf_method           = 'hybr',
    hf_tol              = 1e-8,
    seed                = 0,
)

if ens.solutions is not None and len(ens.solutions) > 0:
    sol = ens.solutions[0]
    rho_up_sol   = np.diag(sol.density['up'].real)
    rho_down_sol = np.diag(sol.density['down'].real)
    m_sol        = rho_up_sol.sum() - rho_down_sol.sum()
    print(f"F={sol.free_energy:.6f}  ΔF={sol.free_energy-F_pm:+.6f}  m={m_sol:.4f}")
    print(f"n_up={np.round(rho_up_sol,4)}  n_dn={np.round(rho_down_sol,4)}")
else:
    print("No converged solutions found.")

# ──────────────────────────────────────────────────────────────────────────────
# ── B: Direct strong Σ kick — bypass targeting entirely ─────────────────────
#     Σ_up = diag(−A,−A,+A)   Σ_down = diag(+A,+A,+A)
#     Tests whether any amplitude A reliably seeds the Hund's basin.
# ──────────────────────────────────────────────────────────────────────────────
#%%
print("\n" + "="*62)
print("  TEST B: solve_from with direct strong Σ kick, sweep A")
print("="*62)

amplitudes = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
results_direct = []
for A in amplitudes:
    sig_up   = np.diag([-A, -A, +A])
    sig_down = np.diag([ A,  A, +A])
    s, F = solve_from(sig_up, sig_down,
                      f"direct Σ kick  A={A:5.1f}")
    rho_up = np.diag(s.G_iw['up'].density().real)
    rho_dn = np.diag(s.G_iw['down'].density().real)
    m = rho_up.sum() - rho_dn.sum()
    results_direct.append((A, s.hf_converged, m, F))

print("\n" + "─"*62)
print(f"  {'A':>6}  {'conv':>5}  {'m':>7}  {'F':>10}  {'ΔF':>10}")
for A, conv, m, F in results_direct:
    print(f"  {A:6.1f}  {str(conv):>5}  {m:7.4f}  {F:10.4f}  {F-F_pm:+10.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# ── C: Ensemble with prior_solutions carrying the strong Σ directly ──────────
#     prior_solutions path recycles Σ_HF verbatim — zero targeting steps.
# ──────────────────────────────────────────────────────────────────────────────
#%%
print("\n" + "="*62)
print("  TEST C: EnsembleSolver, prior_solutions with direct Σ kick (A=10)")
print("="*62)

A_kick = 10.0
mock_solution = Solution(
    G_iw          = G0_conv,                 # placeholder — only used by pruning distance
    Sigma_HF      = {
        'up':   np.diag([-A_kick, -A_kick, +A_kick]).astype(float),
        'down': np.diag([ A_kick,  A_kick,  A_kick]).astype(float),
    },
    density       = {'up': np.diag([1., 1., 0.]), 'down': np.zeros((norb, norb))},
    free_energy   = 0.0,
    weight        = 1.0,
    magnetisation = 2.0,
    n_total       = 2.0,
)

direct_dm = DensityMatrixProposals(
    n_proposals       = 1,
    n_targeting_steps = 0,        # no targeting — Σ used as-is
    targeting_alpha   = 0.0,
    half_occ_prob     = 0.0,
    force_real        = True,
    prior_solutions   = [mock_solution],
)

ens2 = IncoherentEnsembleSolver(
    gf_struct            = gf_struct,
    beta                 = beta,
    w_max                = w_max,
    eps                  = eps_dlr,
    dc_fixed_value       = 0.0,
    force_real           = True,
    enforce_paramagnetic = False,
    prune_tol            = 0.0,
    verbosity            = 'normal',
)
ens2.G0_iw['up'].data[:]   = G0_conv['up'].data
ens2.G0_iw['down'].data[:] = G0_conv['down'].data
ens2.solve(h_int, [direct_dm], mu=mu, n_elec_total=n_target,
           with_fock=True, hf_method='hybr', hf_tol=1e-8, seed=0)

if ens2.solutions is not None and len(ens2.solutions) > 0:
    sol2 = ens2.solutions[0]
    rho_up2 = np.diag(sol2.density['up'].real)
    rho_dn2 = np.diag(sol2.density['down'].real)
    print(f"F={sol2.free_energy:.6f}  ΔF={sol2.free_energy-F_pm:+.6f}  m={sol2.magnetisation:.4f}")
    print(f"n_up={np.round(rho_up2,4)}  n_dn={np.round(rho_dn2,4)}")
else:
    print("No converged solutions found.")

print("\nDone.")
# %%
