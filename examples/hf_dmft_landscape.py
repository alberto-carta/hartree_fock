"""hf_dmft_landscape.py
=======================
Explore all fully self-consistent HF-DMFT fixed points on the Bethe lattice
by launching many independent DMFT loops from different initial spin kicks.

Initial Σ_HF kicks are generated via the Sobol low-discrepancy sampler
(recycled from IncoherentEnsembleSolver).  Each kick seeds one fully
self-consistent Bethe-lattice DMFT loop.  No PM reference is needed.

Workflow
--------
1. Generate N_seeds Sobol-sampled Σ_HF initial conditions.
2. For each kick launch ``dmft_loop_bethe_hf`` to self-consistency.
3. Collect, deduplicate (by ‖ΔΣ‖_F < tol), and report all fixed points.
4. Plot: magnetisation vs ΔF, E_int/E_kin decomposition, basin histogram.
"""

#%%
import os
import sys

os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'python'))

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import triqs.utility.mpi as mpi

from triqs_hartree_fock import ImpuritySolver
from triqs_hartree_fock.incoherent_ensemble_solver import (
    SobolSigmaProposals,
    _impurity_free_energy_components,
)
from dmft_driver import dmft_loop_bethe_hf, make_h_int_kanamori_simple
from h5 import HDFArchive

colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.size': 12})

# ══════════════════════════════════════════════════════════════════════════════
#  Model parameters  (mirror ensemble_dmft.py for easy comparison)
# ══════════════════════════════════════════════════════════════════════════════
t        = 1.0
beta     = 40.0
w_max    = 20.0
eps_dlr  = 1e-13

norb     = 5
n_target = 6       # half-filling: norb electrons total

U = 3.0
J = 0.5

# ── Crystal field (same as ensemble_dmft.py) ────────────────────────────────
cf_mag = 0.5
cf_t2g = -2/5 * cf_mag
cf_eg  =  3/5 * cf_mag

crystal_field = {
    'up':   np.array([cf_t2g+0.01, cf_t2g+0.02, cf_t2g+0.03, cf_eg+0.01, cf_eg+0.02]),
    'down': np.array([cf_t2g+0.01, cf_t2g+0.02, cf_t2g+0.03, cf_eg+0.01, cf_eg+0.02]),
}

gf_struct = [('up', norb), ('down', norb)]
h_int     = make_h_int_kanamori_simple(U, U - 2*J, J, 0, norb=norb)

# ── Search parameters ─────────────────────────────────────────────────────────
N_seeds      = 20      # number of Sobol-sampled initial conditions
sigma_bound  = U/2       # half-width of Σ_HF sampling range — set to U so kicks
                       # span the full interaction scale
mu_init      = U       # standard particle-hole symmetric starting μ
seed_rng     = 42

# ── DMFT loop settings ───────────────────────────────────────────────────────
dmft_kwargs = dict(
    max_iter   = 300,
    mix        = 0.1,
    eps        = 1e-3,
    verbose    = False,
    adjust_mu  = True,
    mu_bracket = 30.0,
    with_fock  = True,
    one_shot   = True,
    method     = 'hybr',
)

plot_dir     = f'Plots_landscape_norb{norb}_ntarget{n_target}_U{U}_J{J}_cf{cf_mag}_beta{int(beta)}'
archive_path = f'{plot_dir}/results.h5'
os.makedirs(plot_dir, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  Generate initial conditions via Sobol sampler
# ══════════════════════════════════════════════════════════════════════════════
#%%
mpi.report('\n' + '=' * 70)
mpi.report(f'  HF-DMFT landscape  |  norb={norb}  U={U}  J={J}  β={beta}  cf={cf_mag}')
mpi.report('=' * 70)
mpi.report(f'\n  Generating {N_seeds} Sobol-sampled initial conditions ...')

sampler = SobolSigmaProposals(
    n_samples    = N_seeds,
    sigma_bound  = sigma_bound,
    sigma_offset = U,   
)

# generate() only needs norb and gf_struct; G0_iw is unused in SobolSigmaProposals
kicks = sampler.generate(G0_iw=None, n_elec_total=n_target, norb=norb,
                         gf_struct=gf_struct, seed=seed_rng)

# prepend a zero-kick (paramagnetic reference)
kicks = [{'up': np.zeros((norb, norb)), 'down': np.zeros((norb, norb))}] + kicks

mpi.report(f'  Total seeds (including PM zero-kick): {len(kicks)}')

# Initialise the archive with run metadata
with HDFArchive(archive_path, 'w') as ar:
    ar['params'] = dict(norb=norb, U=U, J=J, beta=beta, t=t,
                        n_target=n_target, cf_mag=cf_mag, N_seeds=N_seeds)

n_converged = 0

for i, kick in enumerate(kicks):
    solver = ImpuritySolver(
        gf_struct=gf_struct, beta=beta, w_max=w_max, eps=eps_dlr, dc='cFLL')

    result = dmft_loop_bethe_hf(
        solver        = solver,
        t             = t,
        h_int         = h_int,
        mu_init       = mu_init,
        n_target      = n_target,
        spin_kick     = kick,
        crystal_field = crystal_field,
        **dmft_kwargs,
    )

    if not result['converged']:
        mpi.report(f'  [{i+1:3d}/{len(kicks)}]  NOT CONVERGED')
        continue

    F, E_int, E_kin = _impurity_free_energy_components(solver)
    rho    = result['density']
    n_up   = rho['up'].diagonal().real
    n_down = rho['down'].diagonal().real
    m      = n_up.sum() - n_down.sum()
    n_tot  = n_up.sum() + n_down.sum()

    run_key = f'run_{n_converged:04d}'
    with HDFArchive(archive_path, 'a') as ar:
        ar.create_group(run_key)
        g = ar[run_key]
        g['G_iw']          = solver.G_iw
        g['G0_iw']         = solver.G0_iw
        g['Sigma_HF_up']   = solver.Sigma_HF['up'].copy()
        g['Sigma_HF_down'] = solver.Sigma_HF['down'].copy()
        g['F']             = F
        g['F_int']         = E_int
        g['F_kin']         = E_kin
        g['m']             = m
        g['n_total']       = n_tot
        g['mu']            = result['mu']
        g['n_up']          = n_up.copy()
        g['n_down']        = n_down.copy()
    n_converged += 1

    mpi.report(
        f'  [{i+1:3d}/{len(kicks)}]  '
        f'μ={result["mu"]:+.4f}  m={m:+.4f}  '
        f'F={F:.4f}  E_int={E_int:.4f}  E_kin={E_kin:.4f}'
    )

mpi.report(f'\n  {n_converged}/{len(kicks)} loops converged.')
mpi.report(f'  Results saved to: {archive_path}')


# ══════════════════════════════════════════════════════════════════════════════
#  Load archive → sort by F → report table
# ══════════════════════════════════════════════════════════════════════════════
#%%
def _load_runs(path):
    """Load all converged runs from the archive as a list sorted by F.

    Each run dict contains: G_iw, G0_iw, Sigma_HF (dict up/down),
    F, F_int, F_kin, m, n_total, mu, n_up, n_down.
    """
    with HDFArchive(path, 'r') as ar:
        run_keys = sorted(k for k in ar if k.startswith('run_'))
        runs = []
        for k in run_keys:
            g = ar[k]
            runs.append({
                'G_iw'    : g['G_iw'],
                'G0_iw'   : g['G0_iw'],
                'Sigma_HF': {
                    'up'  : np.array(g['Sigma_HF_up']),
                    'down': np.array(g['Sigma_HF_down']),
                },
                'F'       : float(g['F']),
                'F_int'   : float(g['F_int']),
                'F_kin'   : float(g['F_kin']),
                'm'       : float(g['m']),
                'n_total' : float(g['n_total']),
                'mu'      : float(g['mu']),
                'n_up'    : np.array(g['n_up']),
                'n_down'  : np.array(g['n_down']),
            })
    return sorted(runs, key=lambda r: r['F'])


runs = _load_runs(archive_path)
if not runs:
    raise RuntimeError(f'No converged runs found in {archive_path} — run the computation cell first.')
F_min = runs[0]['F']

mpi.report('\n' + '=' * 70)
mpi.report(f'  Converged runs: {len(runs)}')
mpi.report('=' * 70)
mpi.report(
    f"  {'#':>4}  {'F':>10}  {'E_int':>10}  {'E_kin':>10}  "
    f"{'ΔF':>8}  {'m':>8}  {'N':>7}"
)
mpi.report('  ' + '─' * 62)
for i, r in enumerate(runs):
    dF  = r['F'] - F_min
    tag = '*' if i == 0 else ' '
    mpi.report(
        f'  {i:>3}{tag} '
        f'{r["F"]:>10.5f}  {r["F_int"]:>10.5f}  {r["F_kin"]:>10.5f}  '
        f'{dF:>+8.5f}  {r["m"]:>+8.4f}  {r["n_total"]:>7.4f}'
    )
    mpi.report(f'         n_up = {np.round(r["n_up"],  4)}')
    mpi.report(f'         n_dn = {np.round(r["n_down"], 4)}')
mpi.report('  ' + '─' * 62)
mpi.report('  (* lowest free energy)')
mpi.report('=' * 70 + '\n')


# ══════════════════════════════════════════════════════════════════════════════
#  Landscape plot  (reads from archive — can be run independently)
# ══════════════════════════════════════════════════════════════════════════════
#%%
runs = _load_runs(archive_path)
with HDFArchive(archive_path, 'r') as ar:
    ar_params = ar['params']
F_min_plot = runs[0]['F']

F_arr    = np.array([r['F']       for r in runs])
Fint_arr = np.array([r['F_int']   for r in runs])
Fkin_arr = np.array([r['F_kin']   for r in runs])
dF_arr   = F_arr - F_min_plot
m_arr    = np.array([r['m']       for r in runs])
n_arr    = np.array([r['n_total'] for r in runs])


# now simple plot of magnetization vs interaction energy

fig = plt.figure(figsize=(6,5))
plt.scatter(m_arr, Fint_arr, s=80, color='b', alpha=0.85, edgecolors='k', linewidths=0.7, zorder=3)
plt.ylabel(r'$E_\mathrm{int}$')
plt.xlabel(r'Magnetisation $m$')
plt.title('Magnetisation vs interaction energy')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plot_dir}/m_vs_Eint.jpg', dpi=300, bbox_inches='tight')
plt.show()  
# %%
