"""hf_dmft_landscape.py
=======================
Front-end for the HF-DMFT fixed-point landscape search on the Bethe lattice.
All logic lives in hf_dmft_landscape_backend.py.

Set model parameters and kick strategy in Cell 1, then run cells sequentially:
  Cell 1 — model parameters + kick generation
  Cell 2 — run landscape (DMFT loops → HDF5 archive)
  Cell 3 — load archive, print sorted table
  Cell 4 — plots (reads archive independently)
"""

# %%  ─── Cell 1: imports, model parameters, kick generation ────────────────
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
from dmft_driver import make_h_int_kanamori_simple, make_h_int_dft_like
from hf_dmft_landscape_backend import (
    generate_sobol_kicks,
    generate_dm_kicks,
    pm_kick,
    run_landscape,
    load_runs,
    orbital_symmetry_projector,
)

plt.rcParams.update({'font.size': 12})

# ══════════════════════════════════════════════════════════════════════════════
#  Model parameters
# ══════════════════════════════════════════════════════════════════════════════
t       = 1.0
beta    = 40.0
w_max   = 20.0
eps_dlr = 1e-13

norb     = 5
n_target = 5       # total filling (summed over spin and orbital)

U = 3.5
J = 0.3





# Crystal field: t2g (orbs 0-2) vs e_g (orbs 3-4), with small orbital splittings
cf_mag = 1
cf_t2g = -2/5 * cf_mag
cf_eg  =  3/5 * cf_mag

crystal_field = {
    # 'up':   np.array([cf_t2g+0.01, cf_t2g+0.02, cf_t2g+0.03, cf_eg+0.01, cf_eg+0.02]),
    'up':   np.array([cf_t2g, cf_t2g, cf_t2g, cf_eg, cf_eg]),
    # 'down': np.array([cf_t2g+0.01, cf_t2g+0.02, cf_t2g+0.03, cf_eg+0.01, cf_eg+0.02]),
    'down':   np.array([cf_t2g, cf_t2g, cf_t2g, cf_eg, cf_eg]),
}

gf_struct = [('up', norb), ('down', norb)]

# ── Interaction type ─────────────────────────────────────────────────────────
# 'dft-like'  : H = U/2 N(N-1) - J Sz²  (Stoner/DFT mean-field Hund)
# 'kanamori'  : full Kanamori with spin-flip and pair-hopping
interaction_type = 'dft-like'

if interaction_type == 'kanamori':
    h_int = make_h_int_kanamori_simple(U, U - 2*J, J, J, norb=norb)
elif interaction_type == 'dft-like':
    h_int = make_h_int_dft_like(U, J, norb)
else:
    raise ValueError(f'Unknown interaction_type={interaction_type!r}')



# ══════════════════════════════════════════════════════════════════════════════
#  Kick strategy
# ══════════════════════════════════════════════════════════════════════════════
N_sobol = 0          # Sobol Σ_HF kicks
N_dm    = 150          # DM-targeted kicks
seed    = 42

sigma_bound      = U / 2   # half-width of Sobol diagonal range
sigma_offset     = U       # centre of range
offdiag_fraction = 0.15

n_target_steps  = 50       # DM targeting iterations before release
targeting_alpha = 1.5      # Lagrange step size α

kicks = (
    [pm_kick(norb, gf_struct)]                                    # PM reference
    # + generate_sobol_kicks(N_sobol, norb, gf_struct,
    #                        sigma_bound=sigma_bound,
    #                        sigma_offset=sigma_offset,
    #                        offdiag_fraction=offdiag_fraction,
    #                        seed=seed)
    + generate_dm_kicks(N_dm, n_target, norb, gf_struct,
                        half_occ_prob=0.40,
                        randomize_oxidation=True,
                        seed=seed + 1)
)

mpi.report(f'  Total kicks: {len(kicks)}'
           f'  (1 PM + {N_sobol} Sobol + {N_dm} DM-targeted)')

# ── Orbital symmetry projector ─────────────────────────────────────────────────────
# t2g (orbitals 0-2) and eg (orbitals 3-4) are each internally equivalent.
# Each run has probability p_sym of enforcing orbital symmetry at EVERY
# iteration, steering it toward the orbitally-symmetric saddle points.
# Set p_sym=0.0 to disable, p_sym=1.0 to enforce in all runs.
equiv_groups = [[0, 1 ,2], [3, 4]] # in tm oxides the middle t2g is in the direction of the other atom so inequivalent
p_sym        = 0.5
orb_proj     = orbital_symmetry_projector(equiv_groups, gf_struct)
for k in kicks:
    k['sym_proj'] = orb_proj
    k['p_sym']    = p_sym


# ══════════════════════════════════════════════════════════════════════════════
#  Output paths
# ══════════════════════════════════════════════════════════════════════════════
run_tag      = (f'Htype_{interaction_type}_norb{norb}_ntgt{n_target}_U{U}_J{J}'
                f'_cf{cf_mag}_beta{int(beta)}')
plot_dir     = f'Plots_landscape_{run_tag}'
archive_path = f'{plot_dir}/results.h5'
os.makedirs(plot_dir, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Solver factory + DMFT settings
# ══════════════════════════════════════════════════════════════════════════════
def make_solver():
    return ImpuritySolver(
        gf_struct=gf_struct, beta=beta, w_max=w_max, eps=eps_dlr, dc='cFLL')


dmft_kwargs = dict(
    max_iter   = 2500,
    mix        = 0.05,
    eps        = 1e-5,
    adjust_mu  = True,
    mu_bracket = 30.0,
    with_fock  = True,
    one_shot   = True,
    method     = 'krylov',
    # tol        = 1e-5,
    print_every = 20,
)

params_dict = dict(
    norb=norb, U=U, J=J, beta=beta, t=t,
    n_target=n_target, cf_mag=cf_mag,
    N_sobol=N_sobol, N_dm=N_dm, seed=seed,
    n_target_steps=n_target_steps,
    targeting_alpha=targeting_alpha,
)


# %%  ─── Cell 2: run landscape ─────────────────────────────────────────────
n_conv = run_landscape(
    kicks          = kicks,
    solver_factory = make_solver,
    t              = t,
    h_int          = h_int,
    mu_init        = U,
    n_target       = n_target,
    crystal_field  = crystal_field,
    archive_path   = archive_path,
    params_dict    = params_dict,
    n_target_steps = n_target_steps,
    targeting_alpha = targeting_alpha,
    **dmft_kwargs,
)


# %%  ─── Cell 3: load archive and print sorted table ───────────────────────
runs = load_runs(archive_path)
if not runs:
    raise RuntimeError(
        f'No converged runs in {archive_path} — run Cell 2 first.')

F_min = runs[0]['F']

with __import__('h5').HDFArchive(archive_path, 'r') as _ar:
    _p = _ar['params']

mpi.report('\n' + '═' * 70)
mpi.report(f'  Converged runs: {len(runs)}'
           f'  |  U={_p["U"]}  J={_p["J"]}  β={_p["beta"]}'
           f'  norb={_p["norb"]}  N={_p["n_target"]}')
mpi.report('═' * 70)
mpi.report(
    f"  {'#':>4}  {'F':>11}  {'E_int':>11}  {'E_cf':>10}  {'E_kin':>11}"
    f"  {'ΔF':>9}  {'m':>8}  {'N':>7}"
)
mpi.report('  ' + '─' * 76)
for i, r in enumerate(runs):
    dF  = r['F'] - F_min
    tag = '*' if i == 0 else ' '
    mpi.report(
        f'  {i:>3}{tag} '
        f'{r["F"]:>11.5f}  {r["F_int"]:>11.5f}  {r["F_cf"]:>10.5f}  {r["F_kin"]:>11.5f}'
        f'  {dF:>+9.5f}  {r["m"]:>+8.4f}  {r["n_total"]:>7.4f}'
    )
    mpi.report(f'         n_up = {np.round(r["n_up"],   4)}')
    mpi.report(f'         n_dn = {np.round(r["n_down"], 4)}')
mpi.report('  ' + '─' * 76)
mpi.report('  (* lowest free energy)')
mpi.report('═' * 70 + '\n')


# %%  ─── Cell 4: plots ─────────────────────────────────────────────────────
runs = load_runs(archive_path)
with __import__('h5').HDFArchive(archive_path, 'r') as _ar:
    _p = _ar['params']

F_arr    = np.array([r['F']       for r in runs])
F_min     = F_arr.min()
Fint_arr = np.array([r['F_int']   for r in runs])
Fkin_arr = np.array([r['F_kin']   for r in runs])
dF_arr   = F_arr - F_min
dFint_arr = Fint_arr - Fint_arr.min()
m_arr    = np.array([r['m']       for r in runs])
n_arr    = np.array([r['n_total'] for r in runs])

# for every run get the magnetization inverted twin, create new arrays and concatenate to the original ones, then sort by energy. This is just for visualisation purposes, it doesn't mean that the twin solutions are actually found by the solver (they are not, in general).
F_arr    = np.concatenate([F_arr, F_arr])
Fint_arr = np.concatenate([Fint_arr, Fint_arr])
Fkin_arr = np.concatenate([Fkin_arr, Fkin_arr])
dF_arr   = np.concatenate([dF_arr, dF_arr])
dFint_arr = np.concatenate([dFint_arr, dFint_arr])
# dFkin_arr = np.concatenate([dFkin_arr, dFkin_arr])
m_arr    = np.concatenate([m_arr, -m_arr])
n_arr    = np.concatenate([n_arr, n_arr])



title = (f'HF-DMFT landscape  |  norb={_p["norb"]}  U={_p["U"]}'
         f'  J={_p["J"]}  β={_p["beta"]}  cf={_p["cf_mag"]}'
         f'\n{len(runs)} converged / {1+_p["N_sobol"]+_p["N_dm"]} seeds total')

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle(title, fontsize=12, y=1.02)

colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Panel 1: m vs ΔF, colour = N_total
ax = axes[0]
sc = ax.scatter(m_arr, dF_arr, s=70, color = 'blue',
                # vmin=n_arr.mean() - 2*n_arr.std(),
                # vmax=n_arr.mean() + 2*n_arr.std(),
                alpha=0.35, edgecolors='k', linewidths=0.7, zorder=3)
fig.colorbar(sc, ax=ax, pad=0.02).set_label('N total', fontsize=11)
ax.scatter([runs[0]['m']], [0], marker='*', s=350, color='none',
           edgecolors='gold', linewidths=1.5, zorder=5,
           label=f'GS  m={runs[0]["m"]:+.3f}')
ax.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel(r'$m = \langle n_\uparrow\rangle - \langle n_\downarrow\rangle$')
ax.set_ylabel(r'Relative  energy [eV$^{-1}$]')
ax.set_title('Fixed points: m vs energy')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, lw=0.5)


# Panel 3: ΔF histogram
ax = axes[1]
ax.hist(dF_arr, bins=max(5, len(runs) // 2),
        color=colorcycle[0], edgecolor='k', linewidth=0.5, alpha=0.85)
ax.set_xlabel(r'Relative  energy [eV$^{-1}$]')
ax.set_ylabel('Count')
# ax.set_title(r'$\Delta F$ distribution')
ax.grid(True, alpha=0.3, axis='y', lw=0.5)

plt.tight_layout()
plt.savefig(f'{plot_dir}/hf_dmft_landscape.jpg', dpi=300, bbox_inches='tight')
# %%
