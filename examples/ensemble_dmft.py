#%%
"""ensemble_dmft.py
==================
Multi-orbital Bethe-lattice HF-DMFT with incoherent ensemble impurity solver.

Workflow
--------
1. Run a standard symmetry-unbroken (paramagnetic) HF-DMFT loop to find the
   converged bath G0 and chemical potential μ.
2. On top of the converged bath, run one IncoherentEnsembleSolver shot:
   – Discover all accessible HF saddle points using two complementary
     initial-condition generators (Sobol + density-matrix targeting).
   – Compute the impurity free energy of each saddle point.
   – Boltzmann-weight an ensemble-averaged Green's function G_ens(iω).
   – Extract the many-body self-energy Σ_ens(iω) = G0⁻¹ − G_ens⁻¹.
3. Plot the saddle-point landscape and the ensemble spectral functions.
"""

#%%
import os
import sys

# ── Single-thread BLAS before any numerical import ──────────────────────────
os.environ["OMP_NUM_THREADS"]    = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]    = "1"

# ── Path setup (dev install) ─────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'python'))

import numpy as np
import matplotlib.pyplot as plt
import triqs.utility.mpi as mpi
from triqs.gf import make_gf_imfreq, Gf, MeshReFreq, iOmega_n, inverse

from triqs_hartree_fock import ImpuritySolver
from triqs_hartree_fock import (
    IncoherentEnsembleSolver,
    SobolSigmaProposals,
    DensityMatrixProposals,
)
from dmft_driver import dmft_loop_bethe_hf, dmft_loop_ensemble_hf, make_h_int_kanamori_simple

plt.rcParams.update({'font.size': 12})

# ══════════════════════════════════════════════════════════════════════════════
#  Model parameters
# ══════════════════════════════════════════════════════════════════════════════
t        = 1.0      # Bethe-lattice hopping; half-bandwidth W = 2t
beta     = 40.0     # inverse temperature
w_max    = 20.0     # DLR energy cutoff
eps_dlr  = 1e-9     # DLR accuracy

norb     = 3        # number of orbitals
n_target = 2     # half-filling: norb electrons total

U = 8.0
J = 1.0


gf_struct = [('up', norb), ('down', norb)]

# ══════════════════════════════════════════════════════════════════════════════
#  Step 1 – Paramagnetic DMFT loop (finds converged G0 and μ)
# ══════════════════════════════════════════════════════════════════════════════
mpi.report('\n' + '=' * 70)
mpi.report(f'  Step 1: PM HF-DMFT   |   norb={norb}   U={U}   J={J}   β={beta}')
mpi.report('=' * 70 + '\n')

h_int = make_h_int_kanamori_simple(U, U - 2*J, J, 0, norb=norb)
# h_int = make_h_int_kanamori_simple(U, U - 2*J, J, J, norb=norb) # this can generate a goldstone mode without crystal field

pm_solver = ImpuritySolver(
    gf_struct = gf_struct,
    beta      = beta,
    w_max     = w_max,
    eps       = eps_dlr,
    dc        = 'cFLL',
)

pm_result = dmft_loop_bethe_hf(
    solver    = pm_solver,
    t         = t,
    h_int     = h_int,
    mu_init   = U,          # particle-hole symmetric starting point
    n_target  = n_target,
    max_iter  = 100,
    mix       = 0.1,
    eps       = 1e-12,
    verbose   = True,
    adjust_mu = True,
    mu_bracket = 75.0,
    with_fock = True,
    one_shot  = True,
    method    = 'hybr',
    tol       = 1e-14,
)

mu = pm_result['mu']
G0_conv = pm_solver.G0_iw   # converged bath Green's function

mpi.report('\n' + '-' * 70)
mpi.report(f'  PM DMFT converged:  μ = {mu:+.6f}   '
           f'n_iter = {pm_result["n_iter"]}   '
           f'converged = {pm_result["converged"]}')

n_up   = pm_result['density']['up'].diagonal().real
n_down = pm_result['density']['down'].diagonal().real
mpi.report(f'  n_up   = {np.round(n_up, 4)}')
mpi.report(f'  n_down = {np.round(n_down, 4)}')
mpi.report(f'  m      = {np.round(n_up - n_down, 4)}')
mpi.report('-' * 70 + '\n')

#%%
# ══════════════════════════════════════════════════════════════════════════════
#  Step 2 – Ensemble impurity solve on the converged bath
# ══════════════════════════════════════════════════════════════════════════════
mpi.report('\n' + '=' * 70)
mpi.report('  Step 2: Ensemble impurity solve')
mpi.report('=' * 70 + '\n')

# ── Proposal generators ───────────────────────────────────────────────────────
# SobolSigmaProposals are less effective than DM targeting for broken-symmetry
# searches; keep the class available but default to DM proposals only.
#
# sobol_proposals = SobolSigmaProposals(
#     n_samples        = 300,
#     sigma_bound      = U / 2,
#     sigma_offset     = mu,
#     offdiag_fraction = 0.2,
# )

dm_proposals = DensityMatrixProposals(
    n_proposals       = 200,
    n_targeting_steps = 30,
    targeting_alpha   = 0.5,
    half_occ_prob     = 0.10,
    force_real        = True,
    # prior_solutions is None on first call; updated automatically in
    # dmft_loop_ensemble_hf for subsequent iterations.
)

# ── Ensemble solver ────────────────────────────────────────────────────────────
ensemble = IncoherentEnsembleSolver(
    gf_struct      = gf_struct,
    beta           = beta,
    w_max          = w_max,
    eps            = eps_dlr,
    dc_fixed_value = 0.0,   # fixed DC shift (same as in state_discovery.py)
    force_real     = True,
)

ensemble.G0_iw['up'].data[:]   = G0_conv['up'].data
ensemble.G0_iw['down'].data[:] = G0_conv['down'].data
ensemble.solve(
    h_int              = h_int,
    proposal_generators = [dm_proposals],
    mu                 = mu,
    n_elec_total       = n_target,
    with_fock          = True,
    hf_method          = 'hybr',
    hf_tol             = 1e-8,
    seed               = 42,
)

# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 – Summary
# ══════════════════════════════════════════════════════════════════════════════
mpi.report('\n' + '=' * 70)
mpi.report(f'  ENSEMBLE SUMMARY   (norb={norb}   U={U}   J={J}   β={beta})')
mpi.report('=' * 70)
mpi.report(f'  Saddle points found : {ensemble.n_converged}')
mpi.report(f'  Failed proposals    : {ensemble.n_failed}')
mpi.report(f'  β·ΔF (max)          : '
           f'{ensemble.beta * (ensemble.free_energies.max() - ensemble.free_energies.min()):.3f}')
mpi.report('\n' + repr(ensemble.solutions))
mpi.report('\n' + '=' * 70 + '\n')

#%%
# ══════════════════════════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════════════════════════

colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# ── 1. Saddle-point landscape ─────────────────────────────────────────────────
landscape_fig = ensemble.plot_landscape(
    show      = False,
    save_path = f'ensemble_landscape_norb{norb}_U{U}_J{J}.jpg',
)
plt.show()

#%%
from triqs.gf import *
from triqs.plot.mpl_interface import oplot
# ── 2. Ensemble G(iω) on Imaginary time axis ───────────────────────────────────────────────
def make_gf_tau(G_iw, norb=norb, beta=beta):
    """Make a GfImTime from a GfImFreq by Fourier transform."""
    tau_mesh = MeshImTime(beta, S='Fermion', n_tau =10000) 
    G_tau = Gf(mesh=tau_mesh, target_shape=[norb, norb])
    G_imfreq = make_gf_imfreq(G_iw,   n_iw=2500)

    G_tau.set_from_fourier(G_imfreq)

    return G_tau

Gtau_ens_up   = make_gf_tau(ensemble.G_iw['up'])
Gtau_ens_down = make_gf_tau(ensemble.G_iw['down'])

oplot(Gtau_ens_up[0,0].real, color=colorcycle[0], lw=2, label='ens up 0 0')
oplot(Gtau_ens_up[1,1].real, color=colorcycle[1], lw=2, label='ens up 1 1')
oplot(Gtau_ens_down[0,0].real, color=colorcycle[0], lw=2, label='ens down 0 0')
oplot(Gtau_ens_down[1,1].real, color=colorcycle[1], lw=2, label='ens down 1 1')

plt.title(
    f'Ensemble Green\'s function  G(τ)   |   U={U}   J={J}   β={beta}',
    fontsize=14, y=1.02
)
plt.xlabel(r'$\tau$', fontsize=13)
plt.ylabel(r'$G(\tau)$', fontsize=13)
# plt.xlim(0, beta)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()


# plt.savefig(f'ensemble_Gtau_norb{norb}_U{U}_J{J

#%%
# ── 3. Ensemble Σ(iω) on Matsubara axis ──────────────────────────────────────
n_iw_dense = 2500

Sig_ens_up   = make_gf_imfreq(ensemble.Sigma_iw['up'],   n_iw=n_iw_dense)
Sig_ens_down = make_gf_imfreq(ensemble.Sigma_iw['down'], n_iw=n_iw_dense)

oplot(ensemble.Sigma_iw['up'][0,0], color=colorcycle[0], lw=2, label='ens up 0 0')
# oplot(ensemble.Sigma_iw['up'][1,1], color=colorcycle[0], lw=2, label='ens up 1 1')


plt.title(
    f'Ensemble self-energy  Σ(iω) = G0⁻¹ − G_ens⁻¹   |   U={U}   J={J}   β={beta}',
    fontsize=14, y=1.02
)

plt.xlim(0, 20)

plt.tight_layout()
plt.savefig(f'ensemble_Sigma_norb{norb}_U{U}_J{J}.jpg', dpi=200, bbox_inches='tight')
plt.show()

#%%
# ── 4. Real-frequency spectral functions via Padé ────────────────────────────
w_window = (-8.0, 8.0)
n_w      = 1000
n_pade   = 200

G_ens_up  = make_gf_imfreq(ensemble.G_iw['up'],   n_iw=n_iw_dense)  
G_ens_down = make_gf_imfreq(ensemble.G_iw['down'], n_iw=n_iw_dense)
G_pm_up    = make_gf_imfreq(pm_solver.G_iw['up'],   n_iw=n_iw_dense)
G_pm_down  = make_gf_imfreq(pm_solver.G0_iw['down'], n_iw=n_iw_dense)

def _pade_block(G_iw_dense):
    """Padé-continue a full BlockGf component to real frequencies."""
    Gw = Gf(mesh=MeshReFreq(window=w_window, n_w=n_w),
            target_shape=[norb, norb])
    Gw.set_from_pade(G_iw_dense, n_points=n_pade)
    return Gw

Gw_ens_up   = _pade_block(G_ens_up)
Gw_ens_down = _pade_block(G_ens_down)
Gw_pm_up    = _pade_block(G_pm_up)
Gw_pm_down  = _pade_block(G_pm_down)

w_vals = np.array(list(Gw_ens_up.mesh.values())).real

fig, axes = plt.subplots(1, norb, figsize=(6 * norb, 4.5), sharey=True)
if norb == 1:
    axes = [axes]

for orb in range(norb):
    ax = axes[orb]
    ax.plot(w_vals, -Gw_ens_up[orb, orb].data.imag / np.pi,
            color=colorcycle[0], lw=2, label='ens up')
    ax.plot(w_vals, -Gw_ens_down[orb, orb].data.imag / np.pi,
            color=colorcycle[1], lw=2, label='ens down')
    ax.plot(w_vals, -Gw_pm_up[orb, orb].data.imag / np.pi,
            color='gray', lw=1.2, ls='--', alpha=0.7, label='PM')
    ax.set_xlabel(r'$\omega$', fontsize=13)
    ax.set_ylabel(r'$-\mathrm{Im}\,G(\omega)/\pi$', fontsize=13) if orb == 0 else None
    ax.set_title(f'Orbital {orb}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    f'Spectral function  A(ω) = −Im G(ω)/π   |   U={U}   J={J}   β={beta}',
    fontsize=14, y=1.02
)
plt.tight_layout()
plt.savefig(f'ensemble_Aw_norb{norb}_U{U}_J{J}.jpg', dpi=200, bbox_inches='tight')
plt.show()

# %%
#%%
# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 – Ensemble DMFT self-consistency loop
# ══════════════════════════════════════════════════════════════════════════════
# Starting from the converged PM bath G0 and the saddle points already found
# in Step 2, iterate the Bethe SC with G_ens as the local Green's function.
# The DM proposals are warm-started from the previous iteration's SolutionSet.
#
# Typical convergence: 3–6 iterations are usually sufficient for qualitative
# results; increase max_iter for production runs.

mpi.report('\n' + '=' * 70)
mpi.report('  Step 4: Ensemble DMFT self-consistency loop')
mpi.report('=' * 70 + '\n')

# Seed the proposals with the solutions already found in Step 2 so the first
# ensemble-DMFT iteration reuses them and only explores new territory.
dm_proposals.prior_solutions = ensemble.solutions

ens_dmft_result = dmft_loop_ensemble_hf(
    ensemble_solver = ensemble,
    dm_proposals    = dm_proposals,
    h_int           = h_int,
    G0_iw           = G0_conv,        # start from PM bath
    t               = t,
    mu_init         = mu,
    n_target        = n_target,
    n_elec_total    = n_target,
    max_iter        = 20,
    eps             = 1e-3,
    mix             = 0.5,
    adjust_mu       = True,
    mu_bracket      = 25.0,
    with_fock       = True,
    hf_method       = 'hybr',
    hf_tol          = 1e-8,
    seed            = 99,
    verbose         = True,
)

mpi.report('\n' + '=' * 70)
mpi.report(f'  Ensemble DMFT converged : {ens_dmft_result["converged"]}')
mpi.report(f'  Iterations              : {ens_dmft_result["n_iter"]}')
mpi.report(f'  Final μ                 : {ens_dmft_result["mu"]:+.6f}')
mpi.report(f'  Unique saddle points    : {ensemble.n_converged}')
mpi.report('\n' + repr(ensemble.solutions))
mpi.report('\n' + '=' * 70 + '\n')

#%%
# ── Plot ensemble DMFT G(τ)


final_ens = ens_dmft_result['ensemble']
# final_ens_G0 = final_ens.G0_iw
final_ens_G = final_ens.G_iw
final_ens_Sigma = final_ens.Sigma_iw

Gtau_ens_up   = make_gf_tau(final_ens_G['up'])
Gtau_ens_down = make_gf_tau(final_ens_G['down'])



oplot(Gtau_ens_up[0,0].real, color=colorcycle[0], lw=2, label='ens DMFT up 0 0')
oplot(Gtau_ens_up[1,1].real, color=colorcycle[1], lw=2, label='ens DMFT up 1 1')
oplot(Gtau_ens_up[2,2].real, color=colorcycle[2], lw=2, label='ens DMFT up 1 1')


# plt.ylim(-0.002, 0)
# oplot


# print the ensemble density

with np.printoptions(precision=4, suppress=True):
    print('Ensemble DMFT densities:')
    mup = final_ens_G['up'].density()
    mdn = final_ens_G['down'].density()
    
    print(f'  n_up: \n {mup}')
    print(f'  n_down: \n {mdn}')
# %%


oplot(final_ens_Sigma['up'].real, color=colorcycle[0], lw=2, label='ens up 0 0', x_window=(0, 20))



#%% pade analytically continue and plot G(w)

G_iw_freq_up = make_gf_imfreq(final_ens_G['up'], n_iw=n_iw_dense)
Gw_ens_up   = _pade_block(G_iw_freq_up)

G_iw_freq_down = make_gf_imfreq(final_ens_G['down'], n_iw=n_iw_dense)
Gw_ens_down   = _pade_block(G_iw_freq_down)



oplot(-1/np.pi*Gw_ens_up[0,0].imag, color=colorcycle[0], lw=2, label='ens DMFT up 0 0')
oplot(-1/np.pi*Gw_ens_up[1,1].imag, color=colorcycle[1], lw=2, label='ens DMFT up 1 1')
oplot(-1/np.pi*Gw_ens_up[2,2].imag, color=colorcycle[2], lw=2, label='ens DMFT up 1 1')
# oplot(-1/np.pi*Gw_ens_down[0,0].imag, color=colorcycle[0], lw=2, label='ens DMFT down 0 0')
# oplot(-1/np.pi*Gw_ens_down[1,1].imag, color=colorcycle[1], lw=2, label='ens DMFT down 1 1')

with np.printoptions(precision=4, suppress=True):
    mup_w = Gw_ens_up.density()
    mdn_w = Gw_ens_down.density()
    print('Ensemble DMFT densities from G(w):')
    print(f'  n_up: \n {mup_w}')
    print(f'  n_down: \n {mdn_w}')

# %% debug here`


