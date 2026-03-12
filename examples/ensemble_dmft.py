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

colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.size': 12})

# ══════════════════════════════════════════════════════════════════════════════
#  Model parameters
# ══════════════════════════════════════════════════════════════════════════════
t        = 1      # Bethe-lattice hopping; half-bandwidth W = 2t
beta     = 40.0     # inverse temperature
w_max    = 20.0     # DLR energy cutoff
eps_dlr  = 1e-13     # DLR accuracy

norb     = 5        # number of orbitals
n_target = 5     # half-filling: norb electrons total

# U = 5.0
# J = 0.3


# norb     = 5        # number of orbitals
# n_target = 5     # half-filling: norb electrons total

U = 7.0
J = 0.5


gf_struct = [('up', norb), ('down', norb)]

# ══════════════════════════════════════════════════════════════════════════════
#  Step 1 – Paramagnetic DMFT loop (finds converged G0 and μ)
# ══════════════════════════════════════════════════════════════════════════════
mpi.report('\n' + '=' * 70)
mpi.report(f'  Step 1: PM HF-DMFT   |   norb={norb}   U={U}   J={J}   β={beta}')
mpi.report('=' * 70 + '\n')

# h_int = make_h_int_kanamori_simple(U, U, J, J, norb=norb)
# h_int = make_h_int_kanamori_simple(U, U - 2*J, J, 0, norb=norb)
h_int = make_h_int_kanamori_simple(U, U - 2*J, J, J, norb=norb) # this can generate a goldstone mode without crystal field

# ── Crystal field (optional) ─────────────────────────────────────────────────
# Diagonal on-site energies per spin block, passed to both DMFT loops.
# The field enters the Weiss field as:
#   G0⁻¹(iω) = (iω + μ)·I − diag(ε_cf) − t²·G_loc(iω)
#
# Set to None to run without a crystal field (default, orbitally degenerate).

cf_mag = 1
cf_t2g = -2/5*cf_mag
cf_eg  = 3/5*cf_mag

crystal_field = {
    'up':   np.array([cf_t2g, cf_t2g, cf_t2g, cf_eg, cf_eg]),
    'down': np.array([cf_t2g, cf_t2g, cf_t2g, cf_eg, cf_eg]),
}

pm_solver = ImpuritySolver(
    gf_struct = gf_struct,
    beta      = beta,
    w_max     = w_max,
    eps       = eps_dlr,
    dc        = 'cFLL',
)

# mu_init = U

pm_result = dmft_loop_bethe_hf(
    solver        = pm_solver,
    t             = t,
    h_int         = h_int,
    mu_init       = U,          # particle-hole symmetric starting point
    n_target      = n_target,
    max_iter      = 100,
    mix           = 0.1,
    eps           = 1e-12,
    verbose       = True,
    adjust_mu     = True,
    mu_bracket    = 25.0,
    crystal_field = crystal_field,
    with_fock     = True,
    one_shot      = True,
    method        = 'hybr',
    tol           = 1e-14,
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
import matplotlib.pyplot as plt
from triqs.gf import *
from triqs.plot.mpl_interface import oplot
pm_solver.G_iw

oplot(pm_solver.G_iw['up'][0,0], '-o', label='G_up')


# mesh_re = MeshReFreq(mesh=solver.G_iw.mesh)

# Get back Green's function on the full Matsubara mesh, no Fourier transform needed
Giw_from_dlr_up = make_gf_imfreq(pm_solver.G_iw['up'], n_iw=2500)
Giw_from_dlr_down = make_gf_imfreq(pm_solver.G_iw['down'], n_iw=2500)

oplot(Giw_from_dlr_up[0,0], '-', label='G_up from DLR')
oplot(Giw_from_dlr_down[0,0], '-', label='G_down from DLR')

plt.xlim(0, 20)

#%%


colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
# set tau mesh and analytically continue to real freq
Gw_up = Gf(mesh=MeshReFreq(window = (-5.0,5.0), n_w=1000), target_shape=[5,5])
Gw_down = Gf(mesh=MeshReFreq(window = (-5.0,5.0), n_w=1000), target_shape=[5,5])
Gw_up.set_from_pade(Giw_from_dlr_up, n_points = 1000)
Gw_down.set_from_pade(Giw_from_dlr_down, n_points = 1000)

oplot(-Gw_up[0,0].imag/np.pi, linewidth=2, color=colorcycle[0], label='00_up')
oplot(-Gw_up[3,3].imag/np.pi, linewidth=2, color=colorcycle[1], label='11_up')
oplot(Gw_down[0,0].imag/np.pi, linewidth=2, color=colorcycle[0], label='00_down')
oplot(Gw_down[3,3].imag/np.pi, linewidth=2, color=colorcycle[1], label='11_down')
plt.legend()



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
#     n_samples        = 50,
#     sigma_bound      = U,
#     sigma_offset     = mu,
#     offdiag_fraction = 0.2,
# )

dm_proposals = DensityMatrixProposals(
    n_proposals       = 100,
    n_targeting_steps = 50,
    targeting_alpha   = 0.5,
    # targeting_alpha   = 2.0,
    half_occ_prob     = 0.2,
    force_real        = True,
    # Explicit Hund's m=2 targets — all three orbital permutations of [1,1,0]/[0,0,0].
    # These guarantee the magnetic fixed points are proposed regardless of random sampling.
    # They count toward n_proposals and are processed through the targeting loop.
    custom_proposals  = [
        {'up': np.diag([1]*norb), 'down': np.diag([0]*norb)},
        {'up': np.diag([1]*norb), 'down': np.diag([0]*norb)},
        # {'up': np.diag([0., 0.]), 'down': np.diag([1., 0.])},
        # {'up': np.diag([0., 0.]), 'down': np.diag([0., 1.])},
        # {'up': np.diag([1., 1.]), 'down': np.diag([0., 1.])},
    ],
    # custom_proposals  = [
    #     {'up': np.diag([1., 1., 0.]), 'down': np.diag([0., 0., 0.])},
    #     {'up': np.diag([1., 0., 1.]), 'down': np.diag([0., 0., 0.])},
    #     {'up': np.diag([0., 1., 1.]), 'down': np.diag([0., 0., 0.])},
    # ],
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

from triqs.gf import SemiCircular

ensemble.G0_iw['up'].data[:]   = G0_conv['up'].data
ensemble.G0_iw['down'].data[:] = G0_conv['down'].data
# ensemble.G0_iw['down'] << SemiCircular(2.0 * t)
# ensemble.G0_iw['down'] << inverse(inverse(ensemble.G0_iw['down']) - mu)

# ensemble.G0_iw['up'] << SemiCircular(2.0 * t)
# ensemble.G0_iw['up'] << inverse(inverse(ensemble.G0_iw['up']) - mu)

ensemble.solve(
    h_int              = h_int,
    proposal_generators = [dm_proposals],
    # proposal_generators = [sobol_proposals],
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



plot_dir = f'Plots_norb{norb}_U{U}_J{J}_cf{cf_mag}_beta{int(beta)}'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)




#%%
landscape_fig = ensemble.plot_landscape(
    show      = False,
)

landscape_fig.suptitle(f'U={U} eV, J = {J} eV, cf = {cf_mag} eV, beta = {beta} eV-1', fontsize=18)

axes = landscape_fig.axes

# get axes from figure
axes[0].set_title("Discovered stationary states", fontsize=16)

plt.savefig(f"{plot_dir}/saddle_point_landscape.jpg", dpi=300, bbox_inches='tight')

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

ensemble.beta_eff = 40


ens_dmft_result = dmft_loop_ensemble_hf(
    ensemble_solver = ensemble,
    dm_proposals    = dm_proposals,
    # dm_proposals    = sobol_proposals,
    h_int           = h_int,
    G0_iw           = G0_conv,        # start from PM bath
    t               = t,
    mu_init         = mu,
    n_target        = n_target,
    n_elec_total    = n_target,
    max_iter        = 20,
    eps             = 1e-3,
    mix             = 0.7,
    adjust_mu       = True,
    mu_bracket      = 55.0,
    crystal_field   = crystal_field,
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

# make a directory to save the plots if it doesn't exist
# encode norb, U, J, and beta, target n




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
# oplot(Gtau_ens_up[2,2].real, color=colorcycle[2], lw=2, label='ens DMFT up 1 1')


# print G_tau(beta/2) for a quick estimate of the density
with np.printoptions(precision=4, suppress=True):
    print('Ensemble DMFT G(β/2):')

    betahalf = int(len(Gtau_ens_up.mesh) // 2)
    print(f'  G_up(β/2): \n {Gtau_ens_up.data[betahalf].real}')
    print(f'  G_down(β/2): \n {Gtau_ens_down.data[betahalf].real}')
    # print(f'  G_down(β/2): \n {Gtau_ens_down.data[:, int(len(Gtau_ens_down.mesh) // 2)].real}')


plt.savefig(f'{plot_dir}/ensemble_DMFT_Gtau.jpg', dpi=200, bbox_inches='tight')



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
#%%

# make a nicer plot with both real and imaginary parts with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

# get ax 0
mesh = final_ens_Sigma['up'].mesh
freqs_array = np.array([ complex(iw) for iw in mesh.values()])
plt.sca(axes[0])
for iorb in range(1):
    # oplot(final_ens_Sigma_up_iw[iorb, iorb].real, color=colorcycle[iorb], linestyle="-",lw=2, label='ens up 0 0', x_window=(0, 20))
    plt.plot(freqs_array.imag,
            final_ens_Sigma['up'][iorb,iorb].data.real,
              color=colorcycle[iorb],
            linestyle="-",lw=2,
            marker='o', markersize=4, 
            label='ens up 0 0')
    # plt
plt.sca(axes[1])
for iorb in range(1):
    plt.plot(freqs_array.imag, final_ens_Sigma['up'][iorb,iorb].data.imag,
             color=colorcycle[iorb],
             linestyle="-", lw=2,
             marker='o', markersize=4,
             label='ens up 0 0')

axes[0].set_title('Real part of Σ(iω)', fontsize=14)
axes[1].set_title('Imaginary part of Σ(iω)', fontsize=14)
axes[0].set_xlabel(r'Matsubara $i\omega_n$', fontsize=13)
axes[1].set_xlabel(r'Matsubara $i\omega_n$', fontsize  =13)
axes[0].set_ylabel(r'$\mathrm{Re}\,\Sigma(i\omega_n)$', fontsize=13)
axes[1].set_ylabel(r'$\mathrm{Im}\,\Sigma(i\omega_n)$', fontsize=13)

axes[0].set_xlim([0, 20])
axes[0].set_ylim([20, 22])
axes[1].set_xlim([0, 20])
axes[1].set_ylim([-20, 0.1])






plt.ylim(bottom=-30)
plt.savefig(f'{plot_dir}/ensemble_Sigma_iw.jpg', dpi=300, bbox_inches='tight')




#%% pade analytically continue and plot G(w)

G_iw_freq_up = make_gf_imfreq(final_ens_G['up'], n_iw=n_iw_dense)
Gw_ens_up   = _pade_block(G_iw_freq_up)

G_iw_freq_down = make_gf_imfreq(final_ens_G['down'], n_iw=n_iw_dense)
Gw_ens_down   = _pade_block(G_iw_freq_down)



# oplot(-1/np.pi*Gw_ens_up[0,0].imag, color=colorcycle[0], lw=2, label='ens DMFT up 0 0')
# oplot(-1/np.pi*Gw_ens_up[1,1].imag, color=colorcycle[1], lw=2, label='ens DMFT up 1 1')
# # oplot(-1/np.pi*Gw_ens_up[2,2].imag, color=colorcycle[2], lw=2, label='ens DMFT up 1 1')
# oplot(+1/np.pi*Gw_ens_down[0,0].imag, color=colorcycle[0], lw=2, label='ens DMFT down 0 0')
# oplot(+1/np.pi*Gw_ens_down[1,1].imag, color=colorcycle[1], lw=2, label='ens DMFT down 1 1')


w_wals = np.array(list(Gw_ens_up.mesh.values())).real


for iorb in range(norb):
    plt.plot(w_vals, -1/np.pi*Gw_ens_up[iorb, iorb].data.imag,
             color=colorcycle[iorb], lw=2, label=f'Up spin, orb {iorb}')
    plt.plot(w_vals, +1/np.pi*Gw_ens_down[iorb, iorb].data.imag,
             color=colorcycle[iorb], lw=2, label=f'Down spin, orb {iorb}', linestyle='--')

plt.title(
    f'Ensemble  spectral function\n  A(ω) = −Im G(ω)/π   |   U={U}   J={J}   β={beta}',
    fontsize=14, y=1.02
)
plt.xlabel(r'$\omega$', fontsize=13)
plt.ylabel(r'pDOS [ev$^{-1}$]', fontsize=13)
plt.legend(fontsize=10, ncol=1, bbox_to_anchor=(1.1, 1.05))

plt.ylim(top=0.5)


plt.savefig(f'{plot_dir}/ensemble_pdos.jpg', dpi=300, bbox_inches='tight')






with np.printoptions(precision=4, suppress=True):
    mup_w = Gw_ens_up.density()
    mdn_w = Gw_ens_down.density()
    print('Ensemble DMFT densities from G(w):')
    print(f'  n_up: \n {mup_w}')
    print(f'  n_down: \n {mdn_w}')

# %% debug here`
# sum up and down, trace over orbitals, and plot the total spectral function
Gw_ens_total = Gw_ens_up + Gw_ens_down
A_ens =  Gw_ens_total[0,0].copy()

for iorb in range(norb):
    A_ens = -1/np.pi * Gw_ens_total[iorb, iorb].imag


oplot(A_ens, color='purple', lw=2, label='Total A(ω)')
plt.title(
    f'Ensemble spectral function \n A(ω) = −Im G(ω)/π   |   U={U}   J={J}   β={beta}',   
    fontsize=14, y=1.02
)
plt.xlabel(r'$\omega$', fontsize=13)
plt.ylabel(r'$-\mathrm{Im}\,G(\omega)/\pi$', fontsize=13)
# plt.xlim(0, 20)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

plt.savefig(f'{plot_dir}/ensemble_Aw_tot.jpg', dpi=300, bbox_inches='tight')

plt.tight_layout()







# %%
# save the the final converged ensemble landscape

landscape_fig = ensemble.plot_landscape(
    show      = False,
)

landscape_fig.suptitle(f'Ensemble DMFT saddle-point landscape\n  U={U} eV, J = {J} eV, cf = {cf_mag} eV, beta = {beta} eV-1', fontsize=18)



# %%
