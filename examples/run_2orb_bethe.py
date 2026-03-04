#%%
import os
import sys

# ── Path setup ──────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'python'))  # make triqs_hartree_fock importable (dev)

import numpy as np
import triqs.utility.mpi as mpi

from triqs_hartree_fock import ImpuritySolver
from dmft_driver import dmft_loop_bethe_hf, make_h_int_hubbard, make_h_int_kanamori_simple

# ── Model parameters ────────────────────────────────────────────────────────
t        = 1.0    # hopping;  semicircular bandwidth W = 2t = 2.0
beta     = 40.0   # inverse temperature  (T = 1/β = 0.025)
w_max    = 20.0   # DLR energy cutoff
eps_dlr  = 1e-18  # DLR accuracy

n_target = 2.0    # half filling (2 electrons in 2 orbitals)
norb     = 2

gf_struct = [('up', norb), ('down', norb)]

# ── Optional spin kick to seed symmetry breaking ────────────────────────────
# Provide a dict with one (norb × norb) matrix per spin block that is used
# directly as the initial Sigma_HF guess before the first DMFT iteration.
# Both spin AND orbital channels can be polarised independently.
#
# Examples:
#   spin_kick = None                          → paramagnetic start
#   spin_kick = 1.0                           → scalar shorthand: ±1·I
#   spin_kick = {'up': np.diag([1.0, 0.5]),   → orbital-resolved ferro kick
#                'down': np.diag([-1.0, -0.5])}
#
# Here we use a uniform ferromagnetic kick (same shift on both orbitals):
spin_kick = {
    'up':   2*np.diag([+1.0, +1.0]),   # both orbitals spin-up shifted by +1
    'down': 2*np.diag([+1.0,+1.0]), # both orbitals spin-down shifted by -1
}

# ── U scan ──────────────────────────────────────────────────────────────────
mpi.report('\n' + '='*70)
mpi.report('  2-orbital Bethe lattice HF-DMFT  |  t={:.2f}  β={:.1f}'.format(t, beta))
mpi.report('='*70 + '\n')

U_values = [6.0]

results = {}
for U in U_values:
    mpi.report(f'\n── U = {U:.1f} ──────────────────────────────────────────')

    mu_init = 0 # particle-hole symmetric starting point for half filling

    h_int = make_h_int_kanamori_simple(U, U, 1.0, 1.0, norb=norb)

    solver = ImpuritySolver(
        gf_struct  = gf_struct,
        beta       = beta,
        w_max      = w_max,
        eps        = eps_dlr,
        dc         = 'cFLL',    # no double counting for simple Hubbard model
    )

    res = dmft_loop_bethe_hf(
        solver     = solver,
        t          = t,
        h_int      = h_int,
        mu_init    = mu_init,
        n_target   = n_target,
        max_iter   = 100,
        mix        = 1,
        eps        = 1e-7,
        verbose    = True,
        adjust_mu  = True,
        mu_bracket = 10.0,
        spin_kick  = spin_kick,  # set to None or 0.0 for paramagnetic start
        # ImpuritySolver.solve() kwargs:
        with_fock  = True,
        one_shot   = False,   # fully self-consistent HF within each DMFT step
        method     = 'lm',
        tol        = 1e-8,
    )

    results[U] = res

    # Report spin polarisation  m = (n_up - n_down) / 2
    n_up   = res['density']['up'].diagonal().real
    n_down = res['density']['down'].diagonal().real
    m      = (n_up - n_down) / 2.0

    mpi.report(
        f'  U={U:.1f}  μ={res["mu"]:+.4f}  '
        f'n_up={n_up}  n_down={n_down}  '
        f'm={m}  '
        f'converged={res["converged"]}  '
        f'n_iter={res["n_iter"]}'
    )

#%%
# ── Summary table ────────────────────────────────────────────────────────────
mpi.report('\n' + '='*70)
mpi.report('  SUMMARY   (2-orbital Bethe HF-DMFT, β={:.0f}, kick={})'.format(beta, spin_kick))
mpi.report('='*70)
mpi.report(f'  {"U":>6}  {"μ":>8}  {"n_up":>20}  {"n_dn":>20}  {"m":>12}  {"conv":>6}  {"iter":>5}')
mpi.report('  ' + '-'*80)
for U, res in results.items():
    n_up   = res['density']['up'].diagonal().real
    n_down = res['density']['down'].diagonal().real
    m      = (n_up - n_down) / 2.0
    mpi.report(
        f'  {U:6.1f}  {res["mu"]:+8.4f}  '
        f'{str(np.round(n_up, 4)):>20}  {str(np.round(n_down, 4)):>20}  '
        f'{str(np.round(m, 4)):>12}  '
        f'{str(res["converged"]):>6}  {res["n_iter"]:5d}'
    )
mpi.report('='*70 + '\n')


# %%
import matplotlib.pyplot as plt
from triqs.gf import *
from triqs.plot.mpl_interface import oplot
solver.G_iw

oplot(solver.G_iw['up'][0,0], '-o', label='G_up')


# mesh_re = MeshReFreq(mesh=solver.G_iw.mesh)

# Get back Green's function on the full Matsubara mesh, no Fourier transform needed
Giw_from_dlr_up = make_gf_imfreq(solver.G_iw['up'], n_iw=2500)
Giw_from_dlr_down = make_gf_imfreq(solver.G_iw['down'], n_iw=2500)

oplot(Giw_from_dlr_up[0,0], '-', label='G_up from DLR')
oplot(Giw_from_dlr_down[0,0], '-', label='G_down from DLR')

plt.xlim(0, 20)

#%%

colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
# set tau mesh and analytically continue to real freq
Gw_up = Gf(mesh=MeshReFreq(window = (-10.0,10.0), n_w=1000), target_shape=[2,2])
Gw_down = Gf(mesh=MeshReFreq(window = (-10.0,10.0), n_w=1000), target_shape=[2,2])
Gw_up.set_from_pade(Giw_from_dlr_up, n_points = 1000)
Gw_down.set_from_pade(Giw_from_dlr_down, n_points = 1000)

oplot(-Gw_up[0,0].imag/np.pi, linewidth=2, color=colorcycle[0], label='00_up')
oplot(-Gw_up[1,1].imag/np.pi, linewidth=2, color=colorcycle[1], label='11_up')
oplot(Gw_down[0,0].imag/np.pi, linewidth=2, color=colorcycle[0], label='00_down')
oplot(Gw_down[1,1].imag/np.pi, linewidth=2, color=colorcycle[1], label='11_down')
plt.legend()


#%% plot G0

G0w_up = Gf(mesh=MeshReFreq(window = (-10.0,10.0), n_w=1000), target_shape=[2,2])
G0w_down = Gf(mesh=MeshReFreq(window = (-10.0,10.0), n_w=1000), target_shape=[2,2])
G0w_up.set_from_pade(make_gf_imfreq(solver.G0_iw['up'], n_iw=2500), n_points = 1000)
G0w_down.set_from_pade(make_gf_imfreq(solver.G0_iw['down'], n_iw=2500), n_points = 1000)
# G0
oplot(-G0w_up[0,0].imag/np.pi, linewidth=2, color=colorcycle[0], label='00_up')
oplot(-G0w_up[1,1].imag/np.pi, linewidth=2, color=colorcycle[1], label='11_up')
oplot(G0w_down[0,0].imag/np.pi, linewidth=2, color=colorcycle[0], label='00_down')
oplot(G0w_down[1,1].imag/np.pi, linewidth=2, color=colorcycle[1], label='11_down')
plt.legend()




#%%

# start from the previous solution and give now a different kick to see if we can converge to a different solution (e.g. AFM instead of FM)

solver1 = ImpuritySolver(
    gf_struct  = gf_struct,
    beta       = beta,  
    w_max      = w_max,
    eps        = eps_dlr,
    dc         = 'cFLL',    # no double counting for simple Hubbard model
)

solver2 = ImpuritySolver(
    gf_struct  = gf_struct,
    beta       = beta,  
    w_max      = w_max,
    eps        = eps_dlr,
    dc         = 'cFLL',    # no double counting for simple Hubbard model
)

a = 2

mu = res['mu']

second_spin_kick = {
    'up':   np.diag([mu-a, mu-a]),   # orbital 0 spin-up shifted by +1, orbital 1 spin-up shifted by -1
    'down': np.diag([mu+a, mu+a]), # orbital 0 spin-down shifted by -1, orbital 1 spin-down shifted
}

reversed_spin_kick = {
    'up':   np.diag([mu+a, mu+a]),   # orbital 0 spin-up shifted by +1, orbital 1 spin-up shifted by -1
    'down': np.diag([mu-a, mu-a]), # orbital 0 spin-down shifted by -1, orbital 1 spin-down shifted
}

res_up = dmft_loop_bethe_hf(
    solver     = solver1,
    t          = t,
    h_int      = h_int,
    mu_init    = mu,
    n_target   = n_target,
    max_iter   = 1,
    mix        = 1,
    eps        = 1e-7,
    verbose    = True,
    adjust_mu  = False,
    mu_bracket = 10.0,
    spin_kick  = second_spin_kick,  # set to None or 0.0 for paramagnetic start
    # ImpuritySolver.solve() kwargs:
    with_fock  = True,
    one_shot   = False,   # fully self-consistent HF within each DMFT step
    method     = 'hybr',
    tol        = 1e-8,
)


res_down = dmft_loop_bethe_hf(
    solver     = solver2,
    t          = t,
    h_int      = h_int,
    mu_init    = mu,
    n_target   = n_target,
    max_iter   = 1,
    mix        = 1,
    eps        = 1e-7,
    verbose    = True,
    adjust_mu  = False,
    mu_bracket = 10.0,
    spin_kick  = reversed_spin_kick,  # set to None or 0.0 for paramagnetic start
    # ImpuritySolver.solve() kwargs:
    with_fock  = True,
    one_shot   = False,   # fully self-consistent HF within each DMFT step
    method     = 'hybr',
    tol        = 1e-8,
)

#%%
G_iw_up = res_up['G_loc_iw']
G_iw_down = res_down['G_loc_iw']

G_total = 1/2 * (G_iw_up + G_iw_down)


# effective self energy 

Sigma_eff = -inverse(G_total) + inverse(solver.G0_iw) 


oplot(Sigma_eff['up'], '-o', label='Sigma_eff_00')
plt.xlim(0, 20)

#%%

#visualize the new solution

Gw2_up = Gf(mesh=MeshReFreq(window = (-10.0,10.0), n_w=1000), target_shape=[2,2])
Gw2_down = Gf(mesh=MeshReFreq(window = (-10.0,10.0), n_w=1000), target_shape=[2,2])
Gw2_up.set_from_pade(make_gf_imfreq(solver1.G_iw['up'], n_iw=2500), n_points = 100)
Gw2_down.set_from_pade(make_gf_imfreq(solver1.G_iw['down'], n_iw=2500), n_points = 100) 

oplot(-Gw2_up[0,0].imag/np.pi, linewidth=2, color=colorcycle[0], label='00_up')
oplot(-Gw2_up[1,1].imag/np.pi, linewidth=2, color=colorcycle[1], label='11_up')
oplot(Gw2_down[0,0].imag/np.pi, linewidth=2, color=colorcycle[0], label='00_down')
oplot(Gw2_down[1,1].imag/np.pi, linewidth=2, color=colorcycle[1], label='11_down')
plt.legend()


# %%

Gw_total = Gf(mesh=MeshReFreq(window = (-10.0,10.0), n_w=1000), target_shape=[2,2])
Gw_total.set_from_pade(make_gf_imfreq(G_total['up'], n_iw=2500), n_points = 100)
oplot(-Gw_total[0,0].imag/np.pi, linewidth=2, color=colorcycle[0], label='00_total')
oplot(-Gw_total[1,1].imag/np.pi, linewidth=2, color=colorcycle[1], label='11_total')
plt.legend()
