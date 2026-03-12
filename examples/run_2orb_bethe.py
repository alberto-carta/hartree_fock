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
t        = 1    # hopping;  semicircular bandwidth W = 2t = 2.0
beta     = 40.0   # inverse temperature  (T = 1/β = 0.025)
w_max    = 20.0   # DLR energy cutoff
eps_dlr  = 1e-12  # DLR accuracy

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

# spin_kick = {
#     'up':   4*np.diag([+1.0,+1.0]),   # both orbitals spin-up shifted by +1
#     'down': 4*np.diag([-1.0,-1.0]), # both orbitals spin-down shifted by -1
# }

# ── U scan ──────────────────────────────────────────────────────────────────
mpi.report('\n' + '='*70)
mpi.report('  2-orbital Bethe lattice HF-DMFT  |  t={:.2f}  β={:.1f}'.format(t, beta))
mpi.report('='*70 + '\n')

U_values = [5.5]

mu_guess = U_values[0]
spin_kick = {
    'up':   np.diag([mu_guess-10.0,mu_guess-10.0]),   # both orbitals spin-up shifted by +1
    'down': np.diag([mu_guess+10.0,mu_guess+10.0]), # both orbitals spin-down shifted by -1
}

results = {}
for U in U_values:
    mpi.report(f'\n── U = {U:.1f} ──────────────────────────────────────────')

    mu_init = U # particle-hole symmetric starting point for half filling

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
        mix        = 0.3,
        eps        = 1e-22,
        verbose    = True,
        adjust_mu  = True,
        mu_bracket = 25.0,
        # spin_kick  = spin_kick,  # set to None or 0.0 for paramagnetic start
        # ImpuritySolver.solve() kwargs:
        with_fock  = True,
        one_shot   = True,   # fully self-consistent HF within each DMFT step
        method     = 'hybr',
        tol        = 1e-15,
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
Gw_up = Gf(mesh=MeshReFreq(window = (-5.0,5.0), n_w=1000), target_shape=[2,2])
Gw_down = Gf(mesh=MeshReFreq(window = (-5.0,5.0), n_w=1000), target_shape=[2,2])
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

#%% Hybridization function from the FM solver
# Δ(iω) = iω + μ - G0^{-1}(iω),  computed on the DLR mesh then Padé-continued
_mu_fm = res['mu']
#define a block Delta_fm on the DLR mesh
Delta_fm = solver.G0_iw.copy()

Delta_fm['up'] << iOmega_n + _mu_fm - inverse(solver.G0_iw['up'])
Delta_fm['down'] << iOmega_n + _mu_fm - inverse(solver.G0_iw['down'])


Diw_fm_up   = make_gf_imfreq(Delta_fm['up'],   n_iw=2500)
Diw_fm_down = make_gf_imfreq(Delta_fm['down'], n_iw=2500)

Dw_fm_up   = Gf(mesh=MeshReFreq(window=(-10.0, 10.0), n_w=1000), target_shape=[2, 2])
Dw_fm_down = Gf(mesh=MeshReFreq(window=(-10.0, 10.0), n_w=1000), target_shape=[2, 2])
Dw_fm_up.set_from_pade(Diw_fm_up,   n_points=100)
Dw_fm_down.set_from_pade(Diw_fm_down, n_points=100)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
_w_d = np.array(list(Dw_fm_up.mesh.values())).real

ax = axes[0]
ax.plot(_w_d, -Dw_fm_up[0,0].data.imag / np.pi,   color=colorcycle[0], label='orb0 up')
ax.plot(_w_d, -Dw_fm_up[1,1].data.imag / np.pi,   color=colorcycle[1], label='orb1 up')
ax.plot(_w_d, -Dw_fm_down[0,0].data.imag / np.pi, color=colorcycle[0], linestyle='--', label='orb0 down')
ax.plot(_w_d, -Dw_fm_down[1,1].data.imag / np.pi, color=colorcycle[1], linestyle='--', label='orb1 down')
ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$-\mathrm{Im}\,\Delta(\omega)/\pi$')
ax.set_title(r'Hybridization $\Delta(\omega)$ — FM solver')
ax.legend()

ax = axes[1]
_iw_d = np.array([w.imag for w in Diw_fm_up.mesh])
ax.plot(_iw_d, Diw_fm_up[0,0].data.imag,   color=colorcycle[0], label='orb0 up')
ax.plot(_iw_d, Diw_fm_up[1,1].data.imag,   color=colorcycle[1], label='orb1 up')
ax.plot(_iw_d, Diw_fm_down[0,0].data.imag, color=colorcycle[0], linestyle='--', label='orb0 down')
ax.plot(_iw_d, Diw_fm_down[1,1].data.imag, color=colorcycle[1], linestyle='--', label='orb1 down')
ax.set_xlabel(r'$i\omega_n$')
ax.set_ylabel(r'$\mathrm{Im}\,\Delta(i\omega_n)$')
ax.set_title(r'Hybridization $\Delta(i\omega_n)$ — FM solver (Matsubara)')
ax.set_xlim(0, 20)
ax.legend()
plt.tight_layout()

#%% plot D * G

oplot((Dw_fm_up[0,0]*Gw_up[0,0]).imag, color=colorcycle[0], label='00 up')


# integrate this to fermi with simpson's rule and get the kinetic energy estimate from the FM solution




#%% estimate kinetic energy

# Tr[Delta * G]

F1 = 0.0
Delta = solver.G0_iw.copy()

for bl in ['up', 'down']:
    Giw_bl = solver.G_iw[bl]
    Delta[bl] << iOmega_n + res['mu'] - inverse(solver.G0_iw[bl])

    # make matsubara freq representation of Delta and G on the same DLR mesh, then sum tr[Δ(iω)·G(iω)] over all Matsubara frequencies
    # Giw_bl_from_dlr = make_gf_imfreq(Giw_bl, n_iw=2500)
    # Diw_bl_from_dlr = make_gf_imfreq(Delta[bl], n_iw=2500)

    # kinetic_contrib = Giw_bl_from_dlr * Diw_bl_from_dlr
    kinetic_contrib = Giw_bl *Delta[bl]

    F1 += np.trace(kinetic_contrib.density())

print(f"Estimated kinetic energy from the FM solution: F1 = {F1:.4f}")





















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
    'up':   np.diag([mu-a, mu-a]).astype(complex),
    'down': np.diag([mu+a, mu+a]).astype(complex),
}

reversed_spin_kick = {
    'up':   np.diag([mu+a, mu+a]).astype(complex),
    'down': np.diag([mu-a, mu-a]).astype(complex),
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
plt.xlim(-1, 20)

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
#%%
# ── Free energy contributions ─────────────────────────────────────────────────
# F ≈ F1 + F2  where
#   F1 = Σ_{bl,orb} Σ_HF[bl][orb,orb]² / U        (field / Stoner penalty)
#   F2 = -(1/π) (1/β) Σ_{iω,bl} tr[Δ_bl(iω) G_bl(iω)]  (effective kinetic)
#
# Δ(iω) = iω - μ - G⁻¹(iω)  built from the FULL interacting G (not G0/Weiss).
# The product Δ·G ~ 1/ω² so the Matsubara sum converges without a tail correction.

def _impurity_free_energy(G_iw_dlr, solver, Sigma_HF_dict, mu_val, U_val, n_iw=5000):
    """Return (F1, F2) for one set of impurity Green's functions.

    F1 = sum_{bl,orb} Sigma_HF[bl][orb,orb]^2 / U
    F2 = -(1/pi) * (1/beta) * sum_{iw,bl} tr[ Delta_bl(iw) * G_bl(iw) ]
         where Delta_bl(iw) = iOmega_n - mu - inverse(G_bl)
    """
    blocks = [bl for bl, _ in G_iw_dlr]

    # F1 ─────────────────────────────────────────────────────────────────────
    # F1 = sum(
    #     np.sum(np.diag(Sigma_HF_dict[bl].real) ** 2) / U_val
    #     for bl in blocks
    # )

    F1 = solver.interaction_energy() 

    #

    # F2: Δ built from full G with << syntax on dense Matsubara grid ──────────
    F2 = 0.0

    # this is roughly int_-inf^fermi (-1/pi w * imag()  )dw

    # we need to find a better way but for now we 
    # can just convert G to real frequencies via 
    # Pade and integrate there


    for bl in blocks:
        # get the real frequency G
        G_w = Gf(mesh=MeshReFreq(window=(-10.0, 10.0), n_w=1000), target_shape=[2, 2])
        G_w.set_from_pade(make_gf_imfreq(G_iw_dlr[bl], n_iw=n_iw), n_points=100)

        prod = G_w.copy()

        prod << Omega * G_w

        F2 += prod.total_density()




    # for bl in blocks:
    #     Delta_bl = G_iw_dlr[bl].copy()
    #     Delta_bl << iOmega_n - mu_val - inverse(G0_iw_dlr[bl]) + Sigma_HF_dict[bl]

    #     prod = Delta_bl * G_iw_dlr[bl]
    #     F2 += np.trace(prod.density())


    return F1, F2



# test the free energy contributions for the FM solution we already have
F1_FM, F2_FM = _impurity_free_energy(solver.G_iw, solver, solver.Sigma_HF, res['mu'], U)

print(f"FM solution: F1 = {F1_FM:.4f}, F2 = {F2_FM:.4f}, F_total = {F1_FM + F2_FM:.4f}")


#%% let's try to get the free energy without pade
from scipy.linalg import logm

# G_iw = make_gf_imfreq(solver.G_iw, n_iw=5000)
G_iw = solver.G_iw.copy()
# make a green function which is just iOmega_n 
iomega = G_iw.copy()
a = 1
iomega['up'] << iOmega_n+ 0
iomega['down'] << iOmega_n + 0

prodlog = iomega * G_iw
# prodlog = G_iw * G_iw

# take the log for every matsubara frequency
for bl in ['up', 'down']:
    for i in range(len(prodlog[bl].mesh)):
        prodlog[bl].data[i] = logm(prodlog[bl].data[i])


oplot(prodlog['up'][0,0], '-o', label='iOmega_n * G_up', x_window=(0, 20))

oplot((G_iw)['up'][0,0], '-o', label='iOmega_n * G_down', x_window=(0, 20))


estimated_F2 = prodlog.total_density() 

print(f"Estimated F2 from logm approach: F2 = {estimated_F2:.4f}")



#%% plot G_iw 


one_over_omega = G_iw.copy()

one_over_omega['up'] << 1 * inverse(iOmega_n)

# oplot(G_iw['up'][0,0], '-o', label='G_up', x_window=(5, 20))
# oplot(one_over_omega['up'][0,0], '-o', label='1/iOmega_n', x_window=(5, 20))


# plot the difference to see if G ~ 1/iOmega_n at high frequencies
diff = G_iw.copy()
diff['up'] <<   one_over_omega['up'] -G_iw['up']

# compare with 1/ omega_n^2 tail

one_over_omega_squared = G_iw.copy()
one_over_omega_squared['up'] << 1 * inverse(iOmega_n)
one_over_omega_squared['up'].data[:] = 0.02*  one_over_omega_squared['up'].data ** 2


oplot(diff['up'][0,0], '-o', label='G_up - 1/iOmega_n', x_window=(40, 80))
oplot(one_over_omega_squared['up'][0,0], '-o', label='1/iOmega_n^2', x_window=(40, 80))






# %%
# ── Averaged DMFT loop ───────────────────────────────────────────────────────
# At each iteration:
#   1. Run two one-shot HF solves with opposite spin kicks (+a, -a)
#   2. Average: G_avg = 0.5 * (G_up + G_down)  ← enforces spin symmetry
#   3. Extract the effective static Sigma:
#         Sigma_eff[bl] = avg over DLR pts of [G0^{-1}(iω) - G_avg^{-1}(iω)]
#      Since HF Sigma is static, this is just (Sigma_up + Sigma_down) / 2
#   4. Update G0 via Bethe: G0^{-1}(iω) = iω + μ - t² G_avg(iω)
#   5. Recompute G_avg via Dyson with the updated Sigma_eff
#   6. Check convergence on max|ΔΣ_eff|

t2_mat_avg = np.diag([t**2] * norb)   # same hopping for all orbitals

# ── Initialise from the previous converged FM solution ──────────────────────
# Start with the G0 of the last converged solver and a symmetric kick.
G0_avg = solver.G0_iw.copy()   # starting G0 from the FM run

mu = res['mu']
a = 2

kick_A = {
    'up':   np.diag([mu-a, mu-a]).astype(complex),
    'down': np.diag([mu+a, mu+a]).astype(complex),
}

kick_B = {
    'up':   np.diag([mu+a, mu+a]).astype(complex),
    'down': np.diag([mu-a, mu-a]).astype(complex),
}

kick_uniform = {
    'up':  np.diag([mu, mu]).astype(complex),
    'down': np.diag([mu, mu]).astype(complex),
}

mix_avg   = 0.1
max_iter_avg = 60
eps_avg   = 1e-20

# Sigma_eff is now a frequency-dependent DLR BlockGf, not a static matrix.
# Initialise to zero using the mesh from G0_avg.
Sigma_eff_old = G0_avg.copy()
for bl in ['up', 'down']:
    Sigma_eff_old[bl].data[:] = 0.0

# ── Step 1: two HF solves with opposite kicks ───────────────────────────
# Each solver is seeded from its own previous solution (or the initial kick
# on the first iteration) so they stay in their respective broken-symmetry
# basins independently.
solver0 = ImpuritySolver(gf_struct=gf_struct, beta=beta, w_max=w_max, eps=eps_dlr, dc='cFLL')
solverA = ImpuritySolver(gf_struct=gf_struct, beta=beta, w_max=w_max, eps=eps_dlr, dc='cFLL')
solverB = ImpuritySolver(gf_struct=gf_struct, beta=beta, w_max=w_max, eps=eps_dlr, dc='cFLL')





# Seed each solver from its own previous Sigma_HF kick
for bl in ['up', 'down']:
    solverA.Sigma_HF[bl] = kick_A[bl].copy()
    solverB.Sigma_HF[bl] = kick_B[bl].copy()
    solver0.Sigma_HF[bl] = kick_uniform[bl].copy()

for it in range(max_iter_avg):

    print(f"Entering iteration {it+1} of the averaged DMFT loop...")


    # Inject current ensemble G0 into both solvers
    for bl in ['up', 'down']:
        solver0.G0_iw[bl].data[:] = G0_avg[bl].data
        solverA.G0_iw[bl].data[:] = G0_avg[bl].data
        solverB.G0_iw[bl].data[:] = G0_avg[bl].data

    # Symmetrise solver0: average up and down so it stays in the paramagnetic sector
    _sym_G0  = 0.5 * (solver0.G0_iw['up'].data  + solver0.G0_iw['down'].data)
    _sym_Sig = 0.5 * (solver0.Sigma_HF['up']     + solver0.Sigma_HF['down'])
    _sym_Giw = 0.5 * (solver0.G_iw['up'].data   + solver0.G_iw['down'].data)
    for bl in ['up', 'down']:
        solver0.G0_iw[bl].data[:] = _sym_G0
        solver0.Sigma_HF[bl]      = _sym_Sig.copy()
        solver0.G_iw[bl].data[:]  = _sym_Giw

    print("\n" + "="*60)
    print(" Solving 0: uniform kick (reference) ")
    _sig0_old = {bl: solver0.Sigma_HF[bl].copy() for bl in ['up', 'down']}
    solver0.solve(h_int, with_fock=True, one_shot=False, method='hybr', tol=1e-8)
    for bl in ['up', 'down']:
        new_sig = (mix_avg * solver0.Sigma_HF[bl] + (1 - mix_avg) * _sig0_old[bl]).real.astype(complex)
        solver0.Sigma_HF[bl] = new_sig

    print(" Solving A: kick_A ")
    _sigA_old = {bl: solverA.Sigma_HF[bl].copy() for bl in ['up', 'down']}
    solverA.solve(h_int, with_fock=True, one_shot=False, method='hybr', tol=1e-8)
    for bl in ['up', 'down']:
        new_sig = (mix_avg * solverA.Sigma_HF[bl] + (1 - mix_avg) * _sigA_old[bl]).real.astype(complex)
        solverA.Sigma_HF[bl] = new_sig

    # Seed solverB from the spin-flip of solverA's converged solution so B
    # always starts as the mirror image of A (up ↔ down).
    solverB.Sigma_HF['up']      = solverA.Sigma_HF['down'].copy()
    solverB.Sigma_HF['down']    = solverA.Sigma_HF['up'].copy()
    solverB.G0_iw['up'].data[:] = solverA.G0_iw['down'].data
    solverB.G0_iw['down'].data[:] = solverA.G0_iw['up'].data
    solverB.G_iw['up'].data[:]  = solverA.G_iw['down'].data
    solverB.G_iw['down'].data[:] = solverA.G_iw['up'].data

    print(" Solving B: spin-flip of A ")
    _sigB_old = {bl: solverB.Sigma_HF[bl].copy() for bl in ['up', 'down']}
    solverB.solve(h_int, with_fock=True, one_shot=False, method='hybr', tol=1e-8)
    for bl in ['up', 'down']:
        new_sig = (mix_avg * solverB.Sigma_HF[bl] + (1 - mix_avg) * _sigB_old[bl]).real.astype(complex)
        solverB.Sigma_HF[bl] = new_sig


    # compute free energy contributions 

    f1_0, f2_0 = _impurity_free_energy(solver0.G_iw, solver0, solver0.Sigma_HF, mu, U)
    f1_A, f2_A = _impurity_free_energy(solverA.G_iw, solverA, solverA.Sigma_HF, mu, U)
    f1_B, f2_B = _impurity_free_energy(solverB.G_iw, solverB, solverB.Sigma_HF, mu, U)
    
    f0 = f1_0 + f2_0
    fA = f1_A + f2_A
    fB = f1_B + f2_B

    # A and B are exact spin-flip partners so they must be degenerate.
    # Symmetrize to remove numerical noise from the Padé-based F2:
    fAB = 0.5 * (fA + fB)
    fA  = fAB
    fB  = fAB

    # rescale by the minimum to avoid overflow in the exp() and get relative weights
    f_min = min(f0, fA, fB)
    f0 -= f_min
    fA -= f_min
    fB -= f_min

    betaeff = 40

    f0 *= betaeff
    fA *= betaeff
    fB *= betaeff



    partition = np.exp(-f0) + np.exp(-fA) + np.exp(-fB)
    boltz_weight0 = np.exp(-f0) / partition
    boltz_weightA = np.exp(-fA) / partition
    boltz_weightB = np.exp(-fB) / partition

    # ── Step 2: ensemble-average the two GFs ────────────────────────────────
    G_avg = solverA.G_iw.copy()
    for bl in ['up', 'down']:
        # G_avg[bl].data[:] = 0.5 * (solverA.G_iw[bl].data + solverB.G_iw[bl].data)
        G_avg[bl].data[:] = (
            boltz_weight0 * solver0.G_iw[bl].data + 
            boltz_weightA * solverA.G_iw[bl].data +
            boltz_weightB * solverB.G_iw[bl].data
        )

    # ── Step 3: effective self-energy from the ensemble via Dyson ────────────
    # Sigma_eff(iω) = G0_avg^{-1}(iω) - G_avg^{-1}(iω)
    # This is frequency-dependent and lives on the same DLR mesh as G0_avg.
    Sigma_eff_new = G0_avg.copy()
    n_pts = G0_avg['up'].data.shape[0]
    for bl in ['up', 'down']:
        for i in range(n_pts):
            G0_val  = G0_avg[bl].data[i]
            G_av_val = G_avg[bl].data[i]
            Sigma_eff_new[bl].data[i] = np.linalg.inv(G0_val) - np.linalg.inv(G_av_val)

    # ── Step 4: mix Sigma_eff ────────────────────────────────────────────────
    if it > 0:
        for bl in ['up', 'down']:
            Sigma_eff_new[bl].data[:] = (
                mix_avg       * Sigma_eff_new[bl].data
                + (1-mix_avg) * Sigma_eff_old[bl].data
            )

    # ── Step 5: update G0 via Bethe on G_avg ─────────────────────────────────
    # G0_new^{-1}(iω) = iω + μ - t² G_avg(iω)
    mesh_pts = list(G0_avg['up'].mesh)
    for bl in ['up', 'down']:
        for i, iw in enumerate(mesh_pts):
            iw_val = complex(iw)
            G_avg_val = G_avg[bl].data[i]
            G0_inv = (iw_val + mu) * np.eye(norb) - t2_mat_avg @ G_avg_val
            G0_avg[bl].data[i] = np.linalg.inv(G0_inv)

    # ── Step 6: convergence check on Sigma_eff ───────────────────────────────
    max_diff = max(
        np.max(np.abs(Sigma_eff_new[bl].data - Sigma_eff_old[bl].data))
        for bl in ['up', 'down']
    )
    for bl in ['up', 'down']:
        Sigma_eff_old[bl].data[:] = Sigma_eff_new[bl].data

    # diagnostics
    n_avg = {bl: G_avg[bl].density().real for bl in ['up', 'down']}
    n_str = ", ".join(f"{n_avg['up'][a,a]:.4f}" for a in range(norb))
    # show Sigma_eff at the first Matsubara point (lowest frequency) as a proxy
    sig_proxy = Sigma_eff_new['up'].data[0].real.diagonal()
    s_str = ", ".join(f"{v:.4f}" for v in sig_proxy)
    print(f"avg-DMFT {it+1:4d}: n_up=[{n_str}]  Σ_eff_up(iω₀)=diag([{s_str}])  dΣ={max_diff:.2e}")

    if max_diff < eps_avg and it > 0:
        print(f"  Averaged DMFT converged after {it+1} iterations.")
        break

# Print total impurity moment

print("\n" + "="*60)
print("  Ensemble-averaged impurity occupation and moment:")
n_up_avg = G_avg['up'].density().real
n_down_avg = G_avg['down'].density().real
m_avg = 0.5 * (n_up_avg - n_down_avg)
for a in range(norb):
    print(f"  Orbital {a}: n_up = {n_up_avg[a,a]:.4f}, n_down = {n_down_avg[a,a]:.4f}, m = {m_avg[a,a]:.4f}")   

print("Total: n_up = {:.4f}, n_down = {:.4f}, m = {:.4f}".format(
    n_up_avg.trace(), n_down_avg.trace(), m_avg.trace()
))
#%%


print("\n" + "="*60)
print("  Free energy contributions  F = F1 + F2")
print("  F1 = Σ Σ_HF² / U          (field / Stoner term)")
print("  F2 = -(1/π) Tr[Δ·G]       (effective kinetic term)")
print("="*60)

_sig_avg = {bl: 0.5 * (solverA.Sigma_HF[bl] + solverB.Sigma_HF[bl]) for bl in ['up', 'down']}



_cases = [
    ("FM solver (original)", solver.G_iw, solver,  solver.Sigma_HF,  mu),
    ("FM solver (new bath)", solver0.G_iw, solver0, solver0.Sigma_HF,  mu),
    ("Ensemble solverA",     solverA.G_iw, solverA, solverA.Sigma_HF, mu),
    ("Ensemble solverB",     solverB.G_iw, solverB, solverB.Sigma_HF, mu),
]

_results_fe = {}
for _label, _Giw, _solver, _Sig, _mu in _cases:
    _F1, _F2 = _impurity_free_energy(_Giw,_solver , _Sig, _mu, U+0.001)
    _results_fe[_label] = (_F1, _F2)
    print(f"\n  {_label}")
    print(f"    F1 (Σ²/U)          = {_F1:+.6f}")
    print(f"    F2 (-(1/π)Tr[ΔG]) = {_F2:+.6f}")
    print(f"    F_total            = {_F1+_F2:+.6f}")

print("\n" + "="*60)
#%%
# ── Visualise the averaged spectral function and G0 ───────────────────
Giw_avg_up   = make_gf_imfreq(G_avg['up'],   n_iw=2500)
Giw_avg_down = make_gf_imfreq(G_avg['down'], n_iw=2500)
G0iw_up      = make_gf_imfreq(G0_avg['up'],   n_iw=2500)
G0iw_down    = make_gf_imfreq(G0_avg['down'], n_iw=2500)

Siw_eff_up   = make_gf_imfreq(Sigma_eff_new['up'],   n_iw=2500)
Siw_eff_down = make_gf_imfreq(Sigma_eff_new['down'], n_iw=2500)

Gw_avg_up   = Gf(mesh=MeshReFreq(window=(-10.0, 10.0), n_w=1000), target_shape=[2, 2])
Gw_avg_down = Gf(mesh=MeshReFreq(window=(-10.0, 10.0), n_w=1000), target_shape=[2, 2])
G0w_avg_up   = Gf(mesh=MeshReFreq(window=(-10.0, 10.0), n_w=1000), target_shape=[2, 2])
G0w_avg_down = Gf(mesh=MeshReFreq(window=(-10.0, 10.0), n_w=1000), target_shape=[2, 2])
Gw_avg_up.set_from_pade(Giw_avg_up,   n_points=100)
Gw_avg_down.set_from_pade(Giw_avg_down, n_points=100)
G0w_avg_up.set_from_pade(G0iw_up,   n_points=100)
G0w_avg_down.set_from_pade(G0iw_down, n_points=100)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

w_vals = np.array(list(Gw_avg_up.mesh.values())).real

ax = axes[0]
ax.plot(w_vals, -Gw_avg_up[0,0].data.imag / np.pi,
        color=colorcycle[0], label='orb0 up')
ax.plot(w_vals, -Gw_avg_up[1,1].data.imag / np.pi,
        color=colorcycle[1], label='orb1 up')
ax.plot(w_vals,  Gw_avg_down[0,0].data.imag / np.pi,
        color=colorcycle[0], linestyle='--', label='orb0 down')
ax.plot(w_vals,  Gw_avg_down[1,1].data.imag / np.pi,
        color=colorcycle[1], linestyle='--', label='orb1 down')

# plot total spectral function as a reference
Gw_total = Gf(mesh=MeshReFreq(window=(-10.0, 10.0), n_w=1000), target_shape=[2, 2])
Gw_total.set_from_pade(make_gf_imfreq(G_avg['up'] + G_avg['down'], n_iw=2500), n_points=100)
ax.plot(w_vals, -Gw_total[0,0].data.imag / np.pi, color='k', linestyle='--', label='total')


ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$A(\omega)$')
ax.set_title('Ensemble-averaged spectral function')
ax.legend()

ax = axes[1]
ax.plot(w_vals, -G0w_avg_up[0,0].data.imag / np.pi,
        color=colorcycle[0], label='orb0 up')
ax.plot(w_vals, -G0w_avg_up[1,1].data.imag / np.pi,
        color=colorcycle[1], label='orb1 up')
ax.plot(w_vals,  G0w_avg_down[0,0].data.imag / np.pi,
        color=colorcycle[0], linestyle='--', label='orb0 down')
ax.plot(w_vals,  G0w_avg_down[1,1].data.imag / np.pi,
        color=colorcycle[1], linestyle='--', label='orb1 down')
ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$A_0(\omega)$')
ax.set_title('Ensemble-averaged G0 spectral function')
ax.legend()
plt.tight_layout()


#%% plot G and G0 of tau
Gt_avg_up   = make_gf_imtime(G_avg['up'],   n_tau=1000)
Gt_avg_down = make_gf_imtime(G_avg['down'], n_tau=1000)
G0t_avg_up   = make_gf_imtime(G0_avg['up'],   n_tau=1000)
G0t_avg_down = make_gf_imtime(G0_avg['down'], n_tau=1000)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
tau_vals = np.array(list(Gt_avg_up.mesh.values()))
ax = axes[0]
ax.plot(tau_vals, Gt_avg_up[0,0].data.real,
        color=colorcycle[0], label='orb0 up')
ax.plot(tau_vals, Gt_avg_up[1,1].data.real,
        color=colorcycle[1], label='orb1 up')
ax.plot(tau_vals, Gt_avg_down[0,0].data.real,
        color=colorcycle[0], linestyle='--', label='orb0 down')
ax.plot(tau_vals, Gt_avg_down[1,1].data.real,
        color=colorcycle[1], linestyle='--', label='orb1 down')
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$G(\tau)$')
ax.set_title('Ensemble-averaged Green\'s function in imaginary time')
ax.legend()

ax = axes[1]
ax.plot(tau_vals, G0t_avg_up[0,0].data.real,
        color=colorcycle[0], label='orb0 up')
ax.plot(tau_vals, G0t_avg_up[1,1].data.real,
        color=colorcycle[1], label='orb1 up')
ax.plot(tau_vals, G0t_avg_down[0,0].data.real,
        color=colorcycle[0], linestyle='--', label='orb0 down')
ax.plot(tau_vals, G0t_avg_down[1,1].data.real,
        color=colorcycle[1], linestyle='--', label='orb1 down')
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$G_0(\tau)$')
ax.set_title('Ensemble-averaged G0 in imaginary time')
ax.legend()
# %%

# plot the solver0 in real w
Gw0_up = Gf(mesh=MeshReFreq(window=(-20.0, 20.0), n_w=1000), target_shape=[2, 2])
Gw0_down = Gf(mesh=MeshReFreq(window=(-20.0, 20.0), n_w=1000), target_shape=[2, 2])
Gw0_up.set_from_pade(make_gf_imfreq(solver0.G_iw['up'], n_iw=2500), n_points=40)
Gw0_down.set_from_pade(make_gf_imfreq(solver0.G_iw['down'], n_iw=2500), n_points=40)


# this loses a lot of weight for some reason

oplot(Gw0_up[0,0].imag)



# %%
# ── Hybridization function Δ(iω) = iω + μ - G0^{-1}(iω) ─────────────────────
# Compute on the DLR mesh, then analytically continue via Padé.
Delta_avg = G0_avg.copy()
mesh_pts_delta = list(G0_avg['up'].mesh)
for bl in ['up', 'down']:
    for i, iw in enumerate(mesh_pts_delta):
        iw_val = complex(iw)
        G0_val = G0_avg[bl].data[i]
        Delta_avg[bl].data[i] = (iw_val + mu) * np.eye(norb) - np.linalg.inv(G0_val)

# Convert to dense Matsubara mesh for Padé
Diw_avg_up   = make_gf_imfreq(Delta_avg['up'],   n_iw=2500)
Diw_avg_down = make_gf_imfreq(Delta_avg['down'], n_iw=2500)

# Padé continuation
Dw_avg_up   = Gf(mesh=MeshReFreq(window=(-10.0, 10.0), n_w=1000), target_shape=[2, 2])
Dw_avg_down = Gf(mesh=MeshReFreq(window=(-10.0, 10.0), n_w=1000), target_shape=[2, 2])
Dw_avg_up.set_from_pade(Diw_avg_up,   n_points=100)
Dw_avg_down.set_from_pade(Diw_avg_down, n_points=100)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
w_vals_d = np.array(list(Dw_avg_up.mesh.values())).real

ax = axes[0]
ax.plot(w_vals_d, -Dw_avg_up[0,0].data.imag / np.pi,
        color=colorcycle[0], label='orb0 up')
ax.plot(w_vals_d, -Dw_avg_up[1,1].data.imag / np.pi,
        color=colorcycle[1], label='orb1 up')
ax.plot(w_vals_d, -Dw_avg_down[0,0].data.imag / np.pi,
        color=colorcycle[0], linestyle='--', label='orb0 down')
ax.plot(w_vals_d, -Dw_avg_down[1,1].data.imag / np.pi,
        color=colorcycle[1], linestyle='--', label='orb1 down')
ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$-\mathrm{Im}\,\Delta(\omega)/\pi$')
ax.set_title(r'Hybridization function $\Delta(\omega)$ (real freq)')
ax.legend()

ax = axes[1]
iw_vals_d = np.array([w.imag for w in Diw_avg_up.mesh])
ax.plot(iw_vals_d, Diw_avg_up[0,0].data.imag,
        color=colorcycle[0], label='orb0 up')
ax.plot(iw_vals_d, Diw_avg_up[1,1].data.imag,
        color=colorcycle[1], label='orb1 up')
ax.plot(iw_vals_d, Diw_avg_down[0,0].data.imag,
        color=colorcycle[0], linestyle='--', label='orb0 down')
ax.plot(iw_vals_d, Diw_avg_down[1,1].data.imag,
        color=colorcycle[1], linestyle='--', label='orb1 down')
ax.set_xlabel(r'$i\omega_n$')
ax.set_ylabel(r'$\mathrm{Im}\,\Delta(i\omega_n)$')
ax.set_title(r'Hybridization function $\Delta(i\omega_n)$ (Matsubara)')
ax.set_xlim(0, 20)
ax.legend()
plt.tight_layout()


#%% visualize solverA and B solutions on the real frequency axis
Gaw_A_up   = Gf(mesh=MeshReFreq(window=(-10.0, 10.0), n_w=1000), target_shape=[2, 2])
Gaw_B_up   = Gf(mesh=MeshReFreq(window=(-10.0, 10.0), n_w=1000), target_shape=[2, 2])
Gaw_A_up.set_from_pade(make_gf_imfreq(solverA.G_iw['up'], n_iw=2500), n_points=100)
Gaw_B_up.set_from_pade(make_gf_imfreq(solverB.G_iw['up'], n_iw=2500), n_points=100)



# get hybridization function from HF solution

D_hf_up = G0_avg.copy()['up']

D_hf_up << iOmega_n - inverse(solverA.G0_iw['up'])

D_hf_w = Gf(mesh=MeshReFreq(window=(-10.0, 10.0), n_w=1000), target_shape=[2, 2])
D_hf_w.set_from_pade(make_gf_imfreq(D_hf_up, n_iw=2500), n_points=100)

# oplot(-Gaw_A_up.imag, color=colorcycle[0])
oplot((D_hf_w).imag, color=colorcycle[0], linestyle='--', label='Δ·G A up')

# oplot(-Gaw_B_up.imag, color=colorcycle[1])























# %%

# %%
#  plot in real frequencies the ensamble hybridization function Δ(ω)

Delta_w_avg_up   = Gf(mesh=MeshReFreq(window=(-10.0, 10.0), n_w=1000), target_shape=[2, 2])
Delta_w_avg_down = Gf(mesh=MeshReFreq(window=(-10.0, 10.0), n_w=1000), target_shape=[2, 2])
Delta_w_avg_up.set_from_pade(Diw_avg_up,   n_points=100)
Delta_w_avg_down.set_from_pade(Diw_avg_down, n_points=100)

oplot(-Delta_w_avg_up[0,0].imag/np.pi, color=colorcycle[0], label='orb0 up')
oplot(-Delta_w_avg_up[1,1].imag/np.pi, color=colorcycle[1], label='orb1 up')
#%%

# now plot the product Δ(ω) * G(ω) on the real frequency axis
# for case A, B and the original solution

Gaw_A_up   = Gf(mesh=MeshReFreq(window=(-10.0, 10.0), n_w=1000), target_shape=[2, 2])

Gaw_B_up   = Gf(mesh=MeshReFreq(window=(-10.0, 10.0), n_w=1000), target_shape=[2, 2])
Gaw_A_up.set_from_pade(make_gf_imfreq(solverA.G_iw['up'], n_iw=2500), n_points=100)
Gaw_B_up.set_from_pade(make_gf_imfreq(solverB.G_iw['up'], n_iw=2500), n_points=100)

prod_A = Delta_w_avg_up * Gaw_A_up

oplot(prod_A, color=colorcycle[0], label='Δ·G orb0 up (case A)')



#%% manually construct delta

bl = 'down'

G0_dlr = G0_avg[bl]
G_iw_dlr = solver.G_iw
Sigma = solver.Sigma_HF[bl]
G0_iw_dlr = G0_avg
mu_val = mu

Delta_bl = G_iw_dlr[bl].copy()
Delta_bl << iOmega_n - mu_val - inverse(G0_dlr) + Sigma


Delta_w = Gf(mesh=MeshReFreq(window=(-10.0, 10.0), n_w=1000), target_shape=[2, 2])

Delta_w.set_from_pade(make_gf_imfreq(solverA.G_iw[bl], n_iw=2500), n_points=100)

oplot(Delta_w)