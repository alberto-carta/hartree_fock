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
eps_dlr  = 1e-15  # DLR accuracy

n_target = 1.0    # half filling (2 electrons in 2 orbitals)
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
    'up':   2*np.diag([+1.0,+1.0]),   # both orbitals spin-up shifted by +1
    'down': 2*np.diag([+1.0,+1.0]), # both orbitals spin-down shifted by -1
}

# ── U scan ──────────────────────────────────────────────────────────────────
mpi.report('\n' + '='*70)
mpi.report('  2-orbital Bethe lattice HF-DMFT  |  t={:.2f}  β={:.1f}'.format(t, beta))
mpi.report('='*70 + '\n')

U_values = [8.5]

results = {}
for U in U_values:
    mpi.report(f'\n── U = {U:.1f} ──────────────────────────────────────────')

    mu_init = 8 # particle-hole symmetric starting point for half filling

    h_int = make_h_int_kanamori_simple(U, U, 0.0, 0.0, norb=norb)

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
        mix        = 0.1,
        eps        = 1e-12,
        verbose    = True,
        adjust_mu  = True,
        mu_bracket = 15.0,
        spin_kick  = spin_kick,  # set to None or 0.0 for paramagnetic start
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
    'up':   np.diag([mu+a, mu-a]).astype(complex),
    'down': np.diag([mu+a, mu+a]).astype(complex),
}

kick_B = {
    'up':   np.diag([mu+a, mu+a]).astype(complex),
    'down': np.diag([mu+a, mu-a]).astype(complex),
}

mix_avg   = 0.1
max_iter_avg = 100
eps_avg   = 1e-7

# Sigma_eff is now a frequency-dependent DLR BlockGf, not a static matrix.
# Initialise to zero using the mesh from G0_avg.
Sigma_eff_old = G0_avg.copy()
for bl in ['up', 'down']:
    Sigma_eff_old[bl].data[:] = 0.0

# ── Step 1: two HF solves with opposite kicks ───────────────────────────
# Each solver is seeded from its own previous solution (or the initial kick
# on the first iteration) so they stay in their respective broken-symmetry
# basins independently.
solverA = ImpuritySolver(gf_struct=gf_struct, beta=beta, w_max=w_max, eps=eps_dlr, dc='cFLL')
solverB = ImpuritySolver(gf_struct=gf_struct, beta=beta, w_max=w_max, eps=eps_dlr, dc='cFLL')

# Seed each solver from its own previous Sigma_HF kick
for bl in ['up', 'down']:
    solverA.Sigma_HF[bl] = kick_A[bl].copy()
    solverB.Sigma_HF[bl] = kick_B[bl].copy()

for it in range(max_iter_avg):

    print(f"Entering iteration {it+1} of the averaged DMFT loop...")


    # Inject current ensemble G0 into both solvers
    for bl in ['up', 'down']:
        solverA.G0_iw[bl].data[:] = G0_avg[bl].data
        solverB.G0_iw[bl].data[:] = G0_avg[bl].data


    solverA.solve(h_int, with_fock=True, one_shot=False, method='hybr', tol=1e-8)
    solverB.solve(h_int, with_fock=True, one_shot=False, method='hybr', tol=1e-8)


    # ── Step 2: ensemble-average the two GFs ────────────────────────────────
    G_avg = solverA.G_iw.copy()
    for bl in ['up', 'down']:
        G_avg[bl].data[:] = 0.5 * (solverA.G_iw[bl].data + solverB.G_iw[bl].data)

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

print("\nFinal Sigma_eff at first Matsubara point:")
for bl in ['up', 'down']:
    print(f"  [{bl}] =\n{np.round(Sigma_eff_new[bl].data[0], 6)}")
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

oplot(Sigma_eff_new['up'], '-o', label='Sigma_eff_00')
plt.xlim(0, 2)