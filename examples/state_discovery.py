#%%
import os
import sys

# from run_2orb_bethe import G_iw

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
eps_dlr  = 1e-9  # DLR accuracy

n_target = 1.0    # half filling (2 electrons in 2 orbitals)
norb     = 1

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
# Here we use a uniform kick (same shift on both orbitals):
# spin_kick = {
#     'up':   8*np.diag([-1.0,-1.0]),   # both orbitals spin-up shifted by +1
#     'down': 8*np.diag([-1.0,+1.0]), # both orbitals spin-down shifted by -1
# }

# ── U scan ──────────────────────────────────────────────────────────────────
mpi.report('\n' + '='*70)
mpi.report('  2-orbital Bethe lattice HF-DMFT  |  t={:.2f}  β={:.1f}'.format(t, beta))
mpi.report('='*70 + '\n')

U_values = [6]

J = 1

results = {}
for U in U_values:
    mpi.report(f'\n── U = {U:.1f} ──────────────────────────────────────────')

    mu_init = U # particle-hole symmetric starting point for half filling

    h_int = make_h_int_kanamori_simple(U, U, J, J, norb=norb)

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
        eps        = 1e-22,
        verbose    = True,
        adjust_mu  = True,
        mu_bracket = 75.0,
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
    m      = (n_up - n_down) 

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

from scipy.linalg import logm


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


    # for bl in blocks:
    #     # get the real frequency G
    #     G_w = Gf(mesh=MeshReFreq(window=(-10.0, 10.0), n_w=1000), target_shape=[2, 2])
    #     G_w.set_from_pade(make_gf_imfreq(G_iw_dlr[bl], n_iw=n_iw), n_points=100)

    #     prod = G_w.copy()

    #     prod << Omega * G_w

    #     F2 += prod.total_density()
    

    from scipy.linalg import logm

    # Basically log(G) diverges so one splits the sum into a part that can be
    # computed analytically and a part that is well-behaved and can be computed numerically. The analytical part is the log of iOmega_n, which gives a contribution to the free energy that can be computed analytically. The numerical part is the log of G, which can be computed on the DLR mesh and then integrated over Matsubara frequencies.
    # you take out the omega part from G 
    # the free energy of that part is roughly 0 because it is evaluated at 0+
    # should be in principle (1-exp(-beta*0+)) at low T F of this is 0
    # for true temperature one should evaluate the entire crystal field here
    # and see what it evaluates to 
    # the rest of the sum is then log(iomega * G) which should be
    # resonably well-behaved and can be computed on the DLR mesh and then integrated over Matsubara frequencies.

    G_iw = solver.G_iw.copy()
    # make a green function which is just iOmega_n 
    iomega = G_iw.copy()
    iomega['up'] << iOmega_n+ 0
    iomega['down'] << iOmega_n + 0

    prodlog = iomega * G_iw

    # take the log for every matsubara frequency
    for bl in ['up', 'down']:
        for i in range(len(prodlog[bl].mesh)):
            prodlog[bl].data[i] = logm(prodlog[bl].data[i])




    F2 = prodlog.total_density() 



    return float(np.real(F1 + F2))

#%% Start here discovery
# ══════════════════════════════════════════════════════════════════════════════
#  State-discovery: Sobol-seeded HF sweeps on the converged bath G0
#
#  For each of N_DISCOVERY initial conditions (drawn from a low-discrepancy
#  Sobol sequence in the diagonal-Σ_HF space) we run a fully self-consistent
#  HF solve on the *fixed* converged bath.  No additional DMFT cycles.
#
#  Observables collected per solution:
#    n_total   – total charge (should stay ≈ n_target for physical solutions)
#    m_mean    – mean impurity magnetisation ⟨n↑⟩ − ⟨n↓⟩ over orbitals
#    F_int     – HF interaction energy = ⟨H_int⟩  (exact free-energy proxy
#                within HF; at fixed G0 differences are physically meaningful)
#    n_close_0 – number of spin-orbital occupancies below OCCU_THRESH
#    n_close_1 – number above 1 − OCCU_THRESH
#
#  Output: pandas DataFrame  +  scatter plot  m vs F_int
# ══════════════════════════════════════════════════════════════════════════════
from scipy.stats import qmc
import pandas as pd


# ── Discovery hyper-parameters ───────────────────────────────────────────────
N_DISCOVERY  = 300     # Sobol samples; power-of-2 keeps low-discrepancy ideal
OCCU_THRESH  = 0.10     # threshold to call an occupancy "close to 0 or 1"

# Σ_HF elements sampling bounds:
#   diagonal   : [−SIGMA_BOUND,        +SIGMA_BOUND       ]
#   off-diagonal: [−SIGMA_OFFDIAG_BOUND, +SIGMA_OFFDIAG_BOUND]  (toned down)
_U_disc            = U_values[-1]          # U from the last scan entry
SIGMA_OFFSET = res['mu']
SIGMA_BOUND        =  U/2
SIGMA_OFFDIAG_FRAC = 0.2                # off-diag range as fraction of diag range
SIGMA_OFFDIAG_BOUND = SIGMA_BOUND * SIGMA_OFFDIAG_FRAC

SIGMA_UPPER = SIGMA_OFFSET + SIGMA_BOUND
SIGMA_LOWER = SIGMA_OFFSET - SIGMA_BOUND

_BORDER = '═' * 72
_SEP    = '─' * 72

print(f"\n{_BORDER}")
print(f"  STATE DISCOVERY  ─  Sobol-seeded HF landscape")
print(f"  U = {_U_disc:.2f}  β = {beta:.0f}  n_orb = {norb}  N_samples = {N_DISCOVERY}")
print(f"  Σ_HF diag    ∈ [{SIGMA_LOWER:.2f}, {SIGMA_UPPER:.2f}]")
print(f"  Σ_HF off-diag∈ [−{SIGMA_OFFDIAG_BOUND:.2f}, +{SIGMA_OFFDIAG_BOUND:.2f}]  (×{SIGMA_OFFDIAG_FRAC} of diag)")
print(f"{_BORDER}\n")

# ── Helper: vector → real symmetric norb×norb matrix ─────────────────────────
# Parameter layout per spin block (n_per_block = norb*(norb+1)//2):
#   [d_0 … d_{norb-1},  t_{01}, t_{02}, … t_{(norb-1,norb-1)}]
# where d are diagonal and t are upper/lower triangle entries.
def _vec_to_symmat(v, nb):
    """Build a real symmetric nb×nb matrix from norb*(norb+1)//2 values."""
    m = np.zeros((nb, nb), dtype=float)
    for a in range(nb):
        m[a, a] = v[a]
    idx = nb
    for a in range(nb):
        for b in range(a + 1, nb):
            m[a, b] = v[idx]
            m[b, a] = v[idx]
            idx += 1
    return m

# ── Sobol sampling in full real-symmetric Σ_HF space ─────────────────────────
# n_per_block = norb*(norb+1)//2  (diagonal + upper triangle per spin block)
# d = 2 * n_per_block : [up block params | down block params]
n_per_block  = norb * (norb + 1) // 2
n_params     = 2 * n_per_block

# Per-element bounds: first norb entries per block → diagonal, rest → off-diagonal
_n_offdiag   = n_per_block - norb   # number of independent off-diag entries per block
_lo_block    = [SIGMA_LOWER] * norb + [-SIGMA_OFFDIAG_BOUND] * _n_offdiag
_hi_block    = [SIGMA_UPPER] * norb + [SIGMA_OFFDIAG_BOUND] * _n_offdiag
_lo_bounds   = _lo_block + _lo_block   # both spin blocks
_hi_bounds   = _hi_block + _hi_block

_sampler     = qmc.Sobol(d=n_params, scramble=True, seed=42)
_raw         = _sampler.random(N_DISCOVERY)          # in [0, 1]^d
sigma_samples = qmc.scale(_raw, l_bounds=_lo_bounds, u_bounds=_hi_bounds)
#%%
# ── Converged bath Green's function and interaction term ──────────────────────
G0_conv    = solver.G0_iw   # from the last DMFT run
h_int_disc = h_int          # same Hamiltonian

# ── Main sweep ────────────────────────────────────────────────────────────────
records     = []
n_converged = 0
n_failed    = 0

_progress_every = max(1, N_DISCOVERY // 8)   # print 8 progress updates

for i_s, sigma_flat in enumerate(sigma_samples):

    # ── Build initial Σ_HF (real symmetric matrix per spin block) ──────────────
    Sig_init = {
        'up'  : _vec_to_symmat(sigma_flat[:n_per_block],        norb),
        'down': _vec_to_symmat(sigma_flat[n_per_block:],        norb),
    }

    # ── Fresh solver; inject converged G0 and initial Σ_HF ───────────────────
    disc = ImpuritySolver(
        gf_struct = gf_struct,
        beta      = beta,
        w_max     = w_max,
        eps       = eps_dlr,
        dc        = 'cFLL',
    )
    for bl in ['up', 'down']:
        disc.G0_iw[bl].data[:] = G0_conv[bl].data
        disc.Sigma_HF[bl]      = Sig_init[bl].astype(complex)  # must be complex; flatten/unflatten assume complex dtype

    # ── One self-consistent HF solve + observable collection ─────────────────
    _last_status = 'pending'
    try:
        disc.dc_fixed_value = +1
        disc.solve(h_int_disc, with_fock=True, one_shot=False,
                   method='hybr', tol=1e-8)
                #    method='linearmixing', tol=1e-8)

        # ── Collect observables ───────────────────────────────────────────────
        rho = {bl: disc.G_iw[bl].density().real for bl in ['up', 'down']}

        # Total charge
        n_tot = float(sum(np.trace(rho[bl]) for bl in ['up', 'down']))

        # Per-orbital and mean magnetisation
        m_orb  = np.array([rho['up'][a, a] - rho['down'][a, a] for a in range(norb)])
        m_mean = float(np.sum(m_orb))

        # Occupancies close to integer values
        all_occ  = np.array([rho[bl][a, a] for bl in ['up', 'down'] for a in range(norb)])
        n_close0 = int(np.sum(all_occ <       OCCU_THRESH))
        n_close1 = int(np.sum(all_occ > 1.0 - OCCU_THRESH))

        # F_int = float(np.real(disc.interaction_energy()))
        F_int = _impurity_free_energy(disc.G_iw, disc, disc.Sigma_HF, res['mu'], _U_disc, n_iw=5000)


        # ── Build record ──────────────────────────────────────────────────────
        rec = dict(
            sample_id = i_s,
            n_total   = n_tot,
            m_mean    = m_mean,
            F_int     = F_int,
            n_close_0 = n_close0,
            n_close_1 = n_close1,
        )
        for a in range(norb):
            for bl, tag in [('up', 'u'), ('down', 'd')]:
                rec[f'n{tag}{a}']    = float(rho[bl][a, a])
                rec[f'Sig_{tag}{a}'] = float(np.real(disc.Sigma_HF[bl][a, a]))
            rec[f'm{a}'] = float(m_orb[a])

        records.append(rec)
        n_converged += 1
        _last_status = f"n={n_tot:.3f}  m={m_mean:+.4f}  F={F_int:.5f}"

    except Exception as exc:
        n_failed += 1
        if n_failed == 1:
            print(f"  [!] First failure at sample {i_s}: {type(exc).__name__}: {exc}")
        _last_status = f"FAILED ({type(exc).__name__})"

    # ── Progress bar ─────────────────────────────────────────────────────────
    if (i_s + 1) % _progress_every == 0 or (i_s + 1) == N_DISCOVERY:
        frac   = (i_s + 1) / N_DISCOVERY
        filled = int(frac * 20)
        bar    = '█' * filled + '░' * (20 - filled)
        print(f"  [{bar}]  {i_s+1:4d}/{N_DISCOVERY}  ✓ {n_converged}  ✗ {n_failed}"
              f"  │  last: {_last_status}")

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df_disc = pd.DataFrame(records)
if df_disc.empty:
    print(f"\n  [!] No solutions converged — df_disc is empty.")
    print(f"  Check the first-failure message above for diagnostics.")
else:
    df_disc.sort_values('F_int', inplace=True, ignore_index=True)

print(f"\n{_SEP}")
print(f"  {'DONE':^68}")
print(f"  Converged : {n_converged:4d}  │  Failed : {n_failed:4d}  │  Total : {N_DISCOVERY}")
print(f"{_SEP}")

# ── Descriptive statistics ────────────────────────────────────────────────────
if not df_disc.empty:
    _show_cols = ['n_total', 'm_mean', 'F_int', 'n_close_0', 'n_close_1']
    print(f"\n  Descriptive statistics:\n")
    print(df_disc[_show_cols].describe().round(5).to_string(index=True))

# ── Sorted top-20 table ───────────────────────────────────────────────────────
_orb_n_cols = ([f'nu{a}' for a in range(norb)] +
               [f'nd{a}' for a in range(norb)])
# build display-friendly column names
_rename = {}
for a in range(norb):
    _rename[f'nu{a}'] = f'nu{a}'
    _rename[f'nd{a}'] = f'nd{a}'

print(f"\n{_SEP}")
print(f"  Top-30 solutions by HF free energy  (★ = |m| > 0.05)\n")

_hdr = (f"  {'#':>4}  {'n_tot':>7}  {'m':>8}  {'F_int':>12}"
        + "".join(f"  {'nu'+str(a):>6}" for a in range(norb))
        + "".join(f"  {'nd'+str(a):>6}" for a in range(norb))
        + f"  {'c0':>4}  {'c1':>4}")
print(_hdr)
print(f"  {_SEP}")

for rank, row in enumerate(df_disc.head(30).itertuples(), start=1):
    star   = '★' if abs(row.m_mean) > 0.05 else ' '
    n_cols = "".join(f"  {row._asdict().get(f'nu{a}', 0.0):6.4f}" for a in range(norb))
    d_cols = "".join(f"  {row._asdict().get(f'nd{a}', 0.0):6.4f}" for a in range(norb))
    print(f"  {rank:4d}  {row.n_total:7.4f}  {row.m_mean:+8.4f}  {row.F_int:12.6f}"
          f"{n_cols}{d_cols}"
          f"  {row.n_close_0:4d}  {row.n_close_1:4d}  {star}")

print(f"{_SEP}\n")
#%%
# set font size for plots
plt.rcParams.update({'font.size': 12})
fontsize_legend = 11
fontsize_label = 14
fontsize_title = 16

# ── Scatter plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
fig.suptitle(
    f"HF state landscape  ─  U={_U_disc:.1f}  β={beta:.0f}  n_orb={norb}"
    f"  │  {n_converged}/{N_DISCOVERY} converged",
    fontsize=13, y=1.01
)

# ── Left: m vs F_int (main scatter) ──────────────────────────────────────────
ax = axes[0]
_cm     = plt.get_cmap('RdBu_r')
_n_arr  = df_disc['n_total'].values
_vmin   = max(0, _n_arr.mean() - 2 * _n_arr.std())
_vmax   =        _n_arr.mean() + 2 * _n_arr.std()

sc = ax.scatter(
    df_disc['m_mean'], df_disc['F_int']-min(df_disc['F_int']),   # shift F_int so lowest is at 0 for better color scaling
    c          = df_disc['n_total'],
    cmap       = _cm,
    vmin       = _vmin,
    vmax       = _vmax,
    s          = 50,
    alpha      = 0.78,
    edgecolors = 'k',
    linewidths = 1,
    zorder     = 3,
)

# ax.set_ylim((-0.1,1))


cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label('Total charge  n', fontsize=10)

# Mark lowest-F solution
_best = df_disc.iloc[0]
ax.scatter([_best['m_mean']], [_best['F_int']-min(df_disc['F_int'])],
           marker='*', s=280, color='none', edgecolors='k', alpha=0.8,
           linewidths=0.8, zorder=5, label=f'lowest F  (m={_best["m_mean"]:+.3f})')

ax.axvline(0,  color='gray', lw=0.9, ls='--', alpha=0.55)
ax.set_xlabel(r'Impurity magnetisation  $m = \langle n_\uparrow\rangle - \langle n_\downarrow\rangle$',
              fontsize=fontsize_label)
ax.set_ylabel(r'Impurity free energy  $\langle F \rangle$',
              fontsize=fontsize_label)
ax.set_title('Discovered states', fontsize=fontsize_title)
ax.legend(fontsize=fontsize_legend)
ax.grid(True, alpha=0.3, lw=0.5)


# ── Right: histogram of m values ──────────────────────────────────────────────
ax = axes[1]
_m_arr = df_disc['m_mean'].values
_f_arr = df_disc['F_int'].values - df_disc['F_int'].min()  # shift F so lowest is at 0 for better color scaling
ax.hist(_f_arr, bins=50, color='steelblue', edgecolor='k', linewidth=0.35, alpha=0.82)
ax.axvline(0,              color='crimson',   lw=1.2, ls='--', label='PM  (m = 0)')
ax.axvline(_f_arr.mean(),  color='darkorange', lw=1.0, ls=':',
           label=f'mean  F={_f_arr.mean():+.3f}')
ax.set_xlabel(r'Impurity magnetisation  $m$', fontsize=fontsize_label)
ax.set_ylabel('Count', fontsize=fontsize_label)
ax.set_title('Distribution of converged magnetisations', fontsize=fontsize_title)
ax.legend(fontsize=fontsize_legend)
ax.grid(True, alpha=0.3, lw=0.5)

plt.tight_layout()


# save image to file

imagename = f"n{norb}_nelec{n_target}_U{U}_J{J}.jpg"

plt.savefig(imagename, dpi=300, bbox_inches='tight')
plt.show()


















# %%
