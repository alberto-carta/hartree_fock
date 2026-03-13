"""hf_dmft_landscape_backend.py
================================
Backend for HF-DMFT fixed-point landscape exploration on the Bethe lattice.

Two strategies for initial-condition kicks
-------------------------------------------
Sobol Σ_HF kicks
    Real-symmetric self-energy matrices sampled via a Sobol low-discrepancy
    sequence.  Applied as the starting Σ_HF; no constraint during the loop.
    The fixed point reached depends entirely on the basin of attraction.

DM-targeted kicks
    Each kick specifies a *target density matrix* per spin block.  For the
    first ``n_target_steps`` DMFT iterations, a Lagrange-multiplier correction

        Σ_HF[bl] ← Σ_HF[bl] + α · (ρ[bl] − ρ_target[bl])

    is added after each HF solve to steer the occupancies toward the target.
    After ``n_target_steps`` the constraint is released and the loop converges
    freely to whatever fixed point has been entered.

Public API
----------
    generate_sobol_kicks(n, norb, gf_struct, sigma_bound, sigma_offset,
                         offdiag_fraction, seed)      → list of kick dicts
    generate_dm_kicks(n, n_target, norb, gf_struct,
                      half_occ_prob, seed)             → list of kick dicts
    pm_kick(norb, gf_struct)                           → single kick dict
    run_landscape(kicks, solver_factory, t, h_int, mu_init, n_target,
                  crystal_field, archive_path, params_dict,
                  n_target_steps, targeting_alpha, print_every,
                  **dmft_kwargs)                       → n_converged (int)
    load_runs(path)                                    → list of run dicts
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import warnings

import numpy as np
from scipy.optimize import brentq
from h5 import HDFArchive

import triqs.utility.mpi as mpi
from triqs.gf import SemiCircular, inverse
from triqs.gf.meshes import MeshDLRImFreq

from triqs_hartree_fock.incoherent_ensemble_solver import (
    SobolSigmaProposals,
    _make_target_density_matrix,
    _suppress_hf_output as _suppress_solver_output,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Kick generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_sobol_kicks(n_kicks: int, norb: int, gf_struct: list,
                         sigma_bound: float, sigma_offset: float = 0.0,
                         offdiag_fraction: float = 0.2, seed: int = 0) -> list:
    """Generate *n_kicks* Sobol Σ_HF initial-condition kicks.

    Returns a list of dicts:
        ``{'type': 'sobol', 'sigma': {'up': array, 'down': array}}``

    Parameters
    ----------
    sigma_bound      : half-width of the diagonal sampling range
    sigma_offset     : centre of the diagonal range (e.g. set to μ or U/2)
    offdiag_fraction : off-diagonal range relative to sigma_bound
    """
    sampler = SobolSigmaProposals(
        n_samples        = n_kicks,
        sigma_bound      = sigma_bound,
        sigma_offset     = sigma_offset,
        offdiag_fraction = offdiag_fraction,
    )
    sigmas = sampler.generate(G0_iw=None, n_elec_total=0,
                              norb=norb, gf_struct=gf_struct, seed=seed)
    return [{'type': 'sobol', 'sigma': s} for s in sigmas]


def generate_dm_kicks(n_kicks: int, n_target: float, norb: int,
                      gf_struct: list, half_occ_prob: float = 0.10,
                      seed: int = 0, randomize_oxidation=False) -> list:
    """Generate *n_kicks* random DM-targeted kicks.

    Each kick specifies a target density matrix per spin block.  The
    eigenvalues of the target DM are drawn from {0, 0.5, 1} (integer-biased,
    ``half_occ_prob`` controls the 0.5 fraction) with a random SO(norb)
    rotation applied so the target is not aligned with the orbital basis.

    Returns a list of dicts:
        ``{'type': 'dm', 'rho_target': {'up': array, 'down': array},``
        ``'n_up': int, 'n_down': int}``
    """
    rng    = np.random.default_rng(seed)
    blocks = [bl for bl, _ in gf_struct]
    n_int  = int(round(n_target))
    n_up_min = max(0, n_int - norb)
    n_up_max = min(norb, n_int)

    kicks = []
    for _ in range(n_kicks):
        if randomize_oxidation:
            # pick [-1, 0, 1]
            delta_n = rng.choice([-1, 0, 1])
        else:
            delta_n = 0
        n_up   = int(rng.integers(n_up_min, n_up_max + 1+delta_n))
        n_down = n_int+delta_n - n_up
        rho_target = {
            'up':   _make_target_density_matrix(n_up,   norb, rng, half_occ_prob),
            'down': _make_target_density_matrix(n_down, norb, rng, half_occ_prob),
        }
        kicks.append({'type': 'dm', 'rho_target': rho_target,
                      'n_up': n_up, 'n_down': n_down})
    return kicks


def pm_kick(norb: int, gf_struct: list) -> dict:
    """Return a paramagnetic (zero Σ_HF) reference kick."""
    blocks = [bl for bl, _ in gf_struct]
    return {'type': 'sobol',
            'sigma': {bl: np.zeros((norb, norb)) for bl in blocks}}


# ══════════════════════════════════════════════════════════════════════════════
#  Pretty-print helpers
# ══════════════════════════════════════════════════════════════════════════════

_LW = 70  # default line-width for decorative rules


def _rule(char: str = '─', n: int = _LW) -> str:
    return char * n


def _vfmt(v: np.ndarray, w: int = 7, d: int = 4) -> str:
    """Format a 1-D array as a compact bracketed string."""
    return '[' + '  '.join(f'{x:{w}.{d}f}' for x in np.asarray(v).ravel()) + ']'


def _mat_lines(M: np.ndarray, name: str, lbl: str) -> list:
    """Return formatted lines for an NxN complex matrix, Re | Im per row."""
    n    = np.asarray(M).shape[0]
    lead = f'  {name:<5s} {lbl}  '
    cont = ' ' * len(lead)
    lines = []
    for i in range(n):
        pfx  = lead if i == 0 else cont
        re_s = _vfmt(M[i].real)
        im_s = _vfmt(M[i].imag)
        lines.append(f'{pfx}row {i}  {re_s}  │  {im_s}')
    return lines


def _print_run_header(i_run: int, n_total: int, kick: dict,
                      n_target_steps: int = 0,
                      targeting_alpha: float = 0.0) -> None:
    r = mpi.report
    r('\n' + _rule('═'))
    if kick['type'] == 'dm':
        n_up   = kick['n_up']
        n_down = kick['n_down']
        eig_up = np.sort(np.linalg.eigvalsh(kick['rho_target']['up']))[::-1]
        eig_dn = np.sort(np.linalg.eigvalsh(kick['rho_target']['down']))[::-1]
        r(f'  Run {i_run:3d}/{n_total}  │  DM-targeted kick'
          f'  (n_up_tgt={n_up}, n_dn_tgt={n_down})')
        r(f'  ρ_tgt  up  eigs = {_vfmt(eig_up)}')
        r(f'         dn  eigs = {_vfmt(eig_dn)}')
        r(f'  Targeting: {n_target_steps} DMFT steps  (α={targeting_alpha}), '
          f'then free convergence')
    else:
        sigma  = kick['sigma']
        blocks = list(sigma.keys())
        is_pm  = all(np.allclose(sigma[bl], 0) for bl in blocks)
        label  = 'PM reference (zero Σ_HF)' if is_pm else 'Sobol Σ_HF kick'
        r(f'  Run {i_run:3d}/{n_total}  │  {label}')
        if not is_pm:
            for bl in blocks:
                mat    = sigma[bl]
                eigs_s = _vfmt(np.sort(np.linalg.eigvalsh(mat))[::-1])
                diag_s = _vfmt(mat.diagonal())
                r(f'  Σ_init  {bl:4s}  eigs = {eigs_s}')
                r(f'          {bl:4s}  diag = {diag_s}')
    r(_rule('─'))


def _print_targeting_iter(it: int, mu: float,
                          n_up: float, n_dn: float,
                          dSigma: float, drho: float) -> None:
    mpi.report(
        f'  [T{it:3d}]  μ={mu:+8.4f}  N={n_up+n_dn:7.4f}'
        f'  m={n_up-n_dn:+7.4f}  dΣ={dSigma:.2e}  |Δρ|={drho:.2e}'
    )


def _print_release_iter(it: int, mu: float,
                        n_up: float, n_dn: float,
                        dSigma: float) -> None:
    mpi.report(
        f'  [R{it:3d}]  μ={mu:+8.4f}  N={n_up+n_dn:7.4f}'
        f'  m={n_up-n_dn:+7.4f}  dΣ={dSigma:.2e}'
    )


def _print_final_summary(solver, result: dict,
                         F: float, E_int: float, E_cf: float,
                         E_kin: float) -> None:
    r       = mpi.report
    density = result['density']
    blocks  = [bl for bl, _ in solver.gf_struct]
    labels  = ['up', 'dn']

    n_up_d = density[blocks[0]].diagonal().real
    n_dn_d = density[blocks[1]].diagonal().real
    m      = n_up_d.sum() - n_dn_d.sum()
    N      = n_up_d.sum() + n_dn_d.sum()

    r(_rule('─'))
    r(f'  μ      = {result["mu"]:+.6f}    N = {N:.6f}    m = {m:+.6f}')
    r(f'  F      = {F:.6f}    E_int = {E_int:.6f}'
      f'    E_cf = {E_cf:.6f}    E_kin = {E_kin:.6f}')
    for bl, lbl in zip(blocks, labels):
        rho  = density[bl]                    # full complex NxN
        sig  = solver.Sigma_HF[bl]            # full complex NxN
        eigs = np.sort(np.linalg.eigvalsh(rho.real))[::-1]
        for line in _mat_lines(sig, 'Σ_HF', lbl):
            r(line)
        for line in _mat_lines(rho, 'ρ', lbl):
            r(line)
        r(f'  eigval {lbl} = {_vfmt(eigs)}')


# ══════════════════════════════════════════════════════════════════════════════
#  Lattice free energy
# ══════════════════════════════════════════════════════════════════════════════

def _lattice_free_energy(solver, t, crystal_field=None) -> tuple:
    """Decomposed lattice free energy for the Bethe lattice.

    F = E_int + E_cf + E_kin

    E_int : ½ Tr[Σ_int · ρ]              HF interaction energy (from solver)
    E_cf  : Σ_σ Tr[ε_cf · ρ_σ]          crystal-field energy
    E_kin : (1/β) Σ_{iω} Tr[Δ(iω)·G(iω)]   Bethe kinetic energy
            where Δ(iω) = t²·G(iω)  (Bethe self-consistency relation)

    The kinetic sum is evaluated via the DLR quadrature.  For the Bethe
    lattice the hybridization function is purely local and equal to t²·G_loc,
    so Tr[Δ·G] = Tr[t²·G²] decays as 1/(iω)² — well within the DLR basis.

    Returns
    -------
    (F, E_int, E_cf, E_kin) : tuple of float
    """
    blocks = [bl for bl, _ in solver.gf_struct]
    norb   = solver.gf_struct[0][1]
    t_arr  = np.broadcast_to(np.atleast_1d(np.asarray(t, dtype=float)),
                              (norb,)).copy()
    t2_mat = np.diag(t_arr ** 2)

    # E_int = ½ Tr[Σ_int · ρ]
    E_int = float(np.real(solver.interaction_energy()))

    # E_cf = Σ_σ Tr[ε_cf · ρ_σ]
    E_cf = 0.0
    if crystal_field is not None:
        for bl in blocks:
            cf  = np.diag(np.asarray(crystal_field[bl], dtype=float))
            rho = solver.G_iw[bl].density().real
            E_cf += float(np.real(np.trace(cf @ rho)))

    # E_kin = (1/β) Σ_{iω} Tr[t²·G(iω)²]  via DLR quadrature
    DeltaG = solver.G_iw.copy()
    n_pts  = solver.G_iw[blocks[0]].data.shape[0]
    for bl in blocks:
        for i in range(n_pts):
            G = solver.G_iw[bl].data[i]
            DeltaG[bl].data[i] = t2_mat @ G @ G   # Δ(iω)·G(iω) = t²·G²
    E_kin = float(np.real(DeltaG.total_density()))

    return float(E_int + E_cf + E_kin), E_int, E_cf, E_kin


# ══════════════════════════════════════════════════════════════════════════════
#  DMFT loop with optional DM targeting
# ══════════════════════════════════════════════════════════════════════════════

def dmft_loop_targeted(
        solver,
        t,
        h_int,
        mu_init: float,
        n_target: float,
        max_iter: int   = 300,
        mix: float      = 0.3,
        eps: float      = 1e-6,
        adjust_mu: bool = True,
        mu_bracket: float = 30.0,
        rho_target: dict | None = None,
        n_target_steps: int   = 10,
        targeting_alpha: float = 1.5,
        spin_kick = None,
        crystal_field: dict | None = None,
        print_every: int = 20,
        **solver_kwargs,
) -> dict:
    """Bethe-lattice DMFT loop with optional DM targeting.

    All HF-solver output is silenced; structured per-iteration diagnostics
    are printed via ``mpi.report``.

    Parameters
    ----------
    rho_target     : ``{'up': array, 'down': array}`` or None.
                     If not None, apply Lagrange correction for the first
                     ``n_target_steps`` DMFT iterations, then release.
    n_target_steps : Number of targeting DMFT steps before release.
    targeting_alpha: Step size α for the targeting correction.
    spin_kick      : Initial Σ_HF (dict or float scalar).  Only used when
                     ``rho_target`` is None (Sobol mode).
    print_every    : Print a diagnostics line every this many release iterations.

    Returns
    -------
    dict with keys: ``Sigma_HF``, ``G_loc_iw``, ``density``, ``mu``,
                    ``n_iter``, ``converged``, ``solver``.
    """
    is_one_shot = solver_kwargs.get('one_shot', True)



    beta     = solver.beta
    gf_struct = solver.gf_struct
    blocks   = [bl for bl, _ in gf_struct]
    norb     = gf_struct[0][1]

    t_arr  = np.broadcast_to(np.atleast_1d(np.asarray(t, dtype=float)),
                              (norb,)).copy()
    t2_mat = np.diag(t_arr ** 2)

    cf_mat = (
        {bl: np.diag(np.asarray(crystal_field[bl], dtype=float)) for bl in blocks}
        if crystal_field is not None
        else {bl: np.zeros((norb, norb)) for bl in blocks}
    )

    # ── Apply initial Σ_HF kick (Sobol mode) ────────────────────────────────
    _sig_dtype = complex if not solver.force_real else float
    if spin_kick is not None:
        if isinstance(spin_kick, dict):
            for bl, mat in spin_kick.items():
                solver.Sigma_HF[bl] = np.asarray(mat, dtype=_sig_dtype)
        else:
            delta = float(spin_kick) * np.eye(norb, dtype=_sig_dtype)
            solver.Sigma_HF[blocks[0]] = solver.Sigma_HF[blocks[0]] + delta
            solver.Sigma_HF[blocks[1]] = solver.Sigma_HF[blocks[1]] - delta

    # ── Seed G0_iw and G_iw from semicircular DOS ────────────────────────────
    for bl in blocks:
        solver.G0_iw[bl] << SemiCircular(2.0 * t_arr[0])
        solver.G0_iw[bl] << inverse(inverse(solver.G0_iw[bl]) - cf_mat[bl])
        solver.G_iw[bl]  << inverse(inverse(solver.G0_iw[bl]) - solver.Sigma_HF[bl])

    n_pts = solver.G_iw[blocks[0]].data.shape[0]

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _set_G0(mu_val: float) -> None:
        """G0⁻¹[i] = (iω_i + μ)·I − cf − t²·G[i]  for each DLR point."""
        for bl in blocks:
            for i, iw in enumerate(solver.G0_iw[bl].mesh):
                iw_c   = complex(iw)
                G_loc  = solver.G_iw[bl].data[i]
                G0_inv = (iw_c + mu_val) * np.eye(norb) - cf_mat[bl] - t2_mat @ G_loc
                solver.G0_iw[bl].data[i] = np.linalg.inv(G0_inv)

    def _n_trial(mu_val: float) -> float:
        n = 0.0
        for bl in blocks:
            G_tmp = solver.G0_iw[bl].copy()
            for i, iw in enumerate(G_tmp.mesh):
                iw_c   = complex(iw)
                G_loc  = solver.G_iw[bl].data[i]
                G0_inv = (iw_c + mu_val) * np.eye(norb) - cf_mat[bl] - t2_mat @ G_loc
                G_inv  = G0_inv - solver.Sigma_HF[bl]
                G_tmp.data[i] = np.linalg.inv(G_inv)
            n += G_tmp.density().real.trace()
        return n

    def _update_G_iw() -> None:
        """Recompute G_iw from Dyson with current Σ_HF and G0_iw."""
        for bl in blocks:
            for i in range(n_pts):
                G0_val = solver.G0_iw[bl].data[i]
                solver.G_iw[bl].data[i] = np.linalg.inv(
                    np.linalg.inv(G0_val) - solver.Sigma_HF[bl])

    # Propagate kick asymmetry into G_iw before iterating
    if spin_kick is not None:
        _set_G0(mu_init)
        _update_G_iw()

    mu        = mu_init
    Sigma_old = {bl: solver.Sigma_HF[bl].copy() for bl in blocks}
    converged = False
    n_consec  = 0
    is_tgt    = rho_target is not None

    for it in range(max_iter):
        phase         = ('target' if (is_tgt and it < n_target_steps) else 'release')
        just_released = is_tgt and n_target_steps > 0 and it == n_target_steps

        # ── 1. Bethe self-consistency ─────────────────────────────────────
        _set_G0(mu)

        # ── 2. μ adjustment for target filling ───────────────────────────
        if adjust_mu and abs(_n_trial(mu) - n_target) > 1e-6:
            try:
                mu = brentq(lambda m: _n_trial(m) - n_target,
                            mu - mu_bracket, mu + mu_bracket,
                            xtol=1e-8, maxiter=200)
                _set_G0(mu)
            except ValueError:
                pass

        # ── 3. HF solve (all solver output silenced) ──────────────────────
        with _suppress_solver_output():

        # if targeting is active only perform one shot

            if is_tgt and it < n_target_steps:
                solver_kwargs['one_shot'] = True
            else:
                solver_kwargs['one_shot'] = is_one_shot

            solver.solve(h_int, **solver_kwargs)

        # ── 4. Mix Σ_HF ──────────────────────────────────────────────────
        Sigma_new = {bl: solver.Sigma_HF[bl].copy() for bl in blocks}
        if it > 0:
            for bl in blocks:
                solver.Sigma_HF[bl] = (mix * Sigma_new[bl]
                                       + (1.0 - mix) * Sigma_old[bl])
            _update_G_iw()

        # ── 4b. Targeting correction (targeting phase only) ───────────────
        drho_max = 0.0
        if phase == 'target':
            for bl in blocks:
                rho_cur  = solver.G_iw[bl].density().real
                corr     = targeting_alpha * (rho_cur - rho_target[bl])
                solver.Sigma_HF[bl] = solver.Sigma_HF[bl] + corr
                drho_max = max(drho_max,
                               np.max(np.abs(rho_cur - rho_target[bl])))
            _update_G_iw()

        # ── 5. Convergence metric ─────────────────────────────────────────
        max_diff  = max(
            np.max(np.abs(solver.Sigma_HF[bl] - Sigma_old[bl]))
            for bl in blocks
        )
        Sigma_old = {bl: solver.Sigma_HF[bl].copy() for bl in blocks}

        # ── 6. Diagnostics ────────────────────────────────────────────────
        density  = {bl: solver.G_iw[bl].density().real for bl in blocks}
        n_up_sum = density[blocks[0]].trace()
        n_dn_sum = density[blocks[1]].trace()

        if just_released:
            mpi.report('  ' + _rule('─', 60))
            mpi.report('  ── RELEASING constraints (free convergence from here) ──')
            mpi.report('  ' + _rule('─', 60))

        if phase == 'target':
            _print_targeting_iter(it + 1, mu, n_up_sum, n_dn_sum, max_diff, drho_max)
        else:
            rel_it   = it - (n_target_steps if is_tgt else 0) + 1
            is_last  = (it + 1 >= max_iter)
            if rel_it == 1 or rel_it % print_every == 0 or is_last:
                _print_release_iter(rel_it, mu, n_up_sum, n_dn_sum, max_diff)

        # ── 7. Convergence check (release phase only) ─────────────────────
        if phase == 'release':
            if max_diff < eps and it > 0:
                n_consec += 1
                if n_consec >= 2:
                    converged = True
                    break
            else:
                n_consec = 0

    if converged:
        mpi.report(f'  ── CONVERGED after {it + 1} DMFT iterations ──')
    else:
        mpi.report(f'  ── NOT CONVERGED (max_iter={max_iter}) ──')

    return {
        'Sigma_HF': {bl: solver.Sigma_HF[bl].copy() for bl in blocks},
        'G_loc_iw': solver.G_iw.copy(),
        'density':  {bl: solver.G_iw[bl].density().real for bl in blocks},
        'mu':       mu,
        'n_iter':   it + 1,
        'converged': converged,
        'solver':   solver,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  HDF5 I/O
# ══════════════════════════════════════════════════════════════════════════════

def save_run(archive_path: str, n_converged: int, solver,
             result: dict, F: float, E_int: float, E_cf: float,
             E_kin: float, n_up: np.ndarray, n_down: np.ndarray) -> str:
    """Save one converged run to the HDF5 archive.  Returns the run key."""
    run_key = f'run_{n_converged:04d}'
    blocks  = [bl for bl, _ in solver.gf_struct]
    with HDFArchive(archive_path, 'a') as ar:
        ar.create_group(run_key)
        g = ar[run_key]
        g['G_iw']          = solver.G_iw
        g['G0_iw']         = solver.G0_iw
        g['Sigma_HF_up']   = solver.Sigma_HF[blocks[0]].copy()
        g['Sigma_HF_down'] = solver.Sigma_HF[blocks[1]].copy()
        g['F']             = F
        g['F_int']         = E_int
        g['F_cf']          = E_cf
        g['F_kin']         = E_kin
        g['m']             = float(n_up.sum() - n_down.sum())
        g['n_total']       = float(n_up.sum() + n_down.sum())
        g['mu']            = result['mu']
        g['n_up']          = n_up.copy()
        g['n_down']        = n_down.copy()
    return run_key


def load_runs(path: str) -> list:
    """Load converged runs from the HDF5 archive, sorted by F (ascending).

    Each run is a dict with keys:
        G_iw, G0_iw, Sigma_HF (dict up/down), F, F_int, F_cf, F_kin,
        m, n_total, mu, n_up, n_down.
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
                'F_cf'    : float(g['F_cf']) if 'F_cf' in g else 0.0,
                'F_kin'   : float(g['F_kin']),
                'm'       : float(g['m']),
                'n_total' : float(g['n_total']),
                'mu'      : float(g['mu']),
                'n_up'    : np.array(g['n_up']),
                'n_down'  : np.array(g['n_down']),
            })
    return sorted(runs, key=lambda r: r['F'])


# ══════════════════════════════════════════════════════════════════════════════
#  Outer landscape loop
# ══════════════════════════════════════════════════════════════════════════════

def run_landscape(
        kicks: list,
        solver_factory,
        t,
        h_int,
        mu_init: float,
        n_target: float,
        crystal_field,
        archive_path: str,
        params_dict: dict,
        n_target_steps: int   = 10,
        targeting_alpha: float = 1.5,
        print_every: int      = 20,
        **dmft_kwargs,
) -> int:
    """Run independent DMFT loops for every kick; save converged results.

    Parameters
    ----------
    kicks          : list of kick dicts built by ``generate_sobol_kicks``,
                     ``generate_dm_kicks`` or ``pm_kick``.  Sobol and DM
                     kicks can be mixed freely in the same list.
    solver_factory : callable() → fresh ImpuritySolver instance
    t, h_int, mu_init, n_target, crystal_field : model parameters
    archive_path   : path to HDF5 results file (created / overwritten)
    params_dict    : dict of model parameters stored under ``'params'``
    n_target_steps : DM-kick targeting iterations before release
    targeting_alpha: DM-kick correction step size α
    print_every    : release-phase print interval (iterations)
    **dmft_kwargs  : forwarded to ``dmft_loop_targeted``
                     (max_iter, mix, eps, adjust_mu, mu_bracket,
                      with_fock, one_shot, method, tol, …)

    Returns
    -------
    n_converged : int
    """
    n_total = len(kicks)
    n_conv  = 0

    with HDFArchive(archive_path, 'w') as ar:
        ar['params'] = params_dict

    mpi.report('\n' + _rule('═'))
    mpi.report(f'  HF-DMFT landscape  |  {n_total} kicks total'
               f'  |  archive: {archive_path}')
    mpi.report(_rule('═'))

    for i, kick in enumerate(kicks):
        solver = solver_factory()

        _print_run_header(i + 1, n_total, kick,
                          n_target_steps=n_target_steps,
                          targeting_alpha=targeting_alpha)

        if kick['type'] == 'sobol':
            result = dmft_loop_targeted(
                solver, t, h_int, mu_init, n_target,
                spin_kick      = kick['sigma'],
                rho_target     = None,
                n_target_steps = n_target_steps,
                targeting_alpha = targeting_alpha,
                crystal_field  = crystal_field,
                print_every    = print_every,
                **dmft_kwargs,
            )
        else:  # 'dm'
            result = dmft_loop_targeted(
                solver, t, h_int, mu_init, n_target,
                spin_kick      = None,
                rho_target     = kick['rho_target'],
                n_target_steps = n_target_steps,
                targeting_alpha = targeting_alpha,
                crystal_field  = crystal_field,
                print_every    = print_every,
                **dmft_kwargs,
            )

        if not result['converged']:
            continue

        F, E_int, E_cf, E_kin = _lattice_free_energy(solver, t, crystal_field)
        density  = result['density']
        bl0, bl1 = [bl for bl, _ in solver.gf_struct]
        n_up     = density[bl0].diagonal().real
        n_down   = density[bl1].diagonal().real

        _print_final_summary(solver, result, F, E_int, E_cf, E_kin)
        save_run(archive_path, n_conv, solver, result,
                 F, E_int, E_cf, E_kin, n_up, n_down)
        n_conv += 1
        mpi.report(_rule('═'))

    mpi.report(f'\n  Landscape complete: {n_conv}/{n_total} converged')
    mpi.report(f'  Results saved to: {archive_path}')
    mpi.report(_rule('═') + '\n')
    return n_conv
