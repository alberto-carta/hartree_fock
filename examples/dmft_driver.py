"""dmft_driver.py — DMFT self-consistency loop driver for the Hartree-Fock impurity solver.

Provides a reusable Bethe-lattice DMFT loop that works with the
``triqs_hartree_fock.ImpuritySolver``.

Bethe lattice self-consistency:
    G0⁻¹(iω) = iω + μ - t²_α · G_loc,αα(iω)
where t_α is the orbital hopping amplitude (half-bandwidth W_α = 2t_α).

The driver handles:
    - Bethe lattice self-consistency (no k-sum needed)
    - Chemical potential adjustment for target filling
    - Linear mixing of the static Hartree-Fock self-energy
    - Optional spin-symmetry-breaking initial kick
    - Convergence monitoring
"""

import numpy as np
import triqs.utility.mpi as mpi
from triqs.gf import Gf, BlockGf, MeshImFreq, SemiCircular
from triqs.lattice.tight_binding import TBLattice
from triqs.operators import n, c, c_dag, Operator
from triqs.operators.util.hamiltonians import h_int_kanamori


def make_square_lattice_1orb(t=0.25, tp=0.0):
    """Construct a 1-orbital 2D square lattice.

    Parameters
    ----------
    t : float
        Nearest-neighbor hopping (default 0.25, giving bandwidth W=2).
    tp : float
        Next-nearest-neighbor hopping.

    Returns
    -------
    TBLattice
    """
    return TBLattice(
        units=[(1, 0, 0), (0, 1, 0)],
        hoppings={
            (1, 0): [[t]],
            (-1, 0): [[t]],
            (0, 1): [[t]],
            (0, -1): [[t]],
            (1, 1): [[tp]],
            (-1, -1): [[tp]],
            (1, -1): [[tp]],
            (-1, 1): [[tp]],
        },
        orbital_positions=[(0, 0, 0)],
    )


def make_square_lattice_2orb(t=0.25, delta=0.4):
    """Construct a 2-orbital 2D square lattice with crystal-field splitting.

    The two orbitals have different bandwidths due to the crystal field:
    orbital 0 = narrow band (shifted by +δ/2)
    orbital 1 = wide band (shifted by -δ/2)

    This matches the C++ hk_sqx2_del0.4.dat with t=0.25, delta=0.4.

    Parameters
    ----------
    t : float
        Nearest-neighbor hopping.
    delta : float
        Crystal-field splitting (energy units, not δ/t).

    Returns
    -------
    TBLattice
    """
    # On-site crystal field: orbital 0 gets +δ/2, orbital 1 gets -δ/2
    h0 = [[delta / 2, 0],
           [0, -delta / 2]]

    return TBLattice(
        units=[(1, 0, 0), (0, 1, 0)],
        hoppings={
            (0, 0): h0,
            (1, 0): [[t, 0], [0, t]],
            (-1, 0): [[t, 0], [0, t]],
            (0, 1): [[t, 0], [0, t]],
            (0, -1): [[t, 0], [0, t]],
        },
        orbital_positions=[(0, 0, 0)] * 2,
    )


def make_h_int_hubbard(U, norb=1):
    """Construct a Hubbard interaction Hamiltonian.

    h_int = U Σ_α n_{α,↑} n_{α,↓}

    Parameters
    ----------
    U : float
        On-site Hubbard interaction (in energy units).
    norb : int
        Number of orbitals.

    Returns
    -------
    triqs.operators.Operator
    """
    h = Operator()
    for a in range(norb):
        h += U * n('up', a) * n('down', a)
    return h


def make_h_int_kanamori(U, Up, J, Jp, norb):
    """Construct a Kanamori interaction Hamiltonian.

    Parameters
    ----------
    U : float
        Intra-orbital Hubbard.
    Up : float
        Inter-orbital Hubbard.
    J : float
        Hund's coupling.
    Jp : float
        Pair-hopping (set to J for full rotationally invariant, 0 for
        density-density approximation).
    norb : int
        Number of orbitals.

    Returns
    -------
    triqs.operators.Operator
    """
    return h_int_kanamori(
        ['up', 'down'],
        norb,
        # U_same_spin[a,b] (a≠b): same-spin density-density = U' - J
        np.array([[0, Up - J], [Up - J, 0]]) if norb == 2 else np.full((norb, norb), Up - J) - np.diag(np.full(norb, Up - J)),
        # U_opp_spin[a,b]: opposite-spin density-density = U (diagonal) / U' (off-diagonal)
        np.array([[U, Up], [Up, U]]) if norb == 2 else np.full((norb, norb), Up) + np.diag(np.full(norb, U - Up)),
        J, off_diag=True, map_operator_structure=None
    )


def make_h_int_kanamori_simple(U, Up, J, Jp, norb):
    """Hand-build Kanamori h_int matching the C++ convention.
    For some reason the paper removes the pair-hopping term, so we set Jp=0 to match the C++ code. :/ me no likey...

    h_int = U Σ_a n_{a↑}n_{a↓}
          + Σ_{a<b} [ Up n_{a↑}n_{b↓} + Up n_{a↓}n_{b↑}
                     + (Up-J) n_{a↑}n_{b↑} + (Up-J) n_{a↓}n_{b↓} ]
          - J Σ_{a≠b} c†_{a↑} c†_{a↓} c_{b↓} c_{b↑}   (spin-flip)
          + Jp Σ_{a≠b} c†_{a↑} c†_{b↓} c_{a↓} c_{b↑}  (pair-hopping)
    """
    h = Operator()
    spin_names = ['up', 'down']

    for a in range(norb):
        h += U * n('up', a) * n('down', a)

    for a in range(norb):
        for b in range(norb):
            if a == b:
                continue
            # Density-density inter-orbital
            for s in spin_names:
                for sp in spin_names:
                    if s == sp:
                        h += 0.5 * (Up - J) * n(s, a) * n(sp, b)
                    else:
                        h += 0.5 * Up * n(s, a) * n(sp, b)

            # Spin-flip: -J c†_{a↑} c_{a↓} c†_{b↓} c_{b↑}
            h += -J * c_dag('up', a) * c('down', a) * c_dag('down', b) * c('up', b)

            # Pair-hopping: Jp c†_{a↑} c†_{a↓} c_{b↓} c_{b↑}
            h += Jp * c_dag('up', a) * c_dag('down', a) * c('down', b) * c('up', b)

    return h


# ──────────────────────────────────────────────────────────────────────────────
#  Bethe lattice DMFT loop for the Hartree-Fock impurity solver
# ──────────────────────────────────────────────────────────────────────────────

def dmft_loop_bethe_hf(solver, t, h_int, mu_init, n_target,
                       max_iter=200, mix=0.5, eps=1e-6,
                       verbose=True, adjust_mu=True, mu_bracket=5.0,
                       spin_kick=None,
                       **solver_kwargs):
    """Run a DMFT self-consistency loop on the Bethe lattice using the
    Hartree-Fock impurity solver (``triqs_hartree_fock.ImpuritySolver``).

    The Bethe lattice self-consistency is:

        G0⁻¹(iω) = iω + μ - t²_α · G_loc,αα(iω)

    No k-sum is needed.  The HF self-energy is a static matrix per block,
    so the quasiparticle weight Z = 1 by construction.

    Parameters
    ----------
    solver : ImpuritySolver
        A ``triqs_hartree_fock.ImpuritySolver`` instance with ``gf_struct``,
        ``beta``, ``G0_iw`` (DLR BlockGf) and ``Sigma_HF`` (dict of arrays).
    t : float or array-like of float
        Hopping amplitude(s).  A scalar applies the same value to every
        orbital; an array of length *norb* sets per-orbital hoppings.
        The half-bandwidth of orbital α is W_α = 2 t_α.
    h_int : triqs.operators.Operator
        Local interaction Hamiltonian.
    mu_init : float
        Initial chemical potential guess.
    n_target : float
        Target total filling (summed over all spin and orbital flavours).
    max_iter : int
        Maximum DMFT iterations.
    mix : float
        Linear mixing factor for the static self-energy:
        Σ ← mix·Σ_new + (1-mix)·Σ_old.
    eps : float
        Convergence threshold on max|ΔΣ| (element-wise, over all blocks).
    verbose : bool
        Print per-iteration diagnostics.
    adjust_mu : bool
        Adjust μ at each iteration via Brent's method to hit ``n_target``.
    mu_bracket : float
        Half-width of the μ search bracket around the current μ.
    spin_kick : None, float, or dict
        Optional initial Sigma_HF applied *before* the first DMFT iteration
        to seed spin- and/or orbital-symmetry-broken solutions.

        * ``None``  — no kick (default, fully symmetric start).
        * ``float`` — scalar δ: sets Σ_HF['up'] += δ·I, Σ_HF['down'] -= δ·I.
        * ``dict``  — ``{'up': array_up, 'down': array_down}`` where each
          array is a (norb × norb) matrix that is set *directly* as
          ``Sigma_HF`` for that block (not added — it replaces the zero
          initial value).  This is the most flexible form: you can break
          spin symmetry, orbital degeneracy, or both simultaneously.

          Example (2-orbital ferro + orbital polarisation)::

              spin_kick = {
                  'up':   np.diag([1.0, 0.5]),
                  'down': np.diag([-1.0, -0.5]),
              }
    **solver_kwargs
        Extra keyword arguments forwarded to ``solver.solve()``.  Useful
        parameters include ``with_fock``, ``one_shot``, ``method``, ``tol``.

    Returns
    -------
    dict with keys
        ``Sigma_HF``   : dict of numpy arrays — converged static self-energy
        ``G_loc_iw``   : BlockGf (DLR) — converged local Green's function
        ``density``    : dict — converged density matrices per block
        ``mu``         : float — converged chemical potential
        ``n_iter``     : int — number of iterations performed
        ``converged``  : bool
        ``solver``     : the solver object (carries all GFs)
    """
    from scipy.optimize import brentq

    beta = solver.beta
    gf_struct = solver.gf_struct
    block_names = [bl for bl, _ in gf_struct]
    norb = gf_struct[0][1]
    bl0 = block_names[0]

    # ── Per-orbital t² ──────────────────────────────────────────────────────
    t_arr = np.broadcast_to(np.atleast_1d(np.asarray(t, dtype=float)), (norb,)).copy()
    t2 = t_arr ** 2
    t2_mat = np.diag(t2)  # (norb, norb)

    # ── Optional spin kick to seed symmetry breaking ────────────────────────
    # spin_kick can be:
    #   None  — no kick (paramagnetic start)
    #   float — scalar δ: Σ_HF['up'] = +δ·I,  Σ_HF['down'] = -δ·I
    #   dict  — {'up': (norb×norb) array, 'down': (norb×norb) array}
    #           matrices are used directly as the initial Sigma_HF for each block
    if spin_kick is not None:
        if isinstance(spin_kick, dict):
            for bl, kick_mat in spin_kick.items():
                mat = np.asarray(kick_mat, dtype=complex if not solver.force_real else float)
                assert mat.shape == (norb, norb), (
                    f"spin_kick['{bl}'] must be a ({norb},{norb}) matrix, got {mat.shape}")
                solver.Sigma_HF[bl] = mat
        else:
            delta = float(spin_kick)
            kick_diag = delta * np.eye(norb)
            solver.Sigma_HF['up']   = solver.Sigma_HF['up']   + kick_diag
            solver.Sigma_HF['down'] = solver.Sigma_HF['down'] - kick_diag
        mpi.report('DMFT: applied spin_kick:')
        for bl in block_names:
            mpi.report(f'  Sigma_HF[{bl!r}] =\n{solver.Sigma_HF[bl]}')

    # ── Seed solver.G_iw with the non-interacting semicircular GF ─────────────
    # SemiCircular lazy expression works on DLR meshes via the << operator.
    for bl in block_names:
        for a in range(norb):
            solver.G_iw[bl][a, a] << SemiCircular(2.0 * t_arr[a])

    # Helper: set G0_iw from the current solver.G_iw using the Bethe relation.
    # Operates directly on .data arrays to avoid __call__ on the DLR mesh.
    def _set_G0_bethe(mu_val):
        """G0⁻¹[i] = (iω_i + μ)·I − t²·G_iw[i]  for each DLR point i."""
        for bl in block_names:
            mesh_pts = list(solver.G0_iw[bl].mesh)
            for i, iw in enumerate(mesh_pts):
                iw_val = complex(iw)
                G_loc_val = solver.G_iw[bl].data[i]
                G0_inv = (iw_val + mu_val) * np.eye(norb) - t2_mat @ G_loc_val
                solver.G0_iw[bl].data[i] = np.linalg.inv(G0_inv)

    def _total_density_trial(mu_val):
        """Compute total filling for a trial μ with current Sigma_HF (G_iw fixed)."""
        n_total = 0.0
        for bl in block_names:
            G_tmp = solver.G0_iw[bl].copy()
            mesh_pts = list(G_tmp.mesh)
            for i, iw in enumerate(mesh_pts):
                iw_val = complex(iw)
                G_loc_val = solver.G_iw[bl].data[i]
                G0_inv = (iw_val + mu_val) * np.eye(norb) - t2_mat @ G_loc_val
                G_inv = G0_inv - solver.Sigma_HF[bl]
                G_tmp.data[i] = np.linalg.inv(G_inv)
            n_total += G_tmp.density().real.trace()
        return n_total

    mu = mu_init
    Sigma_old = {bl: solver.Sigma_HF[bl].copy() for bl in block_names}

    # ── If a spin kick was applied: propagate the asymmetry into G_iw ───────
    # G_iw is currently spin-symmetric (seeded from SemiCircular for all blocks).
    # Without this step, _set_G0_bethe() in iteration 1 would produce symmetric
    # G0 and the internal HF root-finder would collapse back to the paramagnetic
    # solution regardless of the kick on Sigma_HF.
    # Fix: compute a preliminary G0 from the symmetric G_iw, then apply Dyson
    # with the kicked (asymmetric) Sigma so that G_iw itself becomes asymmetric.
    if spin_kick is not None:
        _set_G0_bethe(mu_init)
        n_pts = solver.G_iw[block_names[0]].data.shape[0]
        for bl in block_names:
            for i in range(n_pts):
                G0_val = solver.G0_iw[bl].data[i]
                solver.G_iw[bl].data[i] = np.linalg.inv(
                    np.linalg.inv(G0_val) - solver.Sigma_HF[bl]
                )

    converged = False
    n_converged = 0

    for it in range(max_iter):

        # ── Step 1: Bethe self-consistency → set G0_iw ──────────────────────
        _set_G0_bethe(mu)

        # ── Step 2: Optional μ adjustment for target filling ─────────────────
        if adjust_mu:
            n_total = _total_density_trial(mu)

            if abs(n_total - n_target) > 1e-6:
                try:
                    mu = brentq(
                        lambda mu_t: _total_density_trial(mu_t) - n_target,
                        mu - mu_bracket, mu + mu_bracket,
                        xtol=1e-8, maxiter=200,
                    )
                    _set_G0_bethe(mu)
                except ValueError:
                    mpi.report(f'  WARNING: μ brentq bracket failed, keeping μ={mu:.6f}')

        # ── Step 3: Impurity solve ───────────────────────────────────────────
        solver.solve(h_int, **solver_kwargs)

        # ── Step 4: Mix Sigma_HF ─────────────────────────────────────────────
        Sigma_new = {bl: solver.Sigma_HF[bl].copy() for bl in block_names}
        if it > 0:
            for bl in block_names:
                solver.Sigma_HF[bl] = mix * Sigma_new[bl] + (1.0 - mix) * Sigma_old[bl]
            # Recompute G_iw with the mixed Sigma so G_loc stays consistent
            mesh_pts_cache = {bl: list(solver.G_iw[bl].mesh) for bl in block_names}
            for bl in block_names:
                for i in range(len(mesh_pts_cache[bl])):
                    G0_val = solver.G0_iw[bl].data[i]
                    solver.G_iw[bl].data[i] = np.linalg.inv(
                        np.linalg.inv(G0_val) - solver.Sigma_HF[bl]
                    )

        # ── Step 5: Convergence check ────────────────────────────────────────
        max_diff = max(
            np.max(np.abs(solver.Sigma_HF[bl] - Sigma_old[bl]))
            for bl in block_names
        )
        Sigma_old = {bl: solver.Sigma_HF[bl].copy() for bl in block_names}

        # ── Step 6: Diagnostics ──────────────────────────────────────────────
        density = {bl: solver.G_iw[bl].density().real for bl in block_names}
        n_orb_str = ", ".join(f"{density[bl0][a, a]:.4f}" for a in range(norb))
        sig_up_str = ", ".join(
            f"{solver.Sigma_HF[bl0][a, a]:.4f}" for a in range(norb)
        )
        mpi.report(
            f"DMFT {it+1:4d}: μ={mu:.4f}  n=[{n_orb_str}]"
            f"  Σ_HF[{bl0}]=diag([{sig_up_str}])"
            f"  dΣ={max_diff:.2e}"
        )

        # ── Step 7: Check convergence (require 2 consecutive passes) ─────────
        if max_diff < eps and it > 0:
            n_converged += 1
            if n_converged >= 2:
                converged = True
                if verbose:
                    mpi.report(f"  Converged after {it+1} iterations.")
                break
        else:
            n_converged = 0

    return {
        'Sigma_HF':  {bl: solver.Sigma_HF[bl].copy() for bl in block_names},
        'G_loc_iw':  solver.G_iw.copy(),
        'density':   {bl: solver.G_iw[bl].density().real for bl in block_names},
        'mu':        mu,
        'n_iter':    it + 1,
        'converged': converged,
        'solver':    solver,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Ensemble DMFT loop (Bethe lattice)
# ──────────────────────────────────────────────────────────────────────────────

def dmft_loop_ensemble_hf(
    ensemble_solver,
    dm_proposals,
    h_int,
    G0_iw,
    t,
    mu_init,
    n_target,
    n_elec_total,
    max_iter: int = 10,
    eps: float = 1e-3,
    mix: float = 0.5,
    adjust_mu: bool = True,
    mu_bracket: float = 5.0,
    with_fock: bool = True,
    hf_method: str = 'hybr',
    hf_tol: float = 1e-8,
    seed: int = 42,
    verbose: bool = True,
):
    """Run a Bethe-lattice DMFT self-consistency loop using the
    :class:`IncoherentEnsembleSolver` as the impurity solver.

    At each iteration the ensemble-averaged Green's function
    ``G_ens(iω) = Σ_i w_i G_i(iω)`` plays the role of ``G_loc`` in the
    Bethe self-consistency relation:

        G0⁻¹(iω) = (iω + μ)·I − t²_α · G_ens,αα(iω)

    The previous iteration's :class:`SolutionSet` is fed back into
    ``dm_proposals.prior_solutions`` so that the next ensemble solve is
    warm-started from already-discovered saddle points.

    Parameters
    ----------
    ensemble_solver : IncoherentEnsembleSolver
        Fully constructed solver instance (not yet solved; or may have been
        solved previously – its state is overwritten each iteration).
    dm_proposals    : DensityMatrixProposals
        Proposal generator.  Its ``prior_solutions`` attribute is updated
        in-place between iterations.
    h_int           : triqs.operators.Operator
        Local interaction Hamiltonian.
    G0_iw           : BlockGf (DLR)
        Starting bath Green's function (e.g. from a converged PM DMFT run).
        The object is *not* modified; a working copy is maintained internally.
    t               : float or array-like
        Bethe-lattice hopping amplitude(s).  Same convention as
        :func:`dmft_loop_bethe_hf`.
    mu_init         : float
        Starting chemical potential.
    n_target        : float
        Target total filling (Brentq target for μ adjustment).
    n_elec_total    : float
        Passed to ``ensemble_solver.solve()`` for DM proposal spin splits
        (usually equal to ``n_target``).
    max_iter        : int
        Maximum number of ensemble-DMFT iterations (default 10).
    eps             : float
        Convergence threshold on  Σ_bl ‖ΔG_ens[bl]‖_F  (default 1e-3).
    mix             : float
        Linear mixing fraction for the bath G0 update:
        G0 ← (1 − mix)·G0_old + mix·G0_Bethe.  Use values in (0, 1];
        smaller values damp oscillations at the cost of slower convergence.
        Default 0.5.
    adjust_mu       : bool
        Adjust μ via Brent's method after each ensemble solve.
    mu_bracket      : float
        Half-width of the μ search bracket.
    with_fock       : bool
        Include Fock terms in each inner HF solve.
    hf_method       : str
        scipy root-finder method for the inner HF self-consistency.
    hf_tol          : float
        Tolerance for the inner HF solver.
    seed            : int
        Base random seed; iteration ``it`` uses ``seed + it * 1000``.
    verbose         : bool
        Print per-iteration diagnostics.

    Returns
    -------
    dict with keys
        ``ensemble``  : the IncoherentEnsembleSolver (final state)
        ``G0_iw``     : BlockGf – converged bath Green's function
        ``mu``        : float – converged chemical potential
        ``n_iter``    : int
        ``converged`` : bool
    """
    from scipy.optimize import brentq

    gf_struct   = ensemble_solver.gf_struct
    block_names = [bl for bl, _ in gf_struct]
    norb        = gf_struct[0][1]

    t_arr  = np.broadcast_to(np.atleast_1d(np.asarray(t, dtype=float)), (norb,)).copy()
    t2_mat = np.diag(t_arr ** 2)

    mu     = float(mu_init)
    G0_cur = G0_iw.copy()

    def _update_G0(G_ens, mu_val):
        """Overwrite G0_cur in-place: G0⁻¹[i] = (iω_i+μ)·I − t²·G_ens[i]."""
        for bl in block_names:
            for i, iw in enumerate(G0_cur[bl].mesh):
                iw_v   = complex(iw)
                G0_inv = (iw_v + mu_val) * np.eye(norb) - t2_mat @ G_ens[bl].data[i]
                G0_cur[bl].data[i] = np.linalg.inv(G0_inv)

    def _density_with_sigma(mu_val, Sigma_iw):
        """Total local density at trial μ using the current ensemble Σ(iω).

        G_trial⁻¹(iω) = G0⁻¹(iω; μ_trial) − Σ_ens(iω)
                       = [(iω+μ)·I − t²·G_ens] − Σ_ens

        This is the physically correct density for a correlated system:
        the bath G0 alone gives the wrong answer in a Mott state.
        """
        n_tot = 0.0
        for bl in block_names:
            G_trial = G0_cur[bl].copy()
            for i, iw in enumerate(G_trial.mesh):
                iw_v   = complex(iw)
                G0_inv = (iw_v + mu_val) * np.eye(norb) - t2_mat @ G_ens_cur[bl].data[i]
                G_trial.data[i] = np.linalg.inv(G0_inv - Sigma_iw[bl].data[i])
            n_tot += float(G_trial.total_density().real)
        return n_tot
    
    # def density_from_G_ens(mu_val, G_ens):
    #     """Helper to compute total density from the current G_ens at trial μ."""
    #     G_new  = G_ens.copy()

    #     G_new = inverse(invers

    # G_ens_cur is set inside the loop before _density_with_sigma is called
    G_ens_cur = G0_cur.copy()

    G_ens_prev = None
    converged  = False

    for it in range(max_iter):
        if verbose:
            mpi.report(f'\n  {"─"*66}')
            mpi.report(f'  Ensemble DMFT  iteration {it + 1}/{max_iter}   μ = {mu:+.6f}')
            mpi.report(f'  {"─"*66}')

        # ── Run ensemble solve on the current bath ─────────────────────────
        for bl in block_names:
            ensemble_solver.G0_iw[bl].data[:] = G0_cur[bl].data
        ensemble_solver.solve(
            h_int               = h_int,
            proposal_generators = [dm_proposals],
            mu                  = mu,
            n_elec_total        = n_elec_total,
            with_fock           = with_fock,
            hf_method           = hf_method,
            hf_tol              = hf_tol,
            seed                = seed + it * 1000,
        )

        G_ens_cur  = ensemble_solver.G_iw
        Sigma_cur  = ensemble_solver.Sigma_iw

        # ── Measure local density from G_ens (the true observable) ─────────
        n_ens = float(sum(
            G_ens_cur[bl].total_density().real for bl in block_names))

        # ── Optional μ adjustment via Brent on G_trial = (G0⁻¹ − Σ_ens)⁻¹ ─
        # _density_with_sigma uses G_ens_cur and Sigma_cur (closed over above)
        # so the Brent root-finder sees the correct correlated n(μ) curve.
        if adjust_mu and abs(n_ens - n_target) > 1e-6:
            def _f_brent(mu_t):
                return _density_with_sigma(mu_t, Sigma_cur) - n_target
            try:
                mu = brentq(_f_brent,
                             mu - mu_bracket, mu + mu_bracket,
                             xtol=1e-6, maxiter=200)
            except ValueError:
                mpi.report(f'  WARNING: μ Brent bracket [{mu-mu_bracket:.4f}, '
                           f'{mu+mu_bracket:.4f}] failed at iter {it+1}, '
                           f'keeping μ = {mu:+.6f}')

        # ── Update bath via Bethe SC (with mixing) ─────────────────────────
        G0_old = G0_cur.copy()
        _update_G0(G_ens_cur, mu)
        if mix < 1.0:
            for bl in block_names:
                G0_cur[bl].data[:] = (1.0 - mix) * G0_old[bl].data + mix * G0_cur[bl].data

        # ── Convergence check ──────────────────────────────────────────────
        if G_ens_prev is not None:
            delta_G = sum(
                float(np.linalg.norm(G_ens_cur[bl].data - G_ens_prev[bl].data))
                for bl in block_names
            )
            if verbose:
                mpi.report(f'  n_ens = {n_ens:.5f}   μ = {mu:+.6f}'
                           f'   ΔG_ens = {delta_G:.4e}   mix = {mix:.2f}')
            if delta_G < eps:
                converged = True
                if verbose:
                    mpi.report(f'  ✓ Ensemble DMFT converged after {it + 1} '
                               f'iteration(s).')
                break
        else:
            if verbose:
                mpi.report(f'  n_ens = {n_ens:.5f}   μ = {mu:+.6f}   '
                           f'(first iteration, no convergence check)')

        # ── Warm-start next iteration ──────────────────────────────────────
        dm_proposals.prior_solutions = ensemble_solver.solutions
        G_ens_prev = G_ens_cur.copy()

    return {
        'ensemble'  : ensemble_solver,
        'G0_iw'     : G0_cur,
        'mu'        : mu,
        'n_iter'    : it + 1,
        'converged' : converged,
    }
