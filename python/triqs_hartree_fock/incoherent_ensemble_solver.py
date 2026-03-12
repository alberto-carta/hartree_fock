# Copyright (c) 2022 Simons Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Authors: Alberto Carta, Jonathan Karp, Alexander Hampel

"""
Incoherent ensemble impurity solver for Hartree-Fock DMFT.

The solver discovers multiple self-consistent HF saddle points on a fixed
converged bath G0, Boltzmann-weights them by their impurity free energy, and
returns an ensemble-averaged Green's function together with the corresponding
many-body self-energy Σ_ens(iω) = G0⁻¹ - G_ens⁻¹.

Architecture
------------
Proposal generators (separate concern):
    SobolSigmaProposals       – Low-discrepancy sampling of Σ_HF initial
                                conditions directly in the real-symmetric space.
    DensityMatrixProposals    – Proposes random target density matrices
                                (SO(N)-rotated integer-occupation diagonals),
                                runs a fixed-point targeting loop to build a
                                Σ_HF that approximately reproduces them, then
                                uses that Σ_HF as the starting guess for a full
                                self-consistent HF solve.

Main solver:
    IncoherentEnsembleSolver  – Accepts a list of proposal generators, runs an
                                ImpuritySolver for each proposal, and assembles
                                the ensemble.

Free energy
-----------
The impurity free energy used for Boltzmann weighting is

    F = E_int + E_kin

where
    E_int = ½ Tr[Σ_int · ρ]   (interaction energy, from ImpuritySolver)
    E_kin = (1/β) Σ_{iω} Tr[ ln(iω · G(iω)) ]   (kinetic/entropy term)

Both terms vary between saddle points at fixed G0; only differences matter for
the weights.  The kinetic term is computed on the DLR mesh via scipy.linalg.logm
applied frequency-by-frequency.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.linalg import logm
from scipy.stats import ortho_group, qmc

from triqs.gf import BlockGf, Gf, iOmega_n, inverse
from triqs.gf.meshes import MeshDLRImFreq
import triqs.utility.mpi as mpi

from .impurity import ImpuritySolver


# ══════════════════════════════════════════════════════════════════════════════
#  SaddlePoint – data container for one converged HF saddle point
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Solution:
    """Single converged HF saddle point with all associated data.

    Normally obtained from ``IncoherentEnsembleSolver.solutions[i]``.

    Attributes
    ----------
    G_iw           : converged impurity Green's function (DLR BlockGf)
    Sigma_HF       : static HF self-energy, {block: (norb × norb) ndarray}
    density        : spin-resolved density matrices, {block: (norb × norb) ndarray}
    free_energy    : impurity free energy  F = E_int + E_kin
    weight         : Boltzmann weight
    magnetisation  : total magnetisation  m = Σ_a (n↑_a − n↓_a)
    n_total        : total electron number ⟨N⟩
    """
    G_iw:            object = field(repr=False)   # BlockGf
    Sigma_HF:        dict   = field(repr=False)   # {block: ndarray}
    density:         dict   = field(repr=False)   # {block: ndarray}
    free_energy:     float  = field(default=0.0)
    weight:          float  = field(default=0.0)
    magnetisation:   float  = field(default=0.0)
    n_total:         float  = field(default=0.0)
    _proposal_index: int    = field(default=0, repr=False)

    def total_density(self) -> float:
        """Total electron number ⟨N⟩ = Σ_σ Tr[ρ_σ]."""
        return float(sum(np.trace(rho) for rho in self.density.values()))

    def density_matrix_eigenvalues(self) -> dict:
        """Eigenvalues of each spin-block density matrix {block: 1-D ndarray}."""
        return {bl: np.linalg.eigvalsh(rho) for bl, rho in self.density.items()}

    def magnetisation_per_orbital(self) -> np.ndarray:
        """Per-orbital magnetisation  m_a = n↑_a − n↓_a."""
        norb = next(iter(self.density.values())).shape[0]
        return np.array([self.density['up'][a, a] - self.density['down'][a, a]
                         for a in range(norb)])


class SolutionSet:
    """Ordered collection of :class:`Solution` objects, sorted by free energy.

    Supports integer indexing, iteration, and a rich text summary via
    ``repr()`` / direct printing.  The full pandas DataFrame is available via
    :meth:`to_dataframe`.

    Examples
    --------
    >>> ensemble.solutions                       # pretty overview table
    >>> ensemble.solutions[0]                    # lowest-F Solution
    >>> ensemble.solutions.get_green_function(0) # G_iw BlockGf
    >>> ensemble.solutions.get_sigma(0)          # Sigma_HF dict
    >>> ensemble.solutions.get_free_energy(0)    # float
    >>> ensemble.solutions.get_density(0)        # density dict
    >>> ensemble.solutions.get_all_info(0)       # same as [0]
    >>> ensemble.solutions.to_dataframe()        # pandas DataFrame
    """

    def __init__(self, solutions: list):
        self._data = sorted(solutions, key=lambda s: s.free_energy)

    # ── Sequence interface ────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i) -> 'Solution':
        return self._data[i]

    def __iter__(self):
        return iter(self._data)

    # ── Pretty summary ────────────────────────────────────────────────────
    def __repr__(self) -> str:
        if not self._data:
            return 'SolutionSet(empty)'
        norb   = next(iter(self._data[0].density.values())).shape[0]
        blocks = list(self._data[0].density.keys())
        F_min  = self._data[0].free_energy

        # column widths
        W_i, W_F, W_m, W_N, W_w = 6, 14, 12, 9, 9
        ev_w = max(8, norb * 7)          # eigenvalue columns
        sep  = ('─' * W_i + '┼' + '─' * W_F + '┼' + '─' * W_m +
                '┼' + '─' * W_N + '┼' + '─' * W_w +
                '┼' + '─' * ev_w + '┼' + '─' * ev_w)
        hdr  = (f"{'#':>{W_i-1}} "
                f"│ {'F':>{W_F-2}} "
                f"│ {'m':>{W_m-2}} "
                f"│ {'N':>{W_N-2}} "
                f"│ {'weight':>{W_w-2}} "
                f"│ {'eigs ↑':>{ev_w-2}} "
                f"│ {'eigs ↓':>{ev_w-2}}")

        lines = [f'SolutionSet  ({len(self._data)} solutions,  '
                 f'ΔF_max = {self._data[-1].free_energy - F_min:.4f})',
                 hdr, sep]
        for i, sol in enumerate(self._data):
            eigs  = sol.density_matrix_eigenvalues()
            ev_up = '  '.join(f'{v:.3f}' for v in
                              sorted(eigs.get('up', next(iter(eigs.values())))))
            ev_dn = '  '.join(f'{v:.3f}' for v in
                              sorted(eigs.get('down', list(eigs.values())[-1])))
            tag = '*' if i == 0 else ' '
            lines.append(
                f"{i:>{W_i-2}}{tag} "
                f"│ {sol.free_energy:>{W_F-2}.6f} "
                f"│ {sol.magnetisation:>{W_m-2}.5f} "
                f"│ {sol.n_total:>{W_N-2}.4f} "
                f"│ {sol.weight:>{W_w-2}.5f} "
                f"│ {ev_up:>{ev_w-2}} "
                f"│ {ev_dn:>{ev_w-2}}"
            )
        lines.append(sep)
        lines.append('  (* lowest free energy)')
        return '\n'.join(lines)

    # ── Named accessors ──────────────────────────────────────────────────
    def get_green_function(self, i):
        """Converged G_iw (DLR BlockGf) for solution i."""
        return self._data[i].G_iw

    def get_sigma(self, i) -> dict:
        """Static HF self-energy {block: ndarray} for solution i."""
        return self._data[i].Sigma_HF

    def get_free_energy(self, i) -> float:
        """Impurity free energy of solution i."""
        return self._data[i].free_energy

    def get_weight(self, i) -> float:
        """Boltzmann weight of solution i."""
        return self._data[i].weight

    def get_density(self, i) -> dict:
        """Density matrix {block: norb×norb ndarray} for solution i."""
        return self._data[i].density

    def get_all_info(self, i) -> 'Solution':
        """Full :class:`Solution` object for solution i (same as ``[i]``)."""
        return self._data[i]

    # ── Array properties ─────────────────────────────────────────────────
    @property
    def free_energies(self) -> np.ndarray:
        """Free energies, sorted ascending."""
        return np.array([s.free_energy for s in self._data])

    @property
    def weights(self) -> np.ndarray:
        """Boltzmann weights."""
        return np.array([s.weight for s in self._data])

    @property
    def magnetisations(self) -> np.ndarray:
        """Magnetisations."""
        return np.array([s.magnetisation for s in self._data])

    # ── DataFrame export ─────────────────────────────────────────────────
    def to_dataframe(self) -> pd.DataFrame:
        """Export scalar observables to a pandas DataFrame.

        Columns: free_energy, magnetisation, n_total, weight,
        n_up_{a}, n_down_{a}, m_{a}, Sigma_up_{a}, Sigma_down_{a}
        for each orbital a.
        """
        if not self._data:
            return pd.DataFrame()
        blocks = list(self._data[0].density.keys())
        norb   = self._data[0].density[blocks[0]].shape[0]
        rows   = []
        for i, sol in enumerate(self._data):
            rho = sol.density
            sig = sol.Sigma_HF
            row = dict(
                free_energy   = sol.free_energy,
                magnetisation = sol.magnetisation,
                n_total       = sol.n_total,
                weight        = sol.weight,
            )
            for a in range(norb):
                row[f'n_up_{a}']       = float(rho['up'][a, a])   if 'up'   in rho else float('nan')
                row[f'n_down_{a}']     = float(rho['down'][a, a]) if 'down' in rho else float('nan')
                row[f'm_{a}']          = (float(rho['up'][a, a] - rho['down'][a, a])
                                          if 'up' in rho and 'down' in rho else float('nan'))
                row[f'Sigma_up_{a}']   = float(np.real(sig['up'][a, a]))   if 'up'   in sig else float('nan')
                row[f'Sigma_down_{a}'] = float(np.real(sig['down'][a, a])) if 'down' in sig else float('nan')
            rows.append(row)
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  Proposal generators
# ══════════════════════════════════════════════════════════════════════════════

class ProposalGenerator(ABC):
    """Abstract base class for initial-condition proposal generators.

    A proposal generator produces a list of Σ_HF initial guesses, each a dict
    mapping spin-block name → (norb × norb) real ndarray.
    """

    @abstractmethod
    def generate(
        self,
        G0_iw: BlockGf,
        n_elec_total: float,
        norb: int,
        gf_struct: list,
        seed: int = 0,
    ) -> list[dict[str, np.ndarray]]:
        """Return a list of Σ_HF initial guesses.

        Parameters
        ----------
        G0_iw       : converged bath Green's function (DLR mesh)
        n_elec_total: total electron number ⟨N⟩ (used to fix the up/down split)
        norb        : number of orbitals per spin
        gf_struct   : [(block_name, size), …]
        seed        : integer seed for reproducible sampling
        """


class SobolSigmaProposals(ProposalGenerator):
    """Propose Σ_HF initial conditions via a Sobol low-discrepancy sequence.

    The self-energy is sampled uniformly in the space of real-symmetric
    (norb × norb) matrices, with configurable bounds on diagonal and
    off-diagonal elements.

    Parameters
    ----------
    n_samples          : number of Sobol points
    sigma_bound        : half-width of the diagonal sampling range
                         (centred on sigma_offset when provided, else on 0)
    sigma_offset       : centre of the diagonal range  (e.g. the chemical
                         potential μ from the converged DMFT run)
    offdiag_fraction   : off-diagonal range as a fraction of sigma_bound
    """

    def __init__(
        self,
        n_samples: int = 300,
        sigma_bound: float = 3.0,
        sigma_offset: float = 0.0,
        offdiag_fraction: float = 0.2,
    ):
        self.n_samples = n_samples
        self.sigma_bound = sigma_bound
        self.sigma_offset = sigma_offset
        self.offdiag_fraction = offdiag_fraction

    # ------------------------------------------------------------------
    def generate(self, G0_iw, n_elec_total, norb, gf_struct, seed=0):
        blocks = [bl for bl, _ in gf_struct]
        n_per_block = norb * (norb + 1) // 2   # upper-triangle + diagonal
        n_params    = len(blocks) * n_per_block

        diag_lo = self.sigma_offset - self.sigma_bound
        diag_hi = self.sigma_offset + self.sigma_bound
        od_bound = self.sigma_bound * self.offdiag_fraction
        n_offdiag = n_per_block - norb

        lo_block = [diag_lo] * norb + [-od_bound] * n_offdiag
        hi_block = [diag_hi] * norb + [ od_bound] * n_offdiag
        lo_all   = lo_block * len(blocks)
        hi_all   = hi_block * len(blocks)

        sampler = qmc.Sobol(d=n_params, scramble=True, seed=seed)
        raw     = sampler.random(self.n_samples)
        samples = qmc.scale(raw, l_bounds=lo_all, u_bounds=hi_all)

        proposals = []
        for flat in samples:
            sigma_init = {}
            offset = 0
            for bl, _ in gf_struct:
                sigma_init[bl] = _flat_to_symmetric(flat[offset:offset + n_per_block], norb)
                offset += n_per_block
            proposals.append(sigma_init)
        return proposals

    def __repr__(self):
        return (f"SobolSigmaProposals(n_samples={self.n_samples}, "
                f"sigma_bound={self.sigma_bound}, offset={self.sigma_offset})")


class DensityMatrixProposals(ProposalGenerator):
    """Propose Σ_HF via density-matrix targeting.

    For each proposal:
      1. Draw a random target density matrix per spin block: diagonal entries
         from {0, 0.5, 1} (integer-biased, ``half_occ_prob`` controls the
         fraction of 0.5 entries), rotated by a random SO(norb) matrix.
         The up/down split of n_elec_total is drawn uniformly from all
         integer splits in [0, norb] – this naturally covers PM, FM, AFM.
      2. Run ``n_targeting_steps`` gradient-ascent updates
             Σ ← Σ + α · (ρ[Σ] − ρ_target)
         to drive the impurity density toward the target.
      3. Return the resulting Σ_HF as the initial condition for the full HF
         self-consistency.

    Parameters
    ----------
    n_proposals       : number of random targets (total, including recycled ones
                        and custom targets).
    n_targeting_steps : gradient-ascent iterations before release
    targeting_alpha   : step size α
    half_occ_prob     : per-orbital probability of occupation 0.5
                        (rest are integer 0 or 1)
    force_real        : strip imaginary parts from Σ during targeting
    prior_solutions   : optional :class:`SolutionSet` (or list of
                        :class:`Solution`) from a previous ensemble solve.
                        The first ``min(len(prior_solutions), n_proposals)``
                        proposals will reuse those Σ_HF directly; the
                        remaining slots are filled with new random targets.
                        Update ``dm_proposals.prior_solutions`` between DMFT
                        iterations to warm-start successive ensemble solves.
    custom_proposals  : optional list of explicit density-matrix targets.
                        Each entry is a dict mapping block name → (norb × norb)
                        real ndarray (a target density matrix, *not* a Σ_HF).
                        These are processed through the targeting loop first,
                        before any recycled or random proposals.  They count
                        toward ``n_proposals``.
                        Example::

                            custom_proposals = [
                                {'up': np.diag([1,1,0]), 'down': np.diag([0,0,0])},
                            ]
    """

    def __init__(
        self,
        n_proposals: int = 200,
        n_targeting_steps: int = 30,
        targeting_alpha: float = 2.0,
        half_occ_prob: float = 0.10,
        force_real: bool = True,
        prior_solutions=None,
        custom_proposals=None,
    ):
        self.n_proposals       = n_proposals
        self.n_targeting_steps = n_targeting_steps
        self.targeting_alpha   = targeting_alpha
        self.half_occ_prob     = half_occ_prob
        self.force_real        = force_real
        self.prior_solutions   = prior_solutions   # SolutionSet | list[Solution] | None
        self.custom_proposals  = custom_proposals  # list[{bl: rho_target}] | None

    # ------------------------------------------------------------------
    def generate(self, G0_iw, n_elec_total, norb, gf_struct, seed=0):
        rng    = np.random.default_rng(seed)
        blocks = [bl for bl, _ in gf_struct]   # e.g. ['up', 'down']

        proposals = []

        # ── 0a. Custom density-matrix targets (run through targeting loop) ──
        if self.custom_proposals is not None:
            for rho_target in self.custom_proposals:
                if len(proposals) >= self.n_proposals:
                    break
                dtype = float if self.force_real else complex
                sigma = {bl: np.zeros((norb, norb), dtype=dtype) for bl in blocks}
                for _ in range(self.n_targeting_steps):
                    sigma, _ = _targeting_step(G0_iw, sigma, rho_target,
                                               self.targeting_alpha, self.force_real)
                proposals.append(sigma)

        # ── 0b. Recycle Σ_HF from prior solutions ────────────────────────
        if self.prior_solutions is not None:
            prior_list = list(self.prior_solutions)   # works for SolutionSet or list
            n_recycle  = min(len(prior_list), max(0, self.n_proposals - len(proposals)))
            for sol in prior_list[:n_recycle]:
                sigma_recycled = {
                    bl: (np.array(sol.Sigma_HF[bl]).real.astype(float)
                         if self.force_real
                         else np.array(sol.Sigma_HF[bl]).astype(complex))
                    for bl in blocks
                }
                proposals.append(sigma_recycled)
        n_new = max(0, self.n_proposals - len(proposals))

        # ── 1+2. Random DM targets + targeting loop ───────────────────────
        n_up_min = max(0, int(round(n_elec_total)) - norb)
        n_up_max = min(norb, int(round(n_elec_total)))

        for _ in range(n_new):
            n_up   = int(rng.integers(n_up_min, n_up_max + 1))
            n_down = int(round(n_elec_total)) - n_up

            n_per_block = {'up': n_up, 'down': n_down}

            rho_target = {
                bl: _make_target_density_matrix(n_per_block[bl], norb, rng,
                                                self.half_occ_prob)
                for bl in blocks
            }

            dtype = float if self.force_real else complex
            sigma = {bl: np.zeros((norb, norb), dtype=dtype) for bl in blocks}

            for _ in range(self.n_targeting_steps):
                sigma, _ = _targeting_step(G0_iw, sigma, rho_target,
                                           self.targeting_alpha, self.force_real)

            proposals.append(sigma)

        return proposals

    def __repr__(self):
        n_prior  = len(self.prior_solutions)  if self.prior_solutions  is not None else 0
        n_custom = len(self.custom_proposals) if self.custom_proposals is not None else 0
        return (f"DensityMatrixProposals(n_proposals={self.n_proposals}, "
                f"n_steps={self.n_targeting_steps}, alpha={self.targeting_alpha}, "
                f"half_occ_prob={self.half_occ_prob}, n_prior={n_prior}, "
                f"n_custom={n_custom})")


# ══════════════════════════════════════════════════════════════════════════════
#  Main ensemble solver
# ══════════════════════════════════════════════════════════════════════════════

class IncoherentEnsembleSolver:
    """Incoherent ensemble impurity solver.

    Discovers multiple HF saddle points of the impurity problem for a fixed
    converged bath G0, Boltzmann-weights them by their impurity free energy,
    and assembles an ensemble-averaged Green's function and self-energy.

    Parameters
    ----------
    gf_struct            : list of (block_name, size) pairs
    beta                 : inverse temperature
    w_max                : DLR energy cutoff
    eps                  : DLR accuracy
    dc_fixed_value       : double-counting shift (passed to each ImpuritySolver).
                           Use 0 for no DC, or a positive number for a fixed shift.
    force_real           : keep Σ_HF and density matrices real throughout
    enforce_paramagnetic : if True (default), append a spin-flipped copy of every
                           converged solution after the scan.  The bath G0 has
                           up↔down symmetry for a PM background, so the partner
                           has identical free energy and is physically distinct.
    prune_tol            : after augmentation, remove solutions whose G_iw is
                           within this Frobenius-norm distance of an earlier one
                           (lower free energy wins).  Set to 0 to disable.
    verbosity            : 'quiet' – no output;
                           'normal' (default) – progress bar and summary;
                           'debug' – also show inner HF-solver output.

    After ``solve()``:

    Attributes
    ----------
    solutions     : pd.DataFrame
                    One row per converged saddle point.  Columns include
                    ``free_energy``, ``magnetisation``, ``n_total``,
                    ``weight``, per-spin-orbital occupancies ``n_up_{a}``,
                    ``n_down_{a}``, and diagonal self-energies
                    ``Sigma_up_{a}``, ``Sigma_down_{a}``.
    G_iw          : BlockGf  –  ensemble-averaged G(iω)
    Sigma_iw      : BlockGf  –  many-body Σ(iω) = G0⁻¹ − G_ens⁻¹
    weights       : np.ndarray  –  Boltzmann weights (sums to 1)
    free_energies : np.ndarray  –  raw free energies of each saddle point
    n_converged   : int
    n_failed      : int
    """

    def __init__(
        self,
        gf_struct: list,
        beta: float,
        w_max: float,
        eps: float,
        dc_fixed_value: float = 0.0,
        force_real: bool = True,
        enforce_paramagnetic: bool = True,
        prune_tol: float = 0.1,
        verbosity: str = 'normal',
    ):
        self.gf_struct            = gf_struct
        self.beta                 = beta
        self.beta_eff             = beta # to weigh free energy differences for Boltzmann factors (can be scaled down to flatten weights)
        self.w_max                = w_max
        self.eps                  = eps
        self.dc_fixed_value       = dc_fixed_value
        self.force_real           = force_real
        self.enforce_paramagnetic = enforce_paramagnetic
        self.prune_tol            = prune_tol
        self._verbosity           = verbosity
        self.norb                 = gf_struct[0][1]
        self.blocks               = [bl for bl, _ in gf_struct]

        mesh       = MeshDLRImFreq(beta, 'Fermion', w_max, eps, symmetrize=True)
        name_list  = [bl for bl, _ in gf_struct]
        block_list = [Gf(mesh=mesh, target_shape=[sz, sz]) for _, sz in gf_struct]
        self.G0_iw = BlockGf(name_list=name_list, block_list=block_list)

        # initialise output attributes to None – populated by solve()
        self.solutions     = None
        self.G_iw          = None
        self.Sigma_iw      = None
        self.weights       = None
        self.free_energies = None
        self.n_converged   = 0
        self.n_failed      = 0

    # ------------------------------------------------------------------
    def solve(
        self,
        h_int,
        proposal_generators: list[ProposalGenerator],
        mu: float,
        n_elec_total: float,
        with_fock: bool = True,
        hf_method: str = 'hybr',
        hf_tol: float = 1e-8,
        seed: int = 42,
    ):
        """Run the ensemble discovery and assembly.

        Set ``self.G0_iw`` before calling this method (same convention as
        :class:`ImpuritySolver`).

        Parameters
        ----------
        h_int              : TRIQS interaction Hamiltonian
        proposal_generators: list of ProposalGenerator instances
        mu                 : chemical potential (for free-energy calculation)
        n_elec_total       : total electron number (for DM proposals split)
        with_fock          : include Fock terms in each HF solve
        hf_method          : scipy.optimize.root method for HF self-consistency
        hf_tol             : HF root-finder tolerance
        seed               : base random seed (each generator gets seed+i)
        """

        self._report('\n' + '═' * 70)
        self._report('  IncoherentEnsembleSolver.solve()')
        self._report(f'  β = {self.beta:.1f},   β_eff = {self.beta_eff:.1f},   norb = {self.norb}   '
                     f'n_target = {n_elec_total:.2f}   force_real = {self.force_real}')
        for i, gen in enumerate(proposal_generators):
            self._report(f'  Generator {i}: {gen!r}')
        self._report('═' * 70 + '\n')

        # ── Collect all proposals from all generators ──────────────────────
        all_proposals = []
        for i, gen in enumerate(proposal_generators):
            proposals = gen.generate(self.G0_iw, n_elec_total, self.norb,
                                     self.gf_struct, seed=seed + i * 100)
            all_proposals.extend(proposals)
            self._report(f'  {len(proposals):4d} proposals from {gen!r}')

        n_total = len(all_proposals)
        self._report(f'\n  Total proposals: {n_total}\n')

        # ── Run HF solve for each proposal ────────────────────────────────
        records       = []
        green_funcs   = []      # list of BlockGf for converged solutions
        sigma_hfs     = []      # list of {block: ndarray} static HF self-energies
        self.n_converged = 0
        self.n_failed    = 0

        progress_every = max(1, n_total // 10)

        for idx, sigma_init in enumerate(all_proposals):
            solver = ImpuritySolver(
                gf_struct      = self.gf_struct,
                beta           = self.beta,
                w_max          = self.w_max,
                eps            = self.eps,
                dc             = 'cFLL',
                force_real     = self.force_real,
            )

            for bl in self.blocks:
                solver.G0_iw[bl].data[:] = self.G0_iw[bl].data
                solver.Sigma_HF[bl] = (
                    sigma_init[bl].real.astype(float)
                    if self.force_real
                    else sigma_init[bl].astype(complex)
                )

            solver.dc_fixed_value = self.dc_fixed_value
            _ctx = (_suppress_hf_output()
                    if self._verbosity != 'debug'
                    else contextlib.nullcontext())
            with _ctx:
                solver.solve(
                    h_int,
                    with_fock = with_fock,
                    one_shot  = False,
                    method    = hf_method,
                    tol       = hf_tol,
                )

            if not solver.hf_converged:
                self.n_failed += 1
            else:
                rec = self._collect_observables(solver, mu, idx)
                records.append(rec)
                green_funcs.append(solver.G_iw.copy())
                sigma_hfs.append(
                    {bl: np.array(solver.Sigma_HF[bl]) for bl in self.blocks})
                self.n_converged += 1

            if (idx + 1) % progress_every == 0 or (idx + 1) == n_total:
                frac   = (idx + 1) / n_total
                filled = int(frac * 30)
                bar    = '█' * filled + '░' * (30 - filled)
                self._report(f'  [{bar}]  {idx+1:4d}/{n_total}'
                             f'  ✓ {self.n_converged}  ✗ {self.n_failed}')

        self._report(f'\n  Converged: {self.n_converged}  '
                     f'Failed: {self.n_failed}  Total: {n_total}')

        if self.n_converged == 0:
            raise RuntimeError('No saddle points converged.  '
                               'Try wider sigma_bound or more proposals.')

        # ── Spin-flip augmentation (enforce_paramagnetic) ─────────────────
        if self.enforce_paramagnetic and set(self.blocks) == {'up', 'down'}:
            n_before_flip = len(records)
            flip_rec, flip_gfs, flip_sigs = self._make_spin_flipped(
                records, green_funcs, sigma_hfs)
            records.extend(flip_rec)
            green_funcs.extend(flip_gfs)
            sigma_hfs.extend(flip_sigs)
            self._report(f'\n  enforce_paramagnetic: added {len(flip_rec)} '
                         f'spin-flipped partners '
                         f'({n_before_flip} → {len(records)} solutions)')

        # ── Sort by free energy before pruning ────────────────────────────
        order       = np.argsort([r['free_energy'] for r in records])
        records     = [records[i]     for i in order]
        green_funcs = [green_funcs[i] for i in order]
        sigma_hfs   = [sigma_hfs[i]   for i in order]

        # ── Prune near-duplicate solutions ────────────────────────────────
        if self.prune_tol > 0.0:
            n_before_prune = len(records)
            records, green_funcs, sigma_hfs = \
                self._prune_solutions(records, green_funcs, sigma_hfs)
            self._report(f'  Pruning (tol={self.prune_tol}): '
                         f'{n_before_prune} → {len(records)} unique solutions '
                         f'({n_before_prune - len(records)} removed)')

        self.n_converged = len(records)

        # ── Boltzmann weights ──────────────────────────────────────────────


        F_arr    = np.array([r['free_energy'] for r in records])
        dF       = F_arr - F_arr.min()
        w_unnorm = np.exp(-self.beta_eff * dF)
        weights  = w_unnorm / w_unnorm.sum()

        self._report('\n  Free-energy range:  '
                     f'min = {F_arr.min():.6f}   max = {F_arr.max():.6f}   '
                     f'ΔF = {F_arr.max()-F_arr.min():.6f}')

        # ── Build SolutionSet ──────────────────────────────────────────────
        self.solutions = SolutionSet([
            Solution(
                G_iw            = gf,
                Sigma_HF        = sig,
                density         = rec['density'],
                free_energy     = rec['free_energy'],
                weight          = w,
                magnetisation   = rec['magnetisation'],
                n_total         = rec['n_total'],
                _proposal_index = rec['proposal_index'],
            )
            for rec, gf, sig, w in zip(records, green_funcs, sigma_hfs, weights)
        ])                                    # SolutionSet sorts by F internally
        self.weights       = self.solutions.weights
        self.free_energies = self.solutions.free_energies

        # ── Ensemble-averaged G(iω) ───────────────────────────────────────
        self.G_iw = self.G0_iw.copy()
        for bl in self.blocks:
            self.G_iw[bl].data[:] = 0.0 + 0.0j

        for sol in self.solutions:
            for bl in self.blocks:
                self.G_iw[bl].data[:] += sol.weight * sol.G_iw[bl].data

        # ── Many-body Σ(iω) = G0⁻¹ − G_ens⁻¹ ────────────────────────────
        self.Sigma_iw = self.G0_iw.copy()
        for bl in self.blocks:
            self.Sigma_iw[bl] << inverse(self.G0_iw[bl]) - inverse(self.G_iw[bl])

        self._report('\n  Ensemble assembled successfully.\n')
        self._report('═' * 70 + '\n')

        mpi.report(f'Unique saddle points found: {self.n_converged}')
        mpi.report('\n' + repr(self.solutions))
        self._report('═' * 70 + '\n')


    # ------------------------------------------------------------------
    def _collect_observables(self, solver: ImpuritySolver, mu: float, idx: int) -> dict:
        """Extract observables from a converged ImpuritySolver."""
        rho   = {bl: solver.G_iw[bl].density().real for bl in self.blocks}
        n_tot = float(sum(np.trace(rho[bl]) for bl in self.blocks))

        m_per_orb = np.array([
            rho['up'][a, a] - rho['down'][a, a]
            for a in range(self.norb)
        ])
        magnetisation = float(np.sum(m_per_orb))

        free_energy = _impurity_free_energy(solver)

        rec = {
            'proposal_index': idx,
            'free_energy'   : free_energy,
            'magnetisation' : magnetisation,
            'n_total'       : n_tot,
            'weight'        : 0.0,   # filled in after all solutions known
        }
        rec['density'] = rho
        return rec

    # ------------------------------------------------------------------
    def _report(self, msg: str) -> None:
        """Emit msg via mpi.report unless verbosity is 'quiet'."""
        if self._verbosity != 'quiet':
            mpi.report(msg)

    # ------------------------------------------------------------------
    def _make_spin_flipped(self, records, green_funcs, sigma_hfs):
        """Create spin-flipped (up ↔ down) partners for every solution.

        The paramagnetic bath has up/down symmetry, so each converged solution
        has a degenerate partner obtained by swapping the spin blocks.  The
        free energy is unchanged; all observables are recomputed from the
        flipped Green's function.
        """
        flip = {'up': 'down', 'down': 'up'}
        flipped_records = []
        flipped_gfs     = []
        flipped_sigmas  = []

        for rec, gf, sig in zip(records, green_funcs, sigma_hfs):
            # ── flip G ────────────────────────────────────────────────────────────
            gf_flip = gf.copy()
            for bl in self.blocks:
                gf_flip[bl].data[:] = gf[flip[bl]].data

            # ── flip static self-energy ────────────────────────────────────────
            sig_flip = {bl: sig[flip[bl]].copy() for bl in self.blocks}

            # ── recompute density observables from flipped G ───────────────────
            rho_flip  = {bl: gf_flip[bl].density().real for bl in self.blocks}
            m_per_orb = np.array([
                rho_flip['up'][a, a] - rho_flip['down'][a, a]
                for a in range(self.norb)
            ])

            rec_flip = dict(rec)
            # Negative proposal_index: unique key, distinct from originals
            rec_flip['proposal_index'] = -(rec['proposal_index'] + 1)
            rec_flip['magnetisation']  = float(np.sum(m_per_orb))
            rec_flip['n_total']        = float(
                sum(np.trace(rho_flip[bl]) for bl in self.blocks))
            for a in range(self.norb):
                rec_flip[f'n_up_{a}']       = float(rho_flip['up'][a, a])
                rec_flip[f'n_down_{a}']     = float(rho_flip['down'][a, a])
                rec_flip[f'Sigma_up_{a}']   = float(np.real(sig_flip['up'][a, a]))
                rec_flip[f'Sigma_down_{a}'] = float(np.real(sig_flip['down'][a, a]))
                rec_flip[f'm_{a}']          = float(m_per_orb[a])
            # free_energy is unchanged by symmetry
            rec_flip['density'] = rho_flip

            flipped_records.append(rec_flip)
            flipped_gfs.append(gf_flip)
            flipped_sigmas.append(sig_flip)

        return flipped_records, flipped_gfs, flipped_sigmas

    # ------------------------------------------------------------------
    def _prune_solutions(self, records, green_funcs, sigma_hfs):
        """Greedy O(n²) pruning: discard solutions within ``prune_tol`` of a
        lower-energy one.  Input must be sorted ascending by free energy.
        """
        n    = len(records)
        keep = [True] * n
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(i + 1, n):
                if not keep[j]:
                    continue
                if self._gf_distance(green_funcs[i], green_funcs[j]) < self.prune_tol:
                    keep[j] = False
        return (
            [r for r, k in zip(records,     keep) if k],
            [g for g, k in zip(green_funcs, keep) if k],
            [s for s, k in zip(sigma_hfs,   keep) if k],
        )

    # ------------------------------------------------------------------
    def _gf_distance(self, gf1, gf2) -> float:
        """Sum of Frobenius norms ‖gf1[bl] − gf2[bl]‖_F over all spin blocks."""
        total = 0.0
        for bl in self.blocks:
            diff   = gf1[bl].data - gf2[bl].data
            total += float(np.linalg.norm(diff))
        return total

    # ------------------------------------------------------------------
    def plot_landscape(
        self,
        fig=None,
        show: bool = True,
        save_path: str | None = None,
        fontsize_label: float = 13,
        fontsize_title: float = 14,
        fontsize_legend: float = 11,
        
    ):
        """Plot the discovered HF state landscape.

        Produces a two-panel figure:
          Left  – scatter of magnetisation vs ΔF, coloured by total charge.
          Right – histogram of ΔF.

        Parameters
        ----------
        fig        : existing matplotlib Figure to draw into (optional)
        show       : call plt.show() when done
        save_path  : if given, save the figure to this path (e.g. 'out.jpg')
        """
        import matplotlib.pyplot as plt

        if self.solutions is None:
            raise RuntimeError('Call solve() before plot_landscape().')

        F     = self.solutions.free_energies
        dF    = F - F.min()
        m     = self.solutions.magnetisations
        n_tot = np.array([s.n_total for s in self.solutions])

        if fig is None:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        else:
            axes = fig.axes

        n_conv   = self.n_converged
        n_total  = self.n_converged + self.n_failed

        fig.suptitle(
            f'HF saddle-point landscape   |   '
            f'β={self.beta:.0f}   norb={self.norb}   '
            f'β_eff={self.beta_eff:.0f}   norb={self.norb}   '
            f'{n_conv}/{n_total} converged',
            fontsize=fontsize_title + 1, y=1.02
        )

        # ── Left: magnetisation vs ΔF ──────────────────────────────────────
        ax = axes[0]
        vmin = max(0.0, n_tot.mean() - 2 * n_tot.std())
        vmax =          n_tot.mean() + 2 * n_tot.std()

        sc = ax.scatter(
            m, dF,
            c=n_tot, cmap='RdBu_r', vmin=vmin, vmax=vmax,
            s=50, alpha=0.78, edgecolors='k', linewidths=0.8, zorder=3,
        )
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label('Total charge  n', fontsize=fontsize_legend)

        # mark lowest-F solution
        best = self.solutions[0]
        ax.scatter([best.magnetisation], [0.0],
                   marker='*', s=300, color='none', edgecolors='gold',
                   linewidths=1.5, zorder=5,
                   label=f'lowest F  (m = {best.magnetisation:+.3f})')

        ax.axvline(0, color='gray', lw=0.9, ls='--', alpha=0.5)
        ax.set_xlabel(r'Magnetisation  $m = \langle n_\uparrow\rangle - \langle n_\downarrow\rangle$',
                      fontsize=fontsize_label)
        ax.set_ylabel(r'$\Delta F = F - F_{\min}$', fontsize=fontsize_label)
        ax.set_title('Discovered saddle points', fontsize=fontsize_title)
        ax.legend(fontsize=fontsize_legend)
        ax.grid(True, alpha=0.3, lw=0.5)

        # ── Right: histogram of ΔF ─────────────────────────────────────────
        ax = axes[1]
        ax.hist(dF, bins=50, color='steelblue', edgecolor='k',
                linewidth=0.35, alpha=0.82)
        ax.axvline(0, color='crimson', lw=1.2, ls='--', label='lowest F')
        ax.axvline(dF.mean(), color='darkorange', lw=1.0, ls=':',
                   label=f'mean  ΔF = {dF.mean():.4f}')
        ax.set_xlabel(r'$\Delta F = F - F_{\min}$', fontsize=fontsize_label)
        ax.set_ylabel('Count', fontsize=fontsize_label)
        ax.set_title('Free-energy distribution', fontsize=fontsize_title)
        ax.legend(fontsize=fontsize_legend)
        ax.grid(True, alpha=0.3, lw=0.5)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helper functions  (not part of public API)
# ══════════════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def _suppress_hf_output():
    """Silence all output from the HF solver.

    Strategy (in order of reliability):
    1. Monkey-patch ``triqs.utility.mpi.report`` to a no-op – this catches all
       TRIQS-level reporting regardless of how the underlying stream is wired.
    2. Simultaneously redirect both ``sys.stdout`` / ``sys.stderr`` and, when
       accessible, the raw file descriptors (fd 1 / fd 2) via ``os.dup2`` so
       that any C-extension writes to the real fds are also swallowed.
    All three layers are restored unconditionally in the ``finally`` block.
    """
    import triqs.utility.mpi as _mpi

    # ── 1. Patch mpi.report ───────────────────────────────────────────────
    _orig_report = _mpi.report
    _mpi.report  = lambda *args, **kwargs: None   # noqa: E731

    # ── 2. Python-level stdout/stderr redirect ────────────────────────────
    _buf = io.StringIO()

    # ── 3. fd-level redirect (best-effort) ───────────────────────────────
    _use_fd = False
    _old_out = _old_err = _devnull = None
    try:
        _fd_out  = sys.stdout.fileno()
        _fd_err  = sys.stderr.fileno()
        _use_fd  = True
    except (AttributeError, io.UnsupportedOperation):
        pass

    if _use_fd:
        _devnull = os.open(os.devnull, os.O_WRONLY)
        _old_out = os.dup(_fd_out)
        _old_err = os.dup(_fd_err)
        os.dup2(_devnull, _fd_out)
        os.dup2(_devnull, _fd_err)
        os.close(_devnull)

    try:
        with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
            yield
    finally:
        # restore fd-level
        if _use_fd:
            os.dup2(_old_out, _fd_out)
            os.dup2(_old_err, _fd_err)
            os.close(_old_out)
            os.close(_old_err)
        # restore mpi.report
        _mpi.report = _orig_report


def _flat_to_symmetric(v: np.ndarray, norb: int) -> np.ndarray:
    """Build a real symmetric (norb × norb) matrix from norb*(norb+1)//2 values.

    Layout: [diag_0, …, diag_{n-1}, upper_{01}, upper_{02}, …]
    """
    m = np.zeros((norb, norb), dtype=float)
    for a in range(norb):
        m[a, a] = v[a]
    idx = norb
    for a in range(norb):
        for b in range(a + 1, norb):
            m[a, b] = v[idx]
            m[b, a] = v[idx]
            idx += 1
    return m


def _make_diagonal_density_matrix(
    n_elec_spin: float,
    norb: int,
    rng: np.random.Generator,
    half_occ_prob: float = 0.10,
) -> np.ndarray:
    """Diagonal density matrix with entries in {0, 0.5, 1}.

    Each orbital independently draws Bernoulli(half_occ_prob) to receive
    occupation 0.5.  The remaining orbitals are filled greedily with 1s then
    0s to match the target charge n_elec_spin.  The diagonal is then shuffled
    so the subsequent SO(N) rotation acts on a random permutation.
    """
    eigs      = np.zeros(norb)
    half_mask = rng.random(norb) < half_occ_prob
    half_idx  = np.where(half_mask)[0]
    free_idx  = np.where(~half_mask)[0]

    eigs[half_idx] = 0.5

    n_free = int(round(n_elec_spin - 0.5 * len(half_idx)))
    n_free = max(0, min(n_free, len(free_idx)))

    rng.shuffle(free_idx)
    eigs[free_idx[:n_free]] = 1.0   # rest stay 0

    rng.shuffle(eigs)
    return np.diag(eigs)


def _make_target_density_matrix(
    n_elec_spin: float,
    norb: int,
    rng: np.random.Generator,
    half_occ_prob: float = 0.10,
) -> np.ndarray:
    """Real symmetric density matrix: diagonal {0,0.5,1} rotated by SO(norb)."""
    diag = _make_diagonal_density_matrix(n_elec_spin, norb, rng, half_occ_prob)
    if norb > 1:
        R = ortho_group.rvs(norb, random_state=rng)
        return R @ diag @ R.T
    return diag


def _targeting_step(
    G0_iw: BlockGf,
    sigma: dict[str, np.ndarray],
    rho_target: dict[str, np.ndarray],
    alpha: float,
    force_real: bool = True,
) -> tuple[dict, dict]:
    """One gradient-ascent step driving ρ[Σ] toward ρ_target.

    Σ ← Σ + α · (ρ[Σ] − ρ_target)

    The sign is correct because raising Σ lowers occupation.

    Returns updated sigma dict and current density dict.
    """
    sigma_new = {}
    rho_cur   = {}
    for bl, G0_bl in G0_iw:
        G_bl         = inverse(inverse(G0_bl) - sigma[bl])
        rho_cur[bl]  = G_bl.density().real
        delta        = alpha * (rho_cur[bl] - rho_target[bl])
        s_new        = sigma[bl] + (delta.real if force_real else delta)
        sigma_new[bl] = s_new.real.astype(float) if force_real else s_new
    return sigma_new, rho_cur


# ── backward-compat alias (remove in a future version) ──────────────────────
SaddlePoint = Solution


def _impurity_free_energy(solver: ImpuritySolver) -> float:
    """Impurity free energy for a converged HF solution.

    F = E_int + E_kin

    E_int = ½ Tr[Σ_int · ρ]   (exact HF interaction energy)
    E_kin = (1/β) Σ_{iω} Tr[ ln(iω · G(iω)) ]

    The kinetic term is computed on the DLR mesh by applying
    scipy.linalg.logm frequency-by-frequency to the product iω·G(iω).
    The sum is approximated by the DLR quadrature via total_density().

    At fixed G0, only differences ΔF between saddle points enter the
    Boltzmann weights, so any additive constant is irrelevant.
    """
    E_int = solver.interaction_energy()

    G_iw    = solver.G_iw
    prodlog = G_iw.copy()

    # build iω · G(iω) then take matrix log
    iomega = G_iw.copy()
    for bl in ['up', 'down']:
        iomega[bl] << iOmega_n + 0

    prod = iomega * G_iw
    for bl in ['up', 'down']:
        for i in range(len(prod[bl].mesh)):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                prodlog[bl].data[i] = logm(prod[bl].data[i])

    E_kin = float(np.real(prodlog.total_density()))

    return float(np.real(E_int + E_kin))
