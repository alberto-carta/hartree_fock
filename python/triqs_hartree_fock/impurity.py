# Copyright (c) 2022 Simons Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You may obtain a copy of the License at
#     https:#www.gnu.org/licenses/gpl-3.0.txt
#
# Authors: Jonathan Karp, Alexander Hampel, Alberto Carta, Nils Wentzell, Hugo U. R. Strand, Olivier Parcollet

import os
import contextlib
import numpy as np
from scipy.optimize import root
from triqs.gf import *
from triqs.gf.meshes import MeshDLRImFreq
import triqs.utility.mpi as mpi
from h5.formats import register_class
from .utils import logo, flatten, unflatten, compute_DC_from_density


class ImpuritySolver(object):

    """ Hartree-Fock Impurity solver. The solver provides a constant sigma which rigidly shifts the local orbital levels
        The solver is always initialized without double counting and it is left to the user to provide the relevant dc values
        before calling the solve() method.

    Parameters
    ----------

    gf_struct : list of pairs [ (str,int), ...]
        Structure of the Green's functions. It must be a
        list of pairs, each containing the name of the
        Green's function block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.

    beta : float
        inverse temperature
    
    dc_U : float
        Hubbard U parameter used in the double counting computation
    
    dc_J : float
        J parameter used in the double counting computation

    w_max : float
        DLR energy cutoff. Controls the frequency range of the representation.
        Typical values: 1-100. Larger values needed for sharply peaked spectral functions.

    eps : float
        DLR accuracy/precision. Controls the number of basis functions.
        Typical values: 1e-6 to 1e-14. Smaller values give higher precision but more basis functions.

    symmeties : optional, list of functions
        symmetry functions acting on self energy at each consistent step

    force_real : optional, bool
        True if the self energy is forced to be real

    """

    def __init__(self, gf_struct, beta, w_max, eps, dc=False, dc_U=0.0, dc_J=0.0, dc_type='cFLL', symmetries=[], force_real=False):

        self.gf_struct = gf_struct
        self.beta = beta
        self.n_orb = gf_struct[0][1]
        self.w_max = w_max
        self.eps = eps
        self.symmetries = symmetries
        self.force_real = force_real
        self.E_dc =0.0
        
        #defaults for DC relevant values
        self.dc = dc          # whether to compute dc value
        self.dc_type = dc_type   # method used for dc_calculation
        self.dc_U = dc_U
        self.dc_J = dc_J
        self.dc_factor = 1.0

        self.dc_fixed_occ = None
        self.dc_fixed_value = None

        # True if the last self-consistent solve converged; False if the root
        # finder reported failure (solution may still be used as a warm start)
        self.hf_converged = False

        # Here Sigma_HF gets initialized to numerical zeros
        # If you want to change this guess, use the method
        # reinitialize_sigma before calling the solve() method
        self.Sigma_HF = self._make_zero_sigma()
        self.Sigma_DC = self._make_zero_sigma()
        self.Sigma_int = self._make_zero_sigma()

        mesh = MeshDLRImFreq(beta, 'Fermion', w_max, eps, symmetrize=True)
        name_list = [bl_name for bl_name, _ in self.gf_struct]
        block_list = [Gf(mesh=mesh, target_shape=[bl_size, bl_size]) for _, bl_size in self.gf_struct]
        self.G0_iw = BlockGf(name_list=name_list, block_list=block_list)
        self.G_iw = self.G0_iw.copy()

        self.git_hash = "@PROJECT_GIT_HASH@"

    def _make_zero_sigma(self):
        """Create a zero-initialized sigma dictionary."""
        dtype = float if self.force_real else complex
        return {bl: np.zeros((bl_size, bl_size), dtype=dtype) for bl, bl_size in self.gf_struct}

    def solve(self, h_int, with_fock=True, one_shot=True, method='krylov', tol=1e-5):
        """ Solve for the Hartree Fock self energy using a root finder method.
        The self energy is stored in the ``Sigma_HF`` object of the ImpuritySolver instance.
        The Green's function is stored in the ``G_iw`` object of the ImpuritySolver instance.

        Parameters
        ----------

        h_int : TRIQS Operator instance
            Local interaction Hamiltonian

        with_fock : optional, bool
            True if the fock terms are included in the self energy. Default is True

        one_shot : optional, bool
            True if the calcualtion is just one shot and not self consistent. Default is False

        method : optional, string
            method for root finder. Only used if one_shot=False, see scipy.optimize.root
            for options. Default : krylov

        tol : optional, float
            tolerance for root finder if one_shot=False. default 1e-5


        """
        self.last_solve_params = {key: val for key, val in locals().items() if key != 'self'}
        mpi.report(logo())
        mpi.report('Running Impurity Solver')
        mpi.report('beta = %.4f' % self.beta)
        mpi.report('h_int =', h_int)
        if one_shot:
            mpi.report('HARTREE SOLVER: starting solver in one-shot mode')
        else:
            mpi.report('HARTREE SOLVER: starting solver in self-consistent mode')
        mpi.report('Including Fock terms:', with_fock)

        def compute_sigma_hartree(Sigma_HF_flat, return_everything=True):
            """ Inner function dedicated to computing

            Parameters
            ----------

            Sigma_HF_flat : list
                Flattened out sigma as computed with the flatten() function in the utils,
                this choice is to interface with the root finding routines in scipy

            return_everything : optional, bool, default=True
                If True, the function returns a tuple of unflattened Sigma_HF, Sigma_int, Sigma_DC 
                if False, the function returns the flattened difference between the input and computed sigma

            """

            Sigma_HF = self._make_zero_sigma()
            Sigma_DC = self._make_zero_sigma()
            Sigma_int = self._make_zero_sigma()

            G_iw = self.G0_iw.copy()
            G_dens = {}
            Sigma_unflattened = unflatten(Sigma_HF_flat, self.gf_struct, self.force_real)
            for bl, G0_bl in self.G0_iw:
                G_iw[bl] << inverse(inverse(G0_bl) - Sigma_unflattened[bl])
                G_dens[bl] = G_iw[bl].density()
                if self.force_real:
                    max_imag = G_dens[bl].imag.max()
                    if max_imag > 1e-10:
                        mpi.report('Warning! Discarding imaginary part of density matrix. Largest imaginary part: %f' % max_imag)
                    G_dens[bl] = G_dens[bl].real
            
            for bl, _ in self.G0_iw:
                #add DC
                n_tot = G_iw.total_density().real
                n_spin = G_iw[bl].total_density().real

                # no dc gets handled as a zero dc_factor
                if not self.dc:
                    self.dc_fixed_value = 0.0

                if self.dc_fixed_occ is not None:
                    mpi.report(f"\nHARTREE SOLVER: modifying occupations in DC calculation with given dc_fixed_occ = {self.dc_fixed_occ:.4f}")
                    n_tot = self.dc_fixed_occ

                if self.dc_fixed_value is None:
                    mpi.report(f"\nHARTREE SOLVER: computing DC for block {bl} with following parameters")
                    mpi.report(f"dc_U = {self.dc_U:.4f} ")
                    mpi.report(f"dc_J = {self.dc_J:.4f}\n")
                    
                    mpi.report(f"HARTREE SOLVER: Calling DC calculation:\n")
                    if self.dc_type not in ['sFLL', 'sAMF']:
                        DC_val, self.E_dc = compute_DC_from_density(n_tot, U = self.dc_U, J = self.dc_J, n_orbitals=self.n_orb, method=self.dc_type)
                    else:
                        DC_val, self.E_dc = compute_DC_from_density(n_tot, U = self.dc_U, J = self.dc_J,
                                                            n_orbitals=self.n_orb, method=self.dc_type, N_spin=n_spin)
                    Sigma_DC[bl] = DC_val * np.eye(G_dens[bl].shape[0])

                    mpi.report(f"\nHARTREE SOLVER: multiplying DC by a factor {self.dc_factor:.4f}")
                    Sigma_DC[bl] *= self.dc_factor
                else:
                    mpi.report(f"\nHARTREE SOLVER: fixing dc to {self.dc_fixed_value:.4f}")
                    Sigma_DC[bl] = self.dc_fixed_value * np.eye(G_dens[bl].shape[0])

            # this assumes that hint is created with off_diag true, i.e. one up and one down block
            # maybe change to the way it is done in TPRF
            for term, coef in h_int:
                bl1, u1 = term[0][1]
                bl2, u2 = term[3][1]
                bl3, u3 = term[1][1]
                bl4, u4 = term[2][1]

                assert(bl1 == bl2 and bl3 == bl4)
                Sigma_int[bl1][u2, u1] += coef * G_dens[bl3][u4, u3] 
                Sigma_int[bl3][u4, u3] += coef * G_dens[bl1][u2, u1]

                if bl1 == bl3 and with_fock:
                    Sigma_int[bl1][u4, u1] -= coef * G_dens[bl3][u2, u3]
                    Sigma_int[bl3][u2, u3] -= coef * G_dens[bl1][u4, u1]

            for function in self.symmetries:
                Sigma_int = function(Sigma_int)
           
            #subtract double counting component from interaction to get total sigma
            for bl, _ in self.G0_iw:
                Sigma_HF[bl] = Sigma_int[bl]-Sigma_DC[bl]
                
                # As a last resort check, whatever the solver does when J=0  should be 
                # reduceable to the Dudarev formula, this is left here as a sanity 
                # check one may perform should the implementation change

                #Sigma_HF[bl] = self.U *(0.5 * np.eye(G_dens[bl].shape[0]) - G_dens[bl])

            if return_everything:
                return Sigma_HF, Sigma_int, Sigma_DC
            else:
                return Sigma_HF_flat - flatten(Sigma_HF, self.force_real)
        
        def report_results(Sigma_HF, G_dens):
            with np.printoptions(suppress=True, precision=4):
              for name, bl in Sigma_HF.items():
                  mpi.report('HARTREE SOLVER: Sigma_HF[\'%s\']:' % name)
                  mpi.report(bl)
              for name, bl in G_dens.items():
                  mpi.report('Final G_dens[\'%s\']:' % name)
                  mpi.report(bl)


        Sigma_HF_init = self.Sigma_HF

        with np.printoptions(suppress=True, precision=4):
            for name, bl in self.Sigma_HF.items():
                mpi.report('HARTREE SOLVER: Sigma_HF before iterating[\'%s\']:' % name)
                mpi.report(bl)

        if one_shot:
            if mpi.is_master_node():
                self.Sigma_HF, self.Sigma_int, self.Sigma_DC = compute_sigma_hartree(flatten(Sigma_HF_init, self.force_real), return_everything=True)
        else:  # self consistent Hartree-Fock
            if mpi.is_master_node():
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                    root_finder = root(lambda x: compute_sigma_hartree(x, return_everything=False),
                                                flatten(Sigma_HF_init, self.force_real),
                                                method=method,
                                                tol=tol
                                        )

                if root_finder['success']:
                    mpi.report('Self Consistent Hartree-Fock converged successfully, performing final iteration')
                    self.hf_converged = True
                else:
                    self.hf_converged = False
                    mpi.report('Hartree-Fock solver did not converge successfully. Feeding last iteration as guess')
                    mpi.report(root_finder['message'])

                self.Sigma_HF, self.Sigma_int, self.Sigma_DC = compute_sigma_hartree(root_finder['x'], return_everything=True)

        mpi.barrier(100)
        self.Sigma_HF = mpi.bcast(self.Sigma_HF)
        self.Sigma_int = mpi.bcast(self.Sigma_int)
        self.Sigma_DC = mpi.bcast(self.Sigma_DC)

        for bl, G0_bl in self.G0_iw:
            self.G_iw[bl] << inverse(inverse(G0_bl) - self.Sigma_HF[bl])
        G_dens = {bl: self.G_iw[bl].density() for bl, _ in self.gf_struct}
        if self.force_real:
            G_dens = {bl: d.real for bl, d in G_dens.items()}
        self.density = G_dens

        report_results(self.Sigma_HF, G_dens)


    def interaction_energy(self):
        """ Calculate the interaction energy

        """
        E = 0
        for bl, gbl in self.G_iw:
            E += 0.5 * np.trace(self.Sigma_int[bl].dot(gbl.density().real))
        return E
    
    def DC_energy(self):
        """ Exposes the DC energy
        """
        return self.E_dc
    
    def reinitialize_sigma(self, Sigma_guess):
        """ Changes in place the sigma with the average over frequencies of a given Gf object.
            Used to update the initial guess, must be called before the solve() method

        Parameters
        ----------
            Sigma_guess : GfImFreq or GfReFreq object
        """
        for bl in self.Sigma_HF:
            if self.force_real:
                self.Sigma_HF[bl] = np.mean(Sigma_guess[bl].data, axis=0).real
            else:
                self.Sigma_HF[bl] = np.mean(Sigma_guess[bl].data, axis=0)

        with np.printoptions(suppress=True, precision=4):
          for name, bl in self.Sigma_HF.items():
              mpi.report('HARTREE SOLVER: Updated guess for Sigma_HF[\'%s\']:' % name)
              mpi.report(bl)

    def set_G0_iw(self, Gloc):
        """ Set G0_iw from local Green's function by evaluating Weiss field at DLR frequency points.

        Parameters
        ----------
            Gloc : BlockGf on MeshImFreq or compatible mesh
                Local Green's function used to compute the Weiss field G0 = (G_loc^{-1} + Sigma)^{-1}
        """
        for bl_name, _ in self.gf_struct:
            for iw in self.G0_iw[bl_name].mesh:
                Gloc_iw = Gloc[bl_name](iw.value)
                Sigma_bl = self.Sigma_HF[bl_name]
                G0_iw = np.linalg.inv(np.linalg.inv(Gloc_iw) + Sigma_bl)
                self.G0_iw[bl_name][iw] = G0_iw

    def __reduce_to_dict__(self):
        store_dict = {'w_max': self.w_max, 'eps': self.eps, 'G0_iw': self.G0_iw, 'G_iw': self.G_iw,
                      'gf_struct': self.gf_struct, 'beta': self.beta,
                      'symmetries': self.symmetries, 'Sigma_HF': self.Sigma_HF,
                      'git_hash': self.git_hash}
        if hasattr(self, 'last_solve_params'):
            store_dict['last_solve_params'] = self.last_solve_params
        return store_dict

    @classmethod
    def __factory_from_dict__(cls, _name, D):
        instance = cls(D['gf_struct'], D['beta'], D['w_max'], D['eps'], symmetries=D.get('symmetries', []))
        instance.Sigma_HF = D['Sigma_HF']
        instance.G0_iw = D['G0_iw']
        instance.G_iw = D['G_iw']
        instance.git_hash = D['git_hash']
        if 'last_solve_params' in D:
            instance.last_solve_params = D['last_solve_params']
        return instance


register_class(ImpuritySolver)
