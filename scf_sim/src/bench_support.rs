//! Entry points for the criterion benchmarks in `benches/`.
//!
//! This module is `pub` only so that the benchmarks can reach `pub(crate)` internals: it
//! is a benchmarking seam, not public API.

pub use crate::basis::AngularMomentum;
use crate::basis::{BasisFunction, BasisSet, PrimitiveGaussian, Shell};
use crate::basis_reader::{build_basis, parse_nwchem_basis_text};
use crate::cube_writer::{Grid, compute_grid, evaluate_electrostatic_potential, evaluate_orbitals};
use crate::integrals;
use crate::sim::{
    OptimizationParameters, compute_two_electron_contribution, density_matrix, run_rhf_simulation,
};
use ndarray::{Array2, Array4};
use scf_core::Atom;
use scf_core::point::Point;
use std::sync::Arc;

/// A molecule with its basis set and converged RHF solution, so that each benchmark can
/// time one step of the simulation in isolation.
pub struct Molecule {
    atoms: Vec<Atom>,
    basis: BasisSet,
    c: Array2<f64>,
    density: Array2<f64>,
    eri: Array4<f64>,
}

impl Molecule {
    /// Builds the basis set from `basis_text` (NWChem format) for `atoms` (positions in
    /// bohr), then runs a full SCF to have realistic MO coefficients and density.
    pub fn new(atoms: Vec<Atom>, basis_text: &str) -> Self {
        let library = parse_nwchem_basis_text(basis_text).expect("valid basis set");
        let basis = build_basis(&atoms, &library).expect("basis functions for every atom");
        let c = run_rhf_simulation(&atoms, &basis, &Self::parameters());
        let n_electrons: usize = atoms.iter().map(|a| a.charge as usize).sum();
        let density = density_matrix(&c, basis.num_occupied_orbitals(n_electrons));
        let eri = basis.electron_repulsion_tensor();
        Self {
            atoms,
            basis,
            c,
            density,
            eri,
        }
    }

    fn parameters() -> OptimizationParameters {
        OptimizationParameters::new(100, 1e-10, 1e-8, 1e-15)
    }

    pub fn num_functions(&self) -> usize {
        self.basis.num_contracted_gaussians()
    }

    /// Overlap, kinetic-energy and nuclear-attraction matrices.
    pub fn one_electron_matrices(&self) -> [Array2<f64>; 3] {
        [
            self.basis.overlap_matrix(),
            self.basis.kinetic_energy_matrix(),
            self.basis.nuclear_attraction_matrix(&self.atoms),
        ]
    }

    pub fn electron_repulsion_tensor(&self) -> Array4<f64> {
        self.basis.electron_repulsion_tensor()
    }

    /// The two-electron part of the Fock matrix for the converged density, the work done
    /// once per SCF iteration.
    pub fn two_electron_fock(&self) -> Array2<f64> {
        compute_two_electron_contribution(&self.density, &self.eri)
    }

    /// A whole RHF simulation, from the integrals to the converged MO coefficients.
    pub fn rhf(&self) -> Array2<f64> {
        run_rhf_simulation(&self.atoms, &self.basis, &Self::parameters())
    }

    /// The `num_slabs` central x-slabs of the grid `rhf_sim` writes its cube files on: a
    /// realistic workload that is much smaller than the whole grid.
    fn central_slabs(&self, num_slabs: usize) -> Grid {
        let full = compute_grid(&self.atoms);
        let nx = num_slabs.min(full.nx);
        let start = (full.nx - nx) / 2;
        Grid {
            origin: Point::new(
                full.origin.x + (start as f64) * full.dx,
                full.origin.y,
                full.origin.z,
            ),
            nx,
            ..full
        }
    }

    /// Number of grid points in `num_slabs` central x-slabs.
    pub fn num_grid_points(&self, num_slabs: usize) -> usize {
        let grid = self.central_slabs(num_slabs);
        grid.nx * grid.ny * grid.nz
    }

    /// Every molecular orbital on `num_slabs` central x-slabs of the cube grid.
    pub fn orbitals_on_slabs(&self, num_slabs: usize) -> Vec<Vec<f64>> {
        evaluate_orbitals(
            &self.central_slabs(num_slabs),
            &self.basis,
            &self.c,
            self.c.ncols(),
        )
    }

    /// The molecular electrostatic potential on `num_slabs` central x-slabs of the cube
    /// grid.
    pub fn electrostatic_potential_on_slabs(&self, num_slabs: usize) -> Vec<f64> {
        evaluate_electrostatic_potential(
            &self.central_slabs(num_slabs),
            &self.basis,
            &self.atoms,
            &self.density,
        )
    }
}

/// Four uncontracted shells of the same angular momentum on four different centers: the
/// general case of the ERI code, isolating the cost of the angular momentum (four
/// contracted f shells would take seconds, with 81 primitive quartets per shell quartet).
pub struct ShellQuartet(Vec<BasisFunction>);

impl ShellQuartet {
    pub fn new(angular_momentum: AngularMomentum) -> Self {
        let centers = [
            Point::new(0.0, 0.0, 0.0),
            Point::new(1.4, 0.0, 0.0),
            Point::new(0.0, 1.8, 0.3),
            Point::new(-0.9, 0.5, 1.6),
        ];
        let functions = centers
            .into_iter()
            .flat_map(|center| {
                let shell = Arc::new(Shell {
                    center,
                    primitives: vec![PrimitiveGaussian::new(1.0, 0.8, center)],
                });
                angular_momentum
                    .cartesian_components()
                    .into_iter()
                    .map(move |component| BasisFunction {
                        shell: Arc::clone(&shell),
                        angular_momentum: component,
                    })
            })
            .collect();
        Self(functions)
    }

    /// The ERI tensor of the four shells, built like the one of a molecule.
    pub fn electron_repulsion_tensor(&self) -> Array4<f64> {
        integrals::electron_repulsion_tensor(&self.0)
    }
}

/// Fills `out[..=n_max]` with the Boys functions `F_0(t)` ... `F_n_max(t)`.
pub fn boys_all(n_max: usize, t: f64, out: &mut [f64]) {
    integrals::boys_all(n_max, t, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_quartet_tensor_matches_reference_integrals() {
        // d takes 17 s to check in debug builds, f even longer.
        for angular_momentum in [AngularMomentum::S, AngularMomentum::P] {
            let quartet = ShellQuartet::new(angular_momentum.clone());
            let eri = quartet.electron_repulsion_tensor();
            let f = &quartet.0;
            for ((a, b, c, d), &value) in eri.indexed_iter() {
                let expected = integrals::electron_repulsion(&f[a], &f[b], &f[c], &f[d]);
                assert!(
                    (value - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                    "{angular_momentum:?} ({a}{b}|{c}{d}): {value} instead of {expected}"
                );
            }
        }
    }
}
