use std::{f64::consts::PI, sync::Arc};

use ndarray::Array2;

use crate::point::Point;

use ndarray::Array4;

use crate::integrals;

#[derive(Clone)]
pub(crate) struct PrimitiveGaussian {
    contraction_coefficient: f64, // already includes normalization
    alpha: f64,
    center: Point,
    angular: (u8, u8, u8), // (lx, ly, lz)
}

impl PrimitiveGaussian {
    pub(crate) fn new(
        contraction_coefficient: f64,
        alpha: f64,
        center: Point,
        angular: (u8, u8, u8),
    ) -> Self {
        PrimitiveGaussian {
            contraction_coefficient: contraction_coefficient * get_normalization_coefficient(alpha),
            alpha,
            center,
            angular,
        }
    }

    pub(crate) fn contraction_coefficient(&self) -> f64 {
        self.contraction_coefficient
    }

    pub(crate) fn alpha(&self) -> f64 {
        self.alpha
    }

    pub(crate) fn center(&self) -> Point {
        self.center
    }

    pub(crate) fn compute(&self, r: &Point) -> f64 {
        let dx = r.x - self.center.x;
        let dy = r.y - self.center.y;
        let dz = r.z - self.center.z;

        let angular_part = dx.powi(self.angular.0.into())
            * dy.powi(self.angular.1.into())
            * dz.powi(self.angular.2.into());

        self.contraction_coefficient
            * angular_part
            * (-(self.alpha * (dx * dx + dy * dy + dz * dz))).exp()
    }
}

fn get_normalization_coefficient(alpha: f64) -> f64 {
    ((2.0 * alpha) / PI).powf(0.75)
}

pub(crate) struct BasisSet {
    pub(crate) shells: Vec<Arc<Shell>>,
    pub(crate) functions: Vec<BasisFunction>,
}

impl BasisSet {
    pub(crate) fn new(shells: Vec<Shell>) -> Self {
        let shells: Vec<Arc<Shell>> = shells.into_iter().map(Arc::new).collect();

        let mut functions = Vec::new();

        for shell in &shells {
            match shell.angular {
                AngularMomentum::S => {
                    functions.push(BasisFunction {
                        shell: Arc::clone(shell),
                        component: 0,
                    });
                }
                AngularMomentum::P => {
                    for i in 0..3 {
                        functions.push(BasisFunction {
                            shell: Arc::clone(shell),
                            component: i,
                        });
                    }
                }
            }
        }

        Self { shells, functions }
    }

    pub(crate) fn num_contracted_gaussians(&self) -> usize {
        self.shells.len()
    }

    pub(crate) fn num_occupied_orbitals(&self, n_electrons: usize) -> usize {
        n_electrons / 2
    }

    pub(crate) fn overlap_matrix(&self) -> Array2<f64> {
        self.one_electron_matrix(integrals::overlap)
    }

    pub(crate) fn kinetic_energy_matrix(&self) -> Array2<f64> {
        self.one_electron_matrix(integrals::kinetic_energy)
    }

    pub(crate) fn nuclear_attraction_matrix(&self) -> Array2<f64> {
        self.one_electron_matrix(integrals::nuclear_attraction)
    }

    fn nbf(&self) -> usize {
        self.functions.len()
    }

    fn one_electron_matrix(
        &self,
        f: impl Fn(&BasisFunction, &BasisFunction) -> f64,
    ) -> Array2<f64> {
        let n = self.nbf();
        let mut m = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                let val = f(&self.functions[i], &self.functions[j]);
                m[[i, j]] = val;
                m[[j, i]] = val;
            }
        }
        m
    }

    pub(crate) fn electron_repulsion_tensor(&self) -> Array4<f64> {
        let n = self.functions.len();
        let mut eri = Array4::zeros((n, n, n, n));

        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    for d in 0..n {
                        eri[[a, b, c, d]] = integrals::electron_repulsion(
                            &self.functions[a],
                            &self.functions[b],
                            &self.functions[c],
                            &self.functions[d],
                        );
                    }
                }
            }
        }

        eri
    }

    pub(crate) fn evaluate_molecular_orbital(
        &self,
        r: &Point,
        coefficients: &Array2<f64>,
        mo_index: usize,
    ) -> f64 {
        self.functions
            .iter()
            .enumerate()
            .map(|(mu, bf)| coefficients[[mu, mo_index]] * bf.compute(r))
            .sum()
    }
}

#[derive(Clone)]
pub(crate) enum AngularMomentum {
    S,
    P,
}

#[derive(Clone)]
pub(crate) struct Shell {
    pub(crate) center: Point,
    pub(crate) angular: AngularMomentum,
    pub(crate) primitives: Vec<PrimitiveGaussian>,
}

pub(crate) struct BasisFunction {
    pub(crate) shell: Arc<Shell>,
    pub(crate) component: u8, // 0=s, 0..2 for p
}

impl BasisFunction {
    pub(crate) fn compute(&self, r: &Point) -> f64 {
        let shell = &self.shell;

        let dx = r.x - shell.center.x;
        let dy = r.y - shell.center.y;
        let dz = r.z - shell.center.z;

        let r2 = dx * dx + dy * dy + dz * dz;

        let gaussian: f64 = shell
            .primitives
            .iter()
            .map(|p| p.contraction_coefficient * (-p.alpha * r2).exp())
            .sum();

        let angular = match (shell.angular.clone(), self.component) {
            (AngularMomentum::S, _) => 1.0,
            (AngularMomentum::P, 0) => dx,
            (AngularMomentum::P, 1) => dy,
            (AngularMomentum::P, 2) => dz,
            (AngularMomentum::P, _) => unreachable!(),
        };

        gaussian * angular
    }
}
