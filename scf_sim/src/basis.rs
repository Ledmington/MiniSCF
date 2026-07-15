use crate::integrals;
use ndarray::Array2;
use ndarray::Array4;
use scf_core::point::Point;
use std::{f64::consts::PI, sync::Arc};

#[derive(Clone, Debug)]
pub(crate) struct PrimitiveGaussian {
    contraction_coefficient: f64, // already includes normalization
    alpha: f64,
    center: Point,
}

impl PrimitiveGaussian {
    pub(crate) fn new(
        contraction_coefficient: f64,
        alpha: f64,
        center: Point,
        angular_momentum: (u8, u8, u8),
    ) -> Self {
        PrimitiveGaussian {
            contraction_coefficient: contraction_coefficient
                * get_normalization_coefficient(alpha, angular_momentum),
            alpha,
            center,
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
}

fn get_normalization_coefficient(alpha: f64, (lx, ly, lz): (u8, u8, u8)) -> f64 {
    let numerator = (4.0 * alpha).powi((lx + ly + lz).into());

    let denominator = double_factorial(2 * lx as i32 - 1)
        * double_factorial(2 * ly as i32 - 1)
        * double_factorial(2 * lz as i32 - 1);

    ((2.0 * alpha) / PI).powf(0.75) * (numerator / (denominator as f64)).sqrt()
}

fn double_factorial(mut n: i32) -> i32 {
    assert!(n >= -1);
    if n <= 1 {
        return 1;
    }
    let mut s = 1;
    while n > 1 {
        s *= n;
        n -= 2;
    }
    s
}

#[derive(Debug)]
pub(crate) struct BasisSet {
    pub(crate) functions: Vec<BasisFunction>,
}

impl BasisSet {
    pub(crate) fn new(shells: Vec<(Shell, AngularMomentum)>) -> Self {
        let mut functions = Vec::new();

        for (shell, angular) in &shells {
            match angular {
                AngularMomentum::S => {
                    functions.push(BasisFunction {
                        shell: Arc::clone(&Arc::new(shell.clone())),
                        angular_momentum: (0, 0, 0),
                    });
                }
                AngularMomentum::P => {
                    // px
                    functions.push(BasisFunction {
                        shell: Arc::clone(&Arc::new(shell.clone())),
                        angular_momentum: (1, 0, 0),
                    });
                    // py
                    functions.push(BasisFunction {
                        shell: Arc::clone(&Arc::new(shell.clone())),
                        angular_momentum: (0, 1, 0),
                    });
                    // pz
                    functions.push(BasisFunction {
                        shell: Arc::clone(&Arc::new(shell.clone())),
                        angular_momentum: (0, 0, 1),
                    });
                }
            }
        }

        Self { functions }
    }

    pub(crate) fn num_contracted_gaussians(&self) -> usize {
        self.functions.len()
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
            m[[i, i]] = f(&self.functions[i], &self.functions[i]);
        }

        for i in 0..n {
            for j in (i + 1)..n {
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

#[derive(Clone, PartialEq, Debug)]
pub(crate) enum AngularMomentum {
    S,
    P,
}

#[derive(Clone, Debug)]
pub(crate) struct Shell {
    pub(crate) center: Point,
    pub(crate) primitives: Vec<PrimitiveGaussian>,
}

#[derive(Debug)]
pub(crate) struct BasisFunction {
    pub(crate) shell: Arc<Shell>,
    pub(crate) angular_momentum: (u8, u8, u8), // (lx, ly, lz)
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

        let angular = match self.angular_momentum {
            (0, 0, 0) => 1.0,
            (1, 0, 0) => dx,
            (0, 1, 0) => dy,
            (0, 0, 1) => dz,
            _ => panic!(
                "Don't know what to do with angular momentum {:?}.",
                self.angular_momentum
            ),
        };

        gaussian * angular
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_overlap_s() {
        let center = Point::new(0.0, 0.0, 0.0);
        let shell = Shell {
            center,
            primitives: vec![
                PrimitiveGaussian::new(0.1543289673, 3.425250914, center, (0, 0, 0)),
                PrimitiveGaussian::new(0.5353281423, 0.6239137298, center, (0, 0, 0)),
                PrimitiveGaussian::new(0.4446345422, 0.168855404, center, (0, 0, 0)),
            ],
        };
        let bf = BasisFunction {
            shell: Arc::new(shell),
            angular_momentum: (0, 0, 0),
        };
        let actual_overlap = integrals::overlap(&bf, &bf);
        let expected_overlap = 1.0;
        assert!(
            (actual_overlap - expected_overlap).abs() < 1e-10,
            "Expected overlap between {:?} and itself to be {} but was {}.",
            bf,
            expected_overlap,
            actual_overlap
        );
    }

    #[test]
    fn self_overlap_p() {
        let center = Point::new(0.0, 0.0, 0.0);
        let shell = Shell {
            center,
            primitives: vec![
                PrimitiveGaussian::new(0.1559162750, 2.941249355, center, (1, 0, 0)),
                PrimitiveGaussian::new(0.6076837186, 0.6834830964, center, (0, 1, 0)),
                PrimitiveGaussian::new(0.3919573931, 0.2222899159, center, (0, 0, 1)),
            ],
        };
        let bf = BasisFunction {
            shell: Arc::new(shell),
            angular_momentum: (1, 0, 0), // FIXME; add also py and pz
        };
        let actual_overlap = integrals::overlap(&bf, &bf);
        let expected_overlap = 1.0;
        assert!(
            (actual_overlap - expected_overlap).abs() < 1e-10,
            "Expected overlap between {:?} and itself to be {} but was {}.",
            bf,
            expected_overlap,
            actual_overlap
        );
    }
}
