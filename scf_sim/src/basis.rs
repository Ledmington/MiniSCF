use crate::integrals;
use crate::integrals::nuclear_attraction;
use ndarray::Array2;
use ndarray::Array4;
use scf_core::Atom;
use scf_core::point::Point;
use std::{f64::consts::PI, sync::Arc};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PrimitiveGaussian {
    contraction_coefficient: f64, // raw, does not include normalization
    alpha: f64,
    center: Point,
}

impl PrimitiveGaussian {
    pub(crate) fn new(contraction_coefficient: f64, alpha: f64, center: Point) -> Self {
        assert!(alpha > 0.0);
        PrimitiveGaussian {
            contraction_coefficient,
            alpha,
            center,
        }
    }

    pub(crate) fn alpha(&self) -> f64 {
        self.alpha
    }

    pub(crate) fn center(&self) -> Point {
        self.center
    }

    pub fn normalized_coefficient(&self, (lx, ly, lz): (u8, u8, u8)) -> f64 {
        let numerator = (4.0 * self.alpha).powi((lx + ly + lz).into());

        let denominator = double_factorial(2 * lx as i32 - 1)
            * double_factorial(2 * ly as i32 - 1)
            * double_factorial(2 * lz as i32 - 1);

        ((2.0 * self.alpha) / PI).powf(0.75) * (numerator / (denominator as f64)).sqrt()
    }
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

#[derive(Debug, PartialEq)]
pub(crate) struct BasisSet {
    pub(crate) functions: Vec<BasisFunction>,
}

impl BasisSet {
    pub(crate) fn new(shells: Vec<(Shell, (u8, u8, u8))>) -> Self {
        let mut functions = Vec::new();

        for (shell, angular_momentum) in &shells {
            functions.push(BasisFunction {
                shell: Arc::new(shell.clone()),
                angular_momentum: *angular_momentum,
            });
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

    pub(crate) fn nuclear_attraction_matrix(&self, nuclei: &[Atom]) -> Array2<f64> {
        let n = self.num_contracted_gaussians();
        let mut m = Array2::zeros((n, n));

        for i in 0..n {
            m[[i, i]] = nuclear_attraction(&self.functions[i], &self.functions[i], nuclei);
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let val = nuclear_attraction(&self.functions[i], &self.functions[j], nuclei);
                m[[i, j]] = val;
                m[[j, i]] = val;
            }
        }
        m
    }

    fn one_electron_matrix(
        &self,
        f: impl Fn(&BasisFunction, &BasisFunction) -> f64,
    ) -> Array2<f64> {
        let n = self.num_contracted_gaussians();
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
        let n = self.num_contracted_gaussians();
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

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Shell {
    pub(crate) center: Point,
    pub(crate) primitives: Vec<PrimitiveGaussian>,
}

#[derive(Debug, PartialEq)]
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
            .map(|p| self.normalized_coefficient(p) * (-p.alpha * r2).exp())
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

    pub fn normalized_coefficient(&self, p: &PrimitiveGaussian) -> f64 {
        p.contraction_coefficient * p.normalized_coefficient(self.angular_momentum)
    }
}

#[cfg(test)]
mod tests {
    use rand::{RngExt, SeedableRng, rngs::ChaCha8Rng};

    use crate::integrals::primitive_nuclear_attraction;

    use super::*;

    #[test]
    fn test_self_overlap_s() {
        let seed = rand::rng().random();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let center = Point::new(
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
        );
        let shell = Shell {
            center,
            primitives: vec![
                PrimitiveGaussian::new(0.1543289673, 3.425250914, center),
                PrimitiveGaussian::new(0.5353281423, 0.6239137298, center),
                PrimitiveGaussian::new(0.4446345422, 0.168855404, center),
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
            "Expected overlap between {bf:?} and itself to be {expected_overlap} but was {actual_overlap} (seed: {seed})."
        );
    }

    #[test]
    fn test_self_overlap_p() {
        let seed = rand::rng().random();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let center = Point::new(
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
        );
        let shell = Shell {
            center,
            primitives: vec![
                PrimitiveGaussian::new(0.1559162750, 2.941249355, center),
                PrimitiveGaussian::new(0.6076837186, 0.6834830964, center),
                PrimitiveGaussian::new(0.3919573931, 0.2222899159, center),
            ],
        };
        let px = BasisFunction {
            shell: Arc::new(shell.clone()),
            angular_momentum: (1, 0, 0),
        };
        let py = BasisFunction {
            shell: Arc::new(shell.clone()),
            angular_momentum: (0, 1, 0),
        };
        let pz = BasisFunction {
            shell: Arc::new(shell),
            angular_momentum: (0, 0, 1),
        };

        {
            let actual_overlap = integrals::overlap(&px, &px);
            let expected_overlap = 1.0;
            assert!(
                (actual_overlap - expected_overlap).abs() < 1e-10,
                "Expected overlap between Px ({px:?}) and itself to be {expected_overlap} but was {actual_overlap} (seed: {seed})."
            );
        }

        {
            let actual_overlap = integrals::overlap(&py, &py);
            let expected_overlap = 1.0;
            assert!(
                (actual_overlap - expected_overlap).abs() < 1e-10,
                "Expected overlap between Py ({py:?}) and itself to be {expected_overlap} but was {actual_overlap} (seed: {seed})."
            );
        }

        {
            let actual_overlap = integrals::overlap(&pz, &pz);
            let expected_overlap = 1.0;
            assert!(
                (actual_overlap - expected_overlap).abs() < 1e-10,
                "Expected overlap between Pz ({pz:?}) and itself to be {expected_overlap} but was {actual_overlap} (seed: {seed})."
            );
        }
    }

    #[test]
    fn test_orthogonal_p_orbitals() {
        let seed = rand::rng().random();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let center = Point::new(
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
        );
        let shell = Shell {
            center,
            primitives: vec![
                PrimitiveGaussian::new(0.1559162750, 2.941249355, center),
                PrimitiveGaussian::new(0.6076837186, 0.6834830964, center),
                PrimitiveGaussian::new(0.3919573931, 0.2222899159, center),
            ],
        };
        let px = BasisFunction {
            shell: Arc::new(shell.clone()),
            angular_momentum: (1, 0, 0),
        };
        let py = BasisFunction {
            shell: Arc::new(shell.clone()),
            angular_momentum: (0, 1, 0),
        };
        let pz = BasisFunction {
            shell: Arc::new(shell),
            angular_momentum: (0, 0, 1),
        };

        {
            let actual_overlap = integrals::overlap(&px, &py);
            let expected_overlap = 0.0;
            assert!(
                (actual_overlap - expected_overlap).abs() < 1e-10,
                "Expected overlap between Px ({px:?}) and Py ({py:?}) to be {expected_overlap} but was {actual_overlap} (seed: {seed})."
            );
        }

        {
            let actual_overlap = integrals::overlap(&px, &pz);
            let expected_overlap = 0.0;
            assert!(
                (actual_overlap - expected_overlap).abs() < 1e-10,
                "Expected overlap between Px ({px:?}) and Pz ({pz:?}) to be {expected_overlap} but was {actual_overlap} (seed: {seed})."
            );
        }

        {
            let actual_overlap = integrals::overlap(&py, &pz);
            let expected_overlap = 0.0;
            assert!(
                (actual_overlap - expected_overlap).abs() < 1e-10,
                "Expected overlap between Py ({py:?}) and Pz ({pz:?}) to be {expected_overlap} but was {actual_overlap} (seed: {seed})."
            );
        }
    }

    #[test]
    fn test_single_normalized_s_primitive_kinetic() {
        let seed = rand::rng().random();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let center = Point::new(
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
        );

        let alpha = 0.5;

        let primitive = PrimitiveGaussian {
            alpha,
            contraction_coefficient: 1.0,
            center,
        };

        let shell = Shell {
            center: primitive.center,
            primitives: vec![primitive],
        };

        let bf = BasisFunction {
            shell: Arc::new(shell.clone()),
            angular_momentum: (0, 0, 0),
        };

        let actual = integrals::kinetic_energy(&bf, &bf);

        let expected = 1.5 * alpha;

        assert!(
            (actual - expected).abs() < 1e-10,
            "Expected kinetic energy between {bf:?} and itself to be {expected} but was {actual} (seed: {seed})."
        );
    }

    #[test]
    fn test_single_normalized_s_primitive_with_contraction() {
        let seed = rand::rng().random();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let center = Point::new(
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
        );

        let alpha = 0.5;
        let c = 0.7;

        let primitive = PrimitiveGaussian {
            alpha,
            contraction_coefficient: c,
            center,
        };

        let shell = Shell {
            center: primitive.center,
            primitives: vec![primitive],
        };

        let bf = BasisFunction {
            shell: Arc::new(shell.clone()),
            angular_momentum: (0, 0, 0),
        };

        let actual = integrals::kinetic_energy(&bf, &bf);
        let expected = c * c * 1.5 * alpha;
        assert!(
            (actual - expected).abs() < 1e-10,
            "Expected kinetic energy between {bf:?} and itself to be {expected} but was {actual} (seed: {seed})."
        );
    }

    #[test]
    fn unnormalized_nuclear_attraction_ss_same_center() {
        let alpha = 1.0;
        let nucleus = Point::new(0.0, 0.0, 0.0);
        let primitive = PrimitiveGaussian::new(1.0, alpha, nucleus);

        let actual =
            primitive_nuclear_attraction(&primitive, &primitive, &nucleus, &(0, 0, 0), &(0, 0, 0));
        let expected = -PI;
        assert!(
            (actual - expected).abs() < 1e-10,
            "Expected nuclear attraction between primitive {primitive:?} and itself with a nucleus at its center to be {expected} but was {actual}."
        );
    }
}
