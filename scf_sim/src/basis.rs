use crate::integrals;
use crate::integrals::nuclear_attraction;
use crate::primitive_gaussian::PrimitiveGaussian;
use ndarray::Array2;
use ndarray::Array4;
use scf_core::Atom;
use scf_core::point::Point;
use std::sync::Arc;

#[derive(Debug, PartialEq)]
pub struct BasisSet {
    pub functions: Vec<BasisFunction>,
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
    pub(crate) contraction_coefficients: Vec<f64>,
    pub(crate) primitives: Vec<PrimitiveGaussian>,
}

#[derive(Debug, PartialEq)]
pub struct BasisFunction {
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
            .map(|p| p.normalization_constant() * (-p.alpha() * r2).exp())
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
    use std::f64::consts::PI;

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
            contraction_coefficients: vec![3.425250914, 0.6239137298, 0.168855404],
            primitives: vec![
                PrimitiveGaussian::new(0, 0, 0, 0.1543289673, center),
                PrimitiveGaussian::new(0, 0, 0, 0.5353281423, center),
                PrimitiveGaussian::new(0, 0, 0, 0.4446345422, center),
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
                PrimitiveGaussian::new(1, 0, 0, 0.1559162750, center),
                PrimitiveGaussian::new(0, 1, 0, 0.6076837186, center),
                PrimitiveGaussian::new(0, 0, 1, 0.3919573931, center),
            ],
            contraction_coefficients: vec![2.941249355, 0.6834830964, 0.2222899159],
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
                PrimitiveGaussian::new(1, 0, 0, 0.1559162750, center),
                PrimitiveGaussian::new(0, 1, 0, 0.6076837186, center),
                PrimitiveGaussian::new(0, 0, 1, 0.3919573931, center),
            ],
            contraction_coefficients: vec![2.941249355, 0.6834830964, 0.2222899159],
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

        let primitive = PrimitiveGaussian::new(0, 0, 0, alpha, center);

        let shell = Shell {
            center: *primitive.center(),
            primitives: vec![primitive],
            contraction_coefficients: vec![1.0],
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

        let primitive = PrimitiveGaussian::new(0, 0, 0, alpha, center);

        let shell = Shell {
            center: *primitive.center(),
            primitives: vec![primitive],
            contraction_coefficients: vec![1.0],
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
        let primitive = PrimitiveGaussian::new(0, 0, 0, alpha, nucleus);

        let actual =
            primitive_nuclear_attraction(&primitive, &primitive, &nucleus, &(0, 0, 0), &(0, 0, 0));
        let expected = -PI;
        assert!(
            (actual - expected).abs() < 1e-10,
            "Expected nuclear attraction between primitive {primitive:?} and itself with a nucleus at its center to be {expected} but was {actual}."
        );
    }

    #[test]
    fn test_single_normalized_s_primitive_electron_repulsion() {
        let seed = rand::rng().random();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let center = Point::new(
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
        );

        let alpha: f64 = 0.5;
        let primitive = PrimitiveGaussian::new(0, 0, 0, alpha, center);
        let shell = Shell {
            center,
            primitives: vec![primitive],
            contraction_coefficients: vec![1.0],
        };
        let bf = BasisFunction {
            shell: Arc::new(shell),
            angular_momentum: (0, 0, 0),
        };

        let actual = integrals::electron_repulsion(&bf, &bf, &bf, &bf);
        let expected = 2.0 * alpha.sqrt() / PI.sqrt();
        assert!(
            (actual - expected).abs() < 1e-10,
            "Expected (ss|ss) ERI to be {expected} but was {actual} (seed: {seed})."
        );
    }

    #[test]
    fn test_ssss_electron_repulsion_same_center() {
        let seed = rand::rng().random();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let center = Point::new(
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
        );

        let make_bf = |alpha| BasisFunction {
            shell: Arc::new(Shell {
                center,
                primitives: vec![PrimitiveGaussian::new(0, 0, 0, alpha, center)],
                contraction_coefficients: vec![1.0],
            }),
            angular_momentum: (0, 0, 0),
        };

        let alpha_1: f64 = 0.5;
        let alpha_2: f64 = 1.0;
        let alpha_3: f64 = 1.5;
        let alpha_4: f64 = 2.0;

        let a = make_bf(alpha_1);
        let b = make_bf(alpha_2);
        let c = make_bf(alpha_3);
        let d = make_bf(alpha_4);

        let actual = integrals::electron_repulsion(&a, &b, &c, &d);
        let p = alpha_1 + alpha_2;
        let q = alpha_3 + alpha_4;
        let expected = (16.0 / PI.sqrt()) * (alpha_1 * alpha_2 * alpha_3 * alpha_4).powf(0.75)
            / (p * q * (p + q).sqrt());
        assert!(
            (actual - expected).abs() < 1e-10,
            "Expected (ss|ss) ERI to be {expected} but was {actual} (seed: {seed})."
        );
    }

    #[test]
    fn test_ssss_electron_repulsion_symmetry() {
        let make_bf = |rng: &mut ChaCha8Rng| BasisFunction {
            shell: Arc::new(Shell {
                center: Point::new(
                    rng.random_range(-10.0..10.0),
                    rng.random_range(-10.0..10.0),
                    rng.random_range(-10.0..10.0),
                ),
                primitives: vec![PrimitiveGaussian::new(
                    0,
                    0,
                    0,
                    rng.random_range(0.1..5.0),
                    Point::new(
                        rng.random_range(-10.0..10.0),
                        rng.random_range(-10.0..10.0),
                        rng.random_range(-10.0..10.0),
                    ),
                )],
                contraction_coefficients: vec![1.0],
            }),
            angular_momentum: (0, 0, 0),
        };

        let seed = rand::rng().random();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let a = make_bf(&mut rng);
        let b = make_bf(&mut rng);
        let c = make_bf(&mut rng);
        let d = make_bf(&mut rng);

        const TOLERANCE: f64 = 1e-10;

        let abcd = integrals::electron_repulsion(&a, &b, &c, &d);

        let abdc = integrals::electron_repulsion(&a, &b, &d, &c);
        assert!(
            (abcd - abdc).abs() < TOLERANCE,
            "Expected (ab|cd) ERI ({abcd}) to be equal to (ab|dc) ERI ({abdc}) (seed: {seed})."
        );

        let bacd = integrals::electron_repulsion(&b, &a, &c, &d);
        assert!(
            (abcd - bacd).abs() < TOLERANCE,
            "Expected (ab|cd) ERI ({abcd}) to be equal to (ba|cd) ERI ({bacd}) (seed: {seed})."
        );

        let badc = integrals::electron_repulsion(&b, &a, &d, &c);
        assert!(
            (abcd - badc).abs() < TOLERANCE,
            "Expected (ab|cd) ERI ({abcd}) to be equal to (ba|dc) ERI ({badc}) (seed: {seed})."
        );

        let cdab = integrals::electron_repulsion(&c, &d, &a, &b);
        assert!(
            (abcd - cdab).abs() < TOLERANCE,
            "Expected (ab|cd) ERI ({abcd}) to be equal to (cd|ab) ERI ({cdab}) (seed: {seed})."
        );

        let cdba = integrals::electron_repulsion(&c, &d, &b, &a);
        assert!(
            (abcd - cdba).abs() < TOLERANCE,
            "Expected (ab|cd) ERI ({abcd}) to be equal to (cd|ba) ERI ({cdba}) (seed: {seed})."
        );

        let dcab = integrals::electron_repulsion(&d, &c, &a, &b);
        assert!(
            (abcd - dcab).abs() < TOLERANCE,
            "Expected (ab|cd) ERI ({abcd}) to be equal to (dc|ab) ERI ({dcab}) (seed: {seed})."
        );

        let dcba = integrals::electron_repulsion(&d, &c, &b, &a);
        assert!(
            (abcd - dcba).abs() < TOLERANCE,
            "Expected (ab|cd) ERI ({abcd}) to be equal to (dc|ba) ERI ({dcba}) (seed: {seed})."
        );
    }
}
