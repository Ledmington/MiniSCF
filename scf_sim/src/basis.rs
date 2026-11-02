use crate::integrals;
use crate::integrals::nuclear_attraction;
use ndarray::Array2;
use ndarray::Array4;
use scf_core::Atom;
use scf_core::point::Point;
use std::{f64::consts::PI, sync::Arc};

#[derive(Debug, Clone, PartialEq)]
pub struct PrimitiveGaussian {
    /// Raw, does not include normalization
    contraction_coefficient: f64,
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

    /// The same primitive with its contraction coefficient multiplied by `factor`.
    pub(crate) fn with_scaled_coefficient(&self, factor: f64) -> Self {
        PrimitiveGaussian {
            contraction_coefficient: self.contraction_coefficient * factor,
            ..self.clone()
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

    /// The electronic contribution to the electrostatic potential at `point`, i.e.
    /// sum_{mu,nu} P[mu,nu] * (mu nu | 1/|r - point|), given the AO density matrix.
    pub(crate) fn electronic_potential(&self, point: &Point, density: &Array2<f64>) -> f64 {
        let n = self.num_contracted_gaussians();
        let mut sum = 0.0;

        for i in 0..n {
            sum += density[[i, i]]
                * integrals::coulomb_potential(&self.functions[i], &self.functions[i], point);
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let val =
                    integrals::coulomb_potential(&self.functions[i], &self.functions[j], point);
                // density and the integral are both symmetric in (i, j), so the off-diagonal
                // terms contribute twice.
                sum += 2.0 * density[[i, j]] * val;
            }
        }

        sum
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

        // Iterate over all unique pairs (i, j)
        for i in 0..n {
            for j in 0..=i {
                // Iterate over all unique pairs (k, l)
                for k in 0..n {
                    for l in 0..=k {
                        // Discard symmetric quadruplets
                        if (i * (i + 1)) / 2 + j < (k * (k + 1)) / 2 + l {
                            continue;
                        }

                        let val = integrals::electron_repulsion(
                            &self.functions[i],
                            &self.functions[j],
                            &self.functions[k],
                            &self.functions[l],
                        );

                        eri[[i, j, k, l]] = val;
                        eri[[i, j, l, k]] = val;
                        eri[[j, i, k, l]] = val;
                        eri[[j, i, l, k]] = val;
                        eri[[k, l, i, j]] = val;
                        eri[[k, l, j, i]] = val;
                        eri[[l, k, i, j]] = val;
                        eri[[l, k, j, i]] = val;
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
    D,
    F,
}

impl AngularMomentum {
    /// The Cartesian components `(lx, ly, lz)` with `lx + ly + lz = l` of a shell, in the
    /// conventional order (e.g. xx, xy, xz, yy, yz, zz for d and xxx, xxy, xxz, xyy, xyz,
    /// xzz, yyy, yyz, yzz, zzz for f). Shells are Cartesian: a d shell has 6 components and
    /// an f shell 10, not the 5 and 7 of the spherical (pure) harmonics.
    pub(crate) fn cartesian_components(&self) -> Vec<(u8, u8, u8)> {
        let l = match self {
            AngularMomentum::S => 0,
            AngularMomentum::P => 1,
            AngularMomentum::D => 2,
            AngularMomentum::F => 3,
        };
        let mut components = Vec::new();
        for lx in (0..=l).rev() {
            for ly in (0..=(l - lx)).rev() {
                components.push((lx, ly, l - lx - ly));
            }
        }
        components
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Shell {
    pub(crate) center: Point,
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
            .map(|p| self.normalized_coefficient(p) * (-p.alpha * r2).exp())
            .sum();

        let (lx, ly, lz) = self.angular_momentum;
        let angular = dx.powi(lx.into()) * dy.powi(ly.into()) * dz.powi(lz.into());

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
    fn test_cartesian_components() {
        assert_eq!(AngularMomentum::S.cartesian_components(), vec![(0, 0, 0)]);
        assert_eq!(
            AngularMomentum::P.cartesian_components(),
            vec![(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        );
        assert_eq!(
            AngularMomentum::D.cartesian_components(),
            vec![
                (2, 0, 0),
                (1, 1, 0),
                (1, 0, 1),
                (0, 2, 0),
                (0, 1, 1),
                (0, 0, 2)
            ]
        );
        assert_eq!(
            AngularMomentum::F.cartesian_components(),
            vec![
                (3, 0, 0),
                (2, 1, 0),
                (2, 0, 1),
                (1, 2, 0),
                (1, 1, 1),
                (1, 0, 2),
                (0, 3, 0),
                (0, 2, 1),
                (0, 1, 2),
                (0, 0, 3)
            ]
        );
    }

    /// `(2n - 1)!!` as a float, with `(-1)!! = 1`.
    fn odd_double_factorial(n: u8) -> f64 {
        (1..=n).map(|k| f64::from(2 * k - 1)).product()
    }

    #[test]
    fn test_overlap_f() {
        let seed = rand::rng().random();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let center = random_center(&mut rng);
        let alpha = rng.random_range(0.1..5.0);

        let f = single_primitive_shell(alpha, center, &AngularMomentum::F);
        let p = single_primitive_shell(alpha, center, &AngularMomentum::P);

        // Normalized Cartesian Gaussians with the same exponent and center: the overlap
        // factorizes over the axes and, with a_i + b_i = 2 m_i along each axis, it is
        // prod_i (2 m_i - 1)!! / sqrt((2 a_i - 1)!! (2 b_i - 1)!!), or 0 if any a_i + b_i
        // is odd. E.g. <f_xxx|f_xxx> = 1, <f_xxx|f_xyy> = 3/sqrt(15 * 3) and
        // <p_x|f_xxx> = 3/sqrt(15).
        let expected_overlap = |a: (u8, u8, u8), b: (u8, u8, u8)| {
            let mut value = 1.0;
            for (ai, bi) in [(a.0, b.0), (a.1, b.1), (a.2, b.2)] {
                if (ai + bi) % 2 == 1 {
                    return 0.0;
                }
                value *= odd_double_factorial((ai + bi) / 2)
                    / (odd_double_factorial(ai) * odd_double_factorial(bi)).sqrt();
            }
            value
        };

        for a in &f {
            for b in f.iter().chain(&p) {
                let expected = expected_overlap(a.angular_momentum, b.angular_momentum);
                let actual = integrals::overlap(a, b);
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "Expected overlap between {:?} and {:?} to be {expected} but was {actual} (seed: {seed}).",
                    a.angular_momentum,
                    b.angular_momentum
                );
            }
        }
    }

    #[test]
    fn test_compute_f() {
        let alpha: f64 = 0.8;
        let center = Point::new(0.1, -0.2, 0.3);
        let r = Point::new(0.6, 0.4, -0.5);
        let (dx, dy, dz) = (0.5, 0.6, -0.8);
        let radial = (-alpha * (dx * dx + dy * dy + dz * dz)).exp();
        let base = (2.0 * alpha / PI).powf(0.75) * (4.0 * alpha).powf(1.5);

        let f = single_primitive_shell(alpha, center, &AngularMomentum::F);
        // xxx: N = base / sqrt(15), xxy: N = base / sqrt(3), xyz: N = base.
        for (index, expected_angular) in [
            (0, base / 15.0_f64.sqrt() * dx * dx * dx),
            (1, base / 3.0_f64.sqrt() * dx * dx * dy),
            (4, base * dx * dy * dz),
            (9, base / 15.0_f64.sqrt() * dz * dz * dz),
        ] {
            let actual = f[index].compute(&r);
            let expected_value = expected_angular * radial;
            assert!(
                (actual - expected_value).abs() < 1e-12,
                "Expected {:?} at {r:?} to be {expected_value} but was {actual}.",
                f[index].angular_momentum
            );
        }
    }

    /// One basis function per Cartesian component of a single-primitive shell.
    fn single_primitive_shell(
        alpha: f64,
        center: Point,
        angular: &AngularMomentum,
    ) -> Vec<BasisFunction> {
        let shell = Arc::new(Shell {
            center,
            primitives: vec![PrimitiveGaussian::new(1.0, alpha, center)],
        });
        angular
            .cartesian_components()
            .into_iter()
            .map(|angular_momentum| BasisFunction {
                shell: Arc::clone(&shell),
                angular_momentum,
            })
            .collect()
    }

    fn random_center(rng: &mut ChaCha8Rng) -> Point {
        Point::new(
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
        )
    }

    #[test]
    fn test_overlap_d() {
        let seed = rand::rng().random();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let center = random_center(&mut rng);
        let alpha = rng.random_range(0.1..5.0);

        let d = single_primitive_shell(alpha, center, &AngularMomentum::D);
        let s = &single_primitive_shell(alpha, center, &AngularMomentum::S)[0];

        // Analytic overlaps of normalized Cartesian Gaussians with the same exponent and
        // center: <d_ii|d_ii> = <d_ij|d_ij> = 1, <d_ii|d_jj> = 1/3, <s|d_ii> = 1/sqrt(3) and
        // every pair with an odd power along some axis vanishes.
        for a in &d {
            for b in &d {
                let (ax, ay, az) = a.angular_momentum;
                let (bx, by, bz) = b.angular_momentum;
                let expected = if a.angular_momentum == b.angular_momentum {
                    1.0
                } else if (ax + bx) % 2 == 1 || (ay + by) % 2 == 1 || (az + bz) % 2 == 1 {
                    0.0
                } else {
                    1.0 / 3.0
                };
                let actual = integrals::overlap(a, b);
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "Expected overlap between {:?} and {:?} to be {expected} but was {actual} (seed: {seed}).",
                    a.angular_momentum,
                    b.angular_momentum
                );
            }

            let expected = if [
                a.angular_momentum.0,
                a.angular_momentum.1,
                a.angular_momentum.2,
            ]
            .contains(&2)
            {
                1.0 / 3.0_f64.sqrt()
            } else {
                0.0
            };
            let actual = integrals::overlap(s, a);
            assert!(
                (actual - expected).abs() < 1e-10,
                "Expected overlap between s and {:?} to be {expected} but was {actual} (seed: {seed}).",
                a.angular_momentum
            );
        }
    }

    #[test]
    fn test_single_normalized_primitive_kinetic_all_angular_momenta() {
        let seed = rand::rng().random();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let center = random_center(&mut rng);
        let alpha = rng.random_range(0.1..5.0);

        // The kinetic energy of a normalized Cartesian Gaussian is a sum over the axes of
        // the 1D terms: for a normalized x^n exp(-alpha x^2),
        // <T> = alpha * (4n - 1) / (2 * (2n - 1)), i.e. alpha/2, 3alpha/2 and 7alpha/6 for
        // n = 0, 1, 2 and 11alpha/10 for n = 3. So it depends on how l is split: d_xx has
        // 13alpha/6, d_xy 7alpha/2.
        let one_dimensional = |n: u8| {
            let order = f64::from(n);
            alpha * (4.0 * order - 1.0) / (2.0 * (2.0 * order - 1.0))
        };
        for angular in [
            AngularMomentum::S,
            AngularMomentum::P,
            AngularMomentum::D,
            AngularMomentum::F,
        ] {
            for bf in single_primitive_shell(alpha, center, &angular) {
                let actual = integrals::kinetic_energy(&bf, &bf);
                let (lx, ly, lz) = bf.angular_momentum;
                let expected = one_dimensional(lx) + one_dimensional(ly) + one_dimensional(lz);
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "Expected kinetic energy of {:?} to be {expected} but was {actual} (seed: {seed}).",
                    bf.angular_momentum
                );
            }
        }
    }

    #[test]
    fn test_single_normalized_primitive_nuclear_attraction_same_center() {
        let seed = rand::rng().random();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let center = random_center(&mut rng);
        let alpha: f64 = rng.random_range(0.1..5.0);
        let nucleus = Atom {
            element: element::HYDROGEN,
            position: center,
            charge: 1,
        };

        // For a normalized Cartesian Gaussian with total angular momentum l and a unit
        // charge at its center, <1/r> = Gamma(l + 1) / Gamma(l + 3/2) * sqrt(2 alpha): the
        // angular parts cancel out between the numerator and the normalization.
        for (angular, gamma_ratio) in [
            (AngularMomentum::S, 2.0 / PI.sqrt()),
            (AngularMomentum::P, 4.0 / (3.0 * PI.sqrt())),
            (AngularMomentum::D, 16.0 / (15.0 * PI.sqrt())),
            (AngularMomentum::F, 32.0 / (35.0 * PI.sqrt())),
        ] {
            for bf in single_primitive_shell(alpha, center, &angular) {
                let actual = nuclear_attraction(&bf, &bf, std::slice::from_ref(&nucleus));
                let expected = -gamma_ratio * (2.0 * alpha).sqrt();
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "Expected nuclear attraction of {:?} to be {expected} but was {actual} (seed: {seed}).",
                    bf.angular_momentum
                );
            }
        }
    }

    #[test]
    fn test_compute_d() {
        let alpha: f64 = 0.8;
        let center = Point::new(0.1, -0.2, 0.3);
        let r = Point::new(0.6, 0.4, -0.5);
        let (dx, dy, dz) = (0.5, 0.6, -0.8);
        let radial = (-alpha * (dx * dx + dy * dy + dz * dz)).exp();
        let n_ii = (2.0 * alpha / PI).powf(0.75) * 4.0 * alpha / 3.0_f64.sqrt();
        let n_ij = (2.0 * alpha / PI).powf(0.75) * 4.0 * alpha;

        let expected = [
            n_ii * dx * dx,
            n_ij * dx * dy,
            n_ij * dx * dz,
            n_ii * dy * dy,
            n_ij * dy * dz,
            n_ii * dz * dz,
        ];
        let d = single_primitive_shell(alpha, center, &AngularMomentum::D);
        for (bf, expected_angular) in d.iter().zip(expected) {
            let actual = bf.compute(&r);
            let expected_value = expected_angular * radial;
            assert!(
                (actual - expected_value).abs() < 1e-12,
                "Expected {:?} at {r:?} to be {expected_value} but was {actual}.",
                bf.angular_momentum
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

    #[test]
    fn test_single_normalized_s_primitive_electron_repulsion() {
        let seed = rand::rng().random();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let center = Point::new(
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
        );

        let alpha = 0.5;
        let primitive = PrimitiveGaussian::new(1.0, alpha, center);
        let shell = Shell {
            center,
            primitives: vec![primitive],
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
                primitives: vec![PrimitiveGaussian::new(1.0, alpha, center)],
            }),
            angular_momentum: (0, 0, 0),
        };

        let alpha_1 = 0.5;
        let alpha_2 = 1.0;
        let alpha_3 = 1.5;
        let alpha_4 = 2.0;

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
                    1.0,
                    rng.random_range(0.1..5.0),
                    Point::new(
                        rng.random_range(-10.0..10.0),
                        rng.random_range(-10.0..10.0),
                        rng.random_range(-10.0..10.0),
                    ),
                )],
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
