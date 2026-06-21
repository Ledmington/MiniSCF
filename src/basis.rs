use ndarray::Array2;

use crate::point::Point;

// basis_set.rs
use ndarray::Array4;

use crate::integrals;

// These need pub(crate) so integrals.rs can name the types.
pub(crate) struct PrimitiveGaussian {
    pub(crate) normalization_constant: f64,
    pub(crate) gaussian_exponent: f64,
    pub(crate) center: Point,
}

impl PrimitiveGaussian {
    pub(crate) fn new(alpha: f64, center: Point) -> Self {
        PrimitiveGaussian {
            normalization_constant: get_normalization_term(alpha),
            gaussian_exponent: alpha,
            center,
        }
    }

    pub(crate) fn compute(&self, r: &Point) -> f64 {
        let dx = r.x - self.center.x;
        let dy = r.y - self.center.y;
        let dz = r.z - self.center.z;
        self.normalization_constant
            * (-(self.gaussian_exponent * (dx * dx + dy * dy + dz * dz))).exp()
    }
}

pub(crate) struct ContractedGaussian {
    pub(crate) coefficients: Vec<f64>,
    pub(crate) primitives: Vec<PrimitiveGaussian>,
}

impl ContractedGaussian {
    pub(crate) fn new(coefficients: &[f64], alpha: &[f64], center: &Point) -> Self {
        assert_eq!(coefficients.len(), alpha.len());
        ContractedGaussian {
            coefficients: coefficients.to_vec(),
            primitives: alpha
                .iter()
                .map(|&a| PrimitiveGaussian::new(a, *center))
                .collect(),
        }
    }

    pub(crate) fn compute(&self, r: &Point) -> f64 {
        self.primitives
            .iter()
            .zip(&self.coefficients)
            .map(|(p, &c)| c * p.compute(r))
            .sum()
    }
}

fn get_normalization_term(alpha: f64) -> f64 {
    use std::f64::consts::PI;
    ((2.0 * alpha) / PI).powf(3.0 / 4.0)
}

pub(crate) struct BasisSet {
    contracted_gaussians: Vec<ContractedGaussian>,
}

impl BasisSet {
    pub(crate) fn new(coefficients: &[f64], alpha: &[f64], centers: &[Point]) -> Self {
        BasisSet {
            contracted_gaussians: centers
                .iter()
                .map(|p| ContractedGaussian::new(coefficients, alpha, p))
                .collect(),
        }
    }

    pub(crate) fn num_contracted_gaussians(&self) -> usize {
        self.contracted_gaussians.len()
    }

    pub(crate) fn num_occupied_orbitals(&self) -> usize {
        1
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

    fn one_electron_matrix(
        &self,
        f: impl Fn(&ContractedGaussian, &ContractedGaussian) -> f64,
    ) -> Array2<f64> {
        let n = self.contracted_gaussians.len();
        let mut m = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                let val = f(&self.contracted_gaussians[i], &self.contracted_gaussians[j]);
                m[[i, j]] = val;
                m[[j, i]] = val;
            }
        }
        m
    }

    pub(crate) fn electron_repulsion_tensor(&self) -> Array4<f64> {
        let n = self.contracted_gaussians.len();
        let cg = &self.contracted_gaussians;
        let mut eri = Array4::zeros((n, n, n, n));
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    for d in 0..n {
                        eri[[a, b, c, d]] =
                            integrals::electron_repulsion(&cg[a], &cg[b], &cg[c], &cg[d]);
                    }
                }
            }
        }
        eri
    }

    pub(crate) fn compute(&self, orbital: usize, r: &Point, c: &Array2<f64>) -> f64 {
        self.contracted_gaussians
            .iter()
            .enumerate()
            .map(|(mu, cg)| c[[mu, orbital]] * cg.compute(r))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use rand::RngExt;

    use super::*;

    fn random_point_on_sphere(center: Point, radius: f64) -> Point {
        let mut rng = rand::rng();
        let z: f64 = rng.random_range(-1.0..=1.0); // z uniformly distributed in [-1, 1]
        let theta: f64 = rng.random_range(0.0..TAU); // azimuth uniformly distributed in [0, 2π)
        let xy = (1.0 - z * z).sqrt();
        Point {
            x: center.x + radius * xy * theta.cos(),
            y: center.y + radius * xy * theta.sin(),
            z: center.z + radius * z,
        }
    }

    #[test]
    fn primitive_gaussian_spherically_symmetric() {
        let center = Point {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        let g = PrimitiveGaussian::new(1.0, center);

        let p0 = random_point_on_sphere(center, 1.0);
        let f0 = g.compute(&p0);
        for _ in 0..1000 {
            let p = random_point_on_sphere(center, 1.0);
            let f = g.compute(&p);
            assert!((f - f0).abs() < 1e-12);
        }
    }

    #[test]
    fn primitive_gaussian_spreading() {
        let center = Point {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        let g1 = PrimitiveGaussian::new(1.0, center);
        let g2 = PrimitiveGaussian::new(2.0, center);
        let p = Point {
            x: 0.0,
            y: -1.0,
            z: -2.0,
        };
        assert!(g1.compute(&p) > g2.compute(&p));
    }

    #[test]
    fn contracted_gaussian_spherically_symmetric() {
        let center = Point {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        let g = ContractedGaussian::new(&[1.0, 2.0], &[3.0, 4.0], &center);

        let p0 = random_point_on_sphere(center, 1.0);
        let f0 = g.compute(&p0);
        for _ in 0..1000 {
            let p = random_point_on_sphere(center, 1.0);
            let f = g.compute(&p);
            assert!((f - f0).abs() < 1e-12);
        }
    }

    #[test]
    fn contracted_gaussian_spreading() {
        let center = Point {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        let g1 = ContractedGaussian::new(&[1.0, 2.0], &[3.0, 4.0], &center);
        let g2 = ContractedGaussian::new(&[1.0, 2.0], &[4.0, 5.0], &center);
        let p = Point {
            x: 0.0,
            y: -1.0,
            z: -2.0,
        };
        assert!(g1.compute(&p) > g2.compute(&p));
    }

    #[test]
    fn contracted_gaussian_coefficients() {
        let center = Point {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        let g1 = ContractedGaussian::new(&[1.0, 2.0], &[3.0, 4.0], &center);
        let g2 = ContractedGaussian::new(&[0.0, 1.0], &[3.0, 4.0], &center);
        let p = Point {
            x: 0.0,
            y: -1.0,
            z: -2.0,
        };
        assert!(g1.compute(&p) > g2.compute(&p));
    }
}
