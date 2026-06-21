use std::f64::consts::PI;

use ndarray::Array2;

use crate::point::Point;

struct PrimitiveGaussian {
    normalization_constant: f64,
    gaussian_exponent: f64,
    center: Point,
}

impl PrimitiveGaussian {
    fn new(alpha: f64, center: Point) -> Self {
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
        let r2 = dx * dx + dy * dy + dz * dz;
        self.normalization_constant * (-self.gaussian_exponent * r2).exp()
    }
}

struct ContractedGaussian {
    coefficients: Vec<f64>,
    primitives: Vec<PrimitiveGaussian>,
}

impl ContractedGaussian {
    fn new(coefficients: &[f64], alpha: &[f64], center: &Point) -> Self {
        assert_eq!(coefficients.len(), alpha.len());
        ContractedGaussian {
            coefficients: coefficients.to_vec(),
            primitives: alpha
                .iter()
                .map(|a| PrimitiveGaussian::new(*a, *center))
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
    ((2.0 * alpha) / PI).powf(3.0 / 4.0)
}

pub(crate) struct BasisSet {
    contracted_gaussians: Vec<ContractedGaussian>,
}

impl BasisSet {
    pub(crate) fn new(coefficients: &[f64], alpha: &[f64], centers: &[Point]) -> Self {
        assert_eq!(coefficients.len(), alpha.len());
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

    fn fill_one_electron_matrix(
        &self,
        m: &mut Array2<f64>,
        f: impl Fn(&ContractedGaussian, &ContractedGaussian) -> f64,
    ) {
        for (i, a) in self.contracted_gaussians.iter().enumerate() {
            for (j, b) in self.contracted_gaussians.iter().enumerate() {
                m[[i, j]] = f(a, b);
            }
        }
    }

    pub(crate) fn compute_contracted_gaussians_overlap(&self, m: &mut Array2<f64>) {
        self.fill_one_electron_matrix(m, Self::compute_overlap);
    }

    pub(crate) fn num_occupied_orbitals(&self) -> usize {
        1
    }

    fn compute_overlap(a: &ContractedGaussian, b: &ContractedGaussian) -> f64 {
        let mut s = 0.0;
        for (i, prim_a) in a.primitives.iter().enumerate() {
            let d_i = a.coefficients[i];
            for (j, prim_b) in b.primitives.iter().enumerate() {
                let d_j = b.coefficients[j];
                s += d_i * d_j * Self::compute_primitive_gaussians_overlap(prim_a, prim_b);
            }
        }
        s
    }

    fn compute_primitive_gaussians_overlap(a: &PrimitiveGaussian, b: &PrimitiveGaussian) -> f64 {
        let p = a.gaussian_exponent + b.gaussian_exponent;
        let mu = (a.gaussian_exponent * b.gaussian_exponent) / p;
        let r2 = a.center.sub(&b.center).norm_squared();
        a.normalization_constant
            * b.normalization_constant
            * (PI / p).powf(3.0 / 2.0)
            * (-mu * r2).exp()
    }

    pub(crate) fn compute_contracted_gaussians_kinetic_energy(&self, m: &mut Array2<f64>) {
        self.fill_one_electron_matrix(m, Self::compute_kinetic_energy);
    }

    fn compute_kinetic_energy(a: &ContractedGaussian, b: &ContractedGaussian) -> f64 {
        let mut s = 0.0;
        for (i, prim_a) in a.primitives.iter().enumerate() {
            let d_i = a.coefficients[i];
            for (j, prim_b) in b.primitives.iter().enumerate() {
                let d_j = b.coefficients[j];
                s += d_i * d_j * Self::compute_primitive_gaussians_kinetic_energy(prim_a, prim_b);
            }
        }
        s
    }

    fn compute_primitive_gaussians_kinetic_energy(
        a: &PrimitiveGaussian,
        b: &PrimitiveGaussian,
    ) -> f64 {
        let p = a.gaussian_exponent + b.gaussian_exponent;
        let mu = (a.gaussian_exponent * b.gaussian_exponent) / p;
        let r2 = a.center.sub(&b.center).norm_squared();
        a.normalization_constant
            * b.normalization_constant
            * (PI / p).powf(3.0 / 2.0)
            * (-mu * r2).exp()
            * mu
            * (3.0 - 2.0 * mu * r2)
    }

    pub(crate) fn compute_contracted_gaussians_nuclear_attraction(&self, m: &mut Array2<f64>) {
        self.fill_one_electron_matrix(m, Self::compute_nuclear_attraction);
    }

    fn compute_nuclear_attraction(a: &ContractedGaussian, b: &ContractedGaussian) -> f64 {
        let mut s = 0.0;
        for (i, prim_a) in a.primitives.iter().enumerate() {
            let d_i = a.coefficients[i];
            for (j, prim_b) in b.primitives.iter().enumerate() {
                let d_j = b.coefficients[j];
                s += d_i
                    * d_j
                    * Self::compute_primitive_gaussians_nuclear_attraction(prim_a, prim_b);
            }
        }
        s
    }

    fn compute_primitive_gaussians_nuclear_attraction(
        a: &PrimitiveGaussian,
        b: &PrimitiveGaussian,
    ) -> f64 {
        let p = a.gaussian_exponent + b.gaussian_exponent;
        let mu = (a.gaussian_exponent * b.gaussian_exponent) / p;
        let r2 = a.center.sub(&b.center).norm_squared();
        -(a.normalization_constant
            * b.normalization_constant
            * ((2.0 * PI) / p)
            * (-mu * r2).exp()
            * boys_0(p * r2))
    }

    pub(crate) fn compute_electron_repulsion(&self, eri: &mut [Vec<Vec<Vec<f64>>>]) {
        for (a, contr_gauss_a) in self.contracted_gaussians.iter().enumerate() {
            for (b, contr_gauss_b) in self.contracted_gaussians.iter().enumerate() {
                for (c, contr_gauss_c) in self.contracted_gaussians.iter().enumerate() {
                    for (d, contr_gauss_d) in self.contracted_gaussians.iter().enumerate() {
                        let mut sum = 0.0;

                        for (i, prim_i) in contr_gauss_a.primitives.iter().enumerate() {
                            for (j, prim_j) in contr_gauss_a.primitives.iter().enumerate() {
                                for (k, prim_k) in contr_gauss_a.primitives.iter().enumerate() {
                                    for (l, prim_l) in contr_gauss_a.primitives.iter().enumerate() {
                                        let d_i = contr_gauss_a.coefficients[i];
                                        let d_j = contr_gauss_b.coefficients[j];
                                        let d_k = contr_gauss_c.coefficients[k];
                                        let d_l = contr_gauss_d.coefficients[l];

                                        sum += d_i
                                            * d_j
                                            * d_k
                                            * d_l
                                            * Self::primitive_eri(prim_i, prim_j, prim_k, prim_l);
                                    }
                                }
                            }
                        }

                        eri[a][b][c][d] = sum;
                    }
                }
            }
        }
    }

    fn primitive_eri(
        a: &PrimitiveGaussian,
        b: &PrimitiveGaussian,
        c: &PrimitiveGaussian,
        d: &PrimitiveGaussian,
    ) -> f64 {
        let r_ab_2 = a.center.sub(&b.center).norm_squared();
        let r_cd_2 = c.center.sub(&d.center).norm_squared();
        let p = a.gaussian_exponent + b.gaussian_exponent;
        let q = c.gaussian_exponent + d.gaussian_exponent;
        let mu = (a.gaussian_exponent * b.gaussian_exponent) / p;
        let v = (c.gaussian_exponent * d.gaussian_exponent) / q;
        let p_center = Point {
            x: (a.gaussian_exponent * a.center.x + b.gaussian_exponent * b.center.x) / p,
            y: (a.gaussian_exponent * a.center.y + b.gaussian_exponent * b.center.y) / p,
            z: (a.gaussian_exponent * a.center.z + b.gaussian_exponent * b.center.z) / p,
        };
        let q_center = Point {
            x: (c.gaussian_exponent * c.center.x + d.gaussian_exponent * d.center.x) / q,
            y: (c.gaussian_exponent * c.center.y + d.gaussian_exponent * d.center.y) / q,
            z: (c.gaussian_exponent * c.center.z + d.gaussian_exponent * d.center.z) / q,
        };
        let r_pq_2 = p_center.sub(&q_center).norm_squared();
        let t = ((p * q) / (p + q)) * r_pq_2;
        a.normalization_constant
            * b.normalization_constant
            * c.normalization_constant
            * d.normalization_constant
            * ((2.0 * PI.powf(5.0 / 2.0)) / (p * q * (p + q).sqrt()))
            * (-mu * r_ab_2).exp()
            * (-v * r_cd_2).exp()
            * boys_0(t)
    }

    pub(crate) fn compute(&self, i: usize, r: &Point, c: &Array2<f64>) -> f64 {
        let mut sum = 0.0;
        for (mu, contr_gauss) in self.contracted_gaussians.iter().enumerate() {
            sum += c[[mu, i]] * contr_gauss.compute(r);
        }
        sum
    }
}

fn erf(x: f64) -> f64 {
    // constants
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    // Abramowitz-Stegun formula
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * (-x * x).exp();

    sign * y
}

fn boys_0(t: f64) -> f64 {
    if t < 1.0e-8 {
        return 1.0;
    }
    0.5 * (PI / t).sqrt() * erf(t.sqrt())
}
