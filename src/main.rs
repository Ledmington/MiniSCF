mod linalg;
mod matrix;

use std::f64::consts::PI;

use crate::matrix::Matrix;

#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
    z: f64,
}

impl Point {
    fn sub(&self, p: &Point) -> Point {
        Point {
            x: self.x - p.x,
            y: self.y - p.y,
            z: self.z - p.z,
        }
    }

    fn norm_squared(&self) -> f64 {
        self.x.powi(2) + self.y.powi(2) + self.z.powi(2)
    }
}

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

    // pub fn compute(&self, r: &Point) -> f64 {
    //     let dx = r.x - self.center.x;
    //     let dy = r.y - self.center.y;
    //     let dz = r.z - self.center.z;
    //     let r2 = dx * dx + dy * dy + dz * dz;
    //     self.normalization_constant * (-self.gaussian_exponent * r2).exp()
    // }
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

    // pub fn compute(&self, r: &Point) -> f64 {
    //     self.primitives.iter().map(|p| p.compute(r)).sum()
    // }
}

fn get_normalization_term(alpha: f64) -> f64 {
    ((2.0 * alpha) / PI).powf(3.0 / 4.0)
}

struct BasisSet {
    contracted_gaussians: Vec<ContractedGaussian>,
}

impl BasisSet {
    fn new(coefficients: &[f64], alpha: &[f64], centers: &[Point]) -> Self {
        assert_eq!(coefficients.len(), alpha.len());
        BasisSet {
            contracted_gaussians: centers
                .iter()
                .map(|p| ContractedGaussian::new(coefficients, alpha, p))
                .collect(),
        }
    }

    // fn contracted_overlap(&self) -> f64 {
    //     let mut s = 0.0;
    //     for a in &self.contracted_gaussians {
    //         for b in &self.contracted_gaussians {
    //             s += compute_contracted_gaussians_overlap(a, b);
    //         }
    //     }
    //     s
    // }
}

fn compute_contracted_gaussians_overlap(a: &ContractedGaussian, b: &ContractedGaussian) -> f64 {
    let mut s = 0.0;
    for (i, prim_a) in a.primitives.iter().enumerate() {
        let d_i = a.coefficients[i];
        for (j, prim_b) in b.primitives.iter().enumerate() {
            let d_j = b.coefficients[j];
            s += d_i * d_j * compute_primitive_gaussians_overlap(prim_a, prim_b);
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

fn compute_contracted_gaussians_kinetic_energy(
    a: &ContractedGaussian,
    b: &ContractedGaussian,
) -> f64 {
    let mut s = 0.0;
    for (i, prim_a) in a.primitives.iter().enumerate() {
        let d_i = a.coefficients[i];
        for (j, prim_b) in b.primitives.iter().enumerate() {
            let d_j = b.coefficients[j];
            s += d_i * d_j * compute_primitive_gaussians_kinetic_energy(prim_a, prim_b);
        }
    }
    s
}

fn compute_primitive_gaussians_kinetic_energy(a: &PrimitiveGaussian, b: &PrimitiveGaussian) -> f64 {
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

fn compute_contracted_gaussians_nuclear_attraction(
    a: &ContractedGaussian,
    b: &ContractedGaussian,
) -> f64 {
    let mut s = 0.0;
    for (i, prim_a) in a.primitives.iter().enumerate() {
        let d_i = a.coefficients[i];
        for (j, prim_b) in b.primitives.iter().enumerate() {
            let d_j = b.coefficients[j];
            s += d_i * d_j * compute_primitive_gaussians_nuclear_attraction(prim_a, prim_b);
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

macro_rules! assert_approx_eq {
    ($left:expr, $right:expr, $tol:expr) => {{
        let left = $left;
        let right = $right;
        let tol = $tol;

        assert!(
            (left - right).abs() <= tol,
            "assertion failed: |{} - {}| <= {}\nleft: {}\nright: {}",
            left,
            right,
            tol,
            left,
            right,
        );
    }};
}

fn main() {
    const R: f64 = 1.4; // bohr

    // Prepare the STO-3G basis
    let sto_3g = BasisSet::new(
        &[0.15432897, 0.53532814, 0.44463454],
        &[3.42525091, 0.62391373, 0.16885540],
        // 2 Hydrogen atoms
        &[
            Point {
                x: 0.0,
                y: 0.0,
                z: -R / 2.0,
            },
            Point {
                x: 0.0,
                y: 0.0,
                z: R / 2.0,
            },
        ],
    );

    let s = Matrix::foreach(
        &sto_3g.contracted_gaussians,
        compute_contracted_gaussians_overlap,
    );

    println!("Overlap (S):");
    println!("{s}");

    assert_approx_eq!(s[0][0], 1.0, 1e-6);
    assert_approx_eq!(s[1][1], 1.0, 1e-6);
    assert_approx_eq!(s[0][1], s[1][0], 1e-6);

    println!();

    let t = Matrix::foreach(
        &sto_3g.contracted_gaussians,
        compute_contracted_gaussians_kinetic_energy,
    );

    println!("Kinetic energy (T):");
    println!("{t}");

    assert_approx_eq!(t[0][1], t[1][0], 1e-6);

    println!();

    let v = Matrix::foreach(
        &sto_3g.contracted_gaussians,
        compute_contracted_gaussians_nuclear_attraction,
    );

    println!("Nuclear attraction (V):");
    println!("{v}");

    assert_approx_eq!(v[0][1], v[1][0], 1e-6);

    println!();

    let h = t + v;

    println!("Hamiltonian (H):");
    println!("{h}");

    assert_approx_eq!(h[0][1], h[1][0], 1e-6);

    println!();

    println!("Electron Repulsion Integrals:");
    let mut eri: Vec<Vec<Vec<Vec<f64>>>> = vec![
        vec![
            vec![
                vec![0.0; sto_3g.contracted_gaussians.len()];
                sto_3g.contracted_gaussians.len()
            ];
            sto_3g.contracted_gaussians.len()
        ];
        sto_3g.contracted_gaussians.len()
    ];
    for (a, contr_gauss_a) in sto_3g.contracted_gaussians.iter().enumerate() {
        for (b, contr_gauss_b) in sto_3g.contracted_gaussians.iter().enumerate() {
            for (c, contr_gauss_c) in sto_3g.contracted_gaussians.iter().enumerate() {
                for (d, contr_gauss_d) in sto_3g.contracted_gaussians.iter().enumerate() {
                    let mut sum = 0.0;

                    for i in 0..3 {
                        for j in 0..3 {
                            for k in 0..3 {
                                for l in 0..3 {
                                    let d_i = contr_gauss_a.coefficients[i];
                                    let d_j = contr_gauss_b.coefficients[j];
                                    let d_k = contr_gauss_c.coefficients[k];
                                    let d_l = contr_gauss_d.coefficients[l];

                                    sum += d_i
                                        * d_j
                                        * d_k
                                        * d_l
                                        * primitive_eri(
                                            &contr_gauss_a.primitives[i],
                                            &contr_gauss_b.primitives[j],
                                            &contr_gauss_c.primitives[k],
                                            &contr_gauss_d.primitives[l],
                                        );
                                }
                            }
                        }
                    }

                    eri[a][b][c][d] = sum;
                }
            }
        }
    }

    for a in 0..sto_3g.contracted_gaussians.len() {
        for b in 0..sto_3g.contracted_gaussians.len() {
            for c in 0..sto_3g.contracted_gaussians.len() {
                for d in 0..sto_3g.contracted_gaussians.len() {
                    println!("⟨{a}{b}|{c}{d}⟩ = {}", eri[a][b][c][d]);
                }
            }
        }
    }

    println!();

    for a in 0..sto_3g.contracted_gaussians.len() {
        for b in 0..sto_3g.contracted_gaussians.len() {
            for c in 0..sto_3g.contracted_gaussians.len() {
                for d in 0..sto_3g.contracted_gaussians.len() {
                    assert_approx_eq!(eri[a][b][c][d], eri[b][a][c][d], 1.0e-6);
                    assert_approx_eq!(eri[a][b][c][d], eri[a][b][d][c], 1.0e-6);
                    assert_approx_eq!(eri[a][b][c][d], eri[c][d][a][b], 1.0e-6);
                }
            }
        }
    }

    let density = Matrix::zero(2, 2);
    println!("Density (P):");
    println!("{density}");
    println!();

    let f = h.clone();
    println!("Fock (F):");
    println!("{f}");
    println!();

    let mut u = Matrix::identity(2, 2);
    let mut d = Matrix::zero(2, 2);
    linalg::factorize(&s, &mut u, &mut d);

    assert!((s.clone() - u.clone() * &d * &u.transposed()).is_zero());

    println!("D:");
    println!("{}", d);
    println!();

    for i in 0..d.rows() {
        d[i][i] = 1.0 / d[i][i].sqrt();
    }

    println!("D^(-1/2):");
    println!("{}", d);
    println!();

    let x = u.clone() * &d * &u.transposed();
    println!("X:");
    println!("{}", x);
    println!();

    assert!(x.is_symmetric());
    assert!((Matrix::identity(d.rows(), d.rows()) - x.transposed() * &s * &x).is_zero()); // this assertion fails

    let h_prime = x.transposed() * &h * &x;

    println!("H':");
    println!("{}", h_prime);
    println!();
}
