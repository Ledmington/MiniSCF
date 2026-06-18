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
    pub fn compute(&self, r: &Point) -> f64 {
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
    pub fn compute(&self, r: &Point) -> f64 {
        self.primitives.iter().map(|p| p.compute(r)).sum()
    }
}

fn get_normalization_term(alpha: f64) -> f64 {
    ((2.0 * alpha) / std::f64::consts::PI).powf(3.0 / 4.0)
}

struct BasisSet {
    contracted_gaussians: Vec<ContractedGaussian>,
}

impl BasisSet {
    fn contracted_overlap(&self) -> f64 {
        let mut s = 0.0;
        for a in &self.contracted_gaussians {
            for b in &self.contracted_gaussians {
                s += compute_contracted_gaussians_overlap(a, b);
            }
        }
        s
    }
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
        * (std::f64::consts::PI / p).powf(3.0 / 2.0)
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
    mu * (3.0 - 2.0 * mu * r2)
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
    0.5 * (std::f64::consts::PI / t).sqrt() * erf(t.sqrt())
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
        * ((2.0 * std::f64::consts::PI) / p)
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
    let t = ((p * q) / (p + q)) * r_ab_2;
    a.normalization_constant
        * b.normalization_constant
        * c.normalization_constant
        * d.normalization_constant
        * ((2.0 * std::f64::consts::PI.powf(5.0 / 2.0)) / (p * q * (p + q).sqrt()))
        * (-mu * r_ab_2).exp()
        * (-v * r_cd_2).exp()
        * boys_0(t)
}

fn main() {
    const R: f64 = 1.4; // bohr

    // 2 Hydrogen atoms
    let a = Point {
        x: 0.0,
        y: 0.0,
        z: -R / 2.0,
    };
    let b = Point {
        x: 0.0,
        y: 0.0,
        z: R / 2.0,
    };

    // Prepare the STO-3G basis
    let alpha: [f64; 3] = [3.42525091, 0.62391373, 0.16885540];
    let coefficients: [f64; 3] = [0.15432897, 0.53532814, 0.44463454];
    let phi_1 = ContractedGaussian {
        coefficients: coefficients.to_vec(),
        primitives: vec![
            PrimitiveGaussian {
                normalization_constant: get_normalization_term(alpha[0]),
                gaussian_exponent: alpha[0],
                center: a,
            },
            PrimitiveGaussian {
                normalization_constant: get_normalization_term(alpha[1]),
                gaussian_exponent: alpha[1],
                center: a,
            },
            PrimitiveGaussian {
                normalization_constant: get_normalization_term(alpha[2]),
                gaussian_exponent: alpha[2],
                center: a,
            },
        ],
    };
    let phi_2 = ContractedGaussian {
        coefficients: coefficients.to_vec(),
        primitives: vec![
            PrimitiveGaussian {
                normalization_constant: get_normalization_term(alpha[0]),
                gaussian_exponent: alpha[0],
                center: b,
            },
            PrimitiveGaussian {
                normalization_constant: get_normalization_term(alpha[1]),
                gaussian_exponent: alpha[1],
                center: b,
            },
            PrimitiveGaussian {
                normalization_constant: get_normalization_term(alpha[2]),
                gaussian_exponent: alpha[2],
                center: b,
            },
        ],
    };
    let sto_3g = BasisSet {
        contracted_gaussians: vec![phi_1, phi_2],
    };

    let s_11 = compute_contracted_gaussians_overlap(
        &sto_3g.contracted_gaussians[0],
        &sto_3g.contracted_gaussians[0],
    );
    let s_12 = compute_contracted_gaussians_overlap(
        &sto_3g.contracted_gaussians[0],
        &sto_3g.contracted_gaussians[1],
    );
    let s_21 = compute_contracted_gaussians_overlap(
        &sto_3g.contracted_gaussians[1],
        &sto_3g.contracted_gaussians[0],
    );
    let s_22 = compute_contracted_gaussians_overlap(
        &sto_3g.contracted_gaussians[1],
        &sto_3g.contracted_gaussians[1],
    );

    println!("Overlap:");
    println!("S = | {s_11:.6} {s_12:.6} |");
    println!("    | {s_21:.6} {s_22:.6} |");

    assert!((s_11 - 1.0).abs() < 1e-6);
    assert!((s_22 - 1.0).abs() < 1e-6);
    assert!((s_12 - s_21).abs() < 1e-6);

    println!();

    let t_11 = compute_contracted_gaussians_kinetic_energy(
        &sto_3g.contracted_gaussians[0],
        &sto_3g.contracted_gaussians[0],
    );
    let t_12 = compute_contracted_gaussians_kinetic_energy(
        &sto_3g.contracted_gaussians[0],
        &sto_3g.contracted_gaussians[1],
    );
    let t_21 = compute_contracted_gaussians_kinetic_energy(
        &sto_3g.contracted_gaussians[1],
        &sto_3g.contracted_gaussians[0],
    );
    let t_22 = compute_contracted_gaussians_kinetic_energy(
        &sto_3g.contracted_gaussians[1],
        &sto_3g.contracted_gaussians[1],
    );

    println!("Kinetic energy:");
    println!("T = | {t_11:.6} {t_12:.6} |");
    println!("    | {t_21:.6} {t_22:.6} |");

    assert!((t_12 - t_21).abs() < 1e-6);

    println!();

    let v_11 = compute_contracted_gaussians_nuclear_attraction(
        &sto_3g.contracted_gaussians[0],
        &sto_3g.contracted_gaussians[0],
    );
    let v_12 = compute_contracted_gaussians_nuclear_attraction(
        &sto_3g.contracted_gaussians[0],
        &sto_3g.contracted_gaussians[1],
    );
    let v_21 = compute_contracted_gaussians_nuclear_attraction(
        &sto_3g.contracted_gaussians[1],
        &sto_3g.contracted_gaussians[0],
    );
    let v_22 = compute_contracted_gaussians_nuclear_attraction(
        &sto_3g.contracted_gaussians[1],
        &sto_3g.contracted_gaussians[1],
    );

    println!("Nuclear attraction:");
    println!("V = | {v_11:.6} {v_12:.6} |");
    println!("    | {v_21:.6} {v_22:.6} |");

    assert!((v_12 - v_21).abs() < 1e-6);

    println!();

    let h_11 = t_11 + v_11;
    let h_12 = t_12 + v_12;
    let h_21 = t_21 + v_21;
    let h_22 = t_22 + v_22;

    println!("Hamiltonian:");
    println!("H = | {h_11:.6} {h_12:.6} |");
    println!("    | {h_21:.6} {h_22:.6} |");

    assert!((h_12 - h_21).abs() < 1e-6);

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
    for a in 0..sto_3g.contracted_gaussians.len() {
        for b in 0..sto_3g.contracted_gaussians.len() {
            for c in 0..sto_3g.contracted_gaussians.len() {
                for d in 0..sto_3g.contracted_gaussians.len() {
                    let A = &sto_3g.contracted_gaussians[a];
                    let B = &sto_3g.contracted_gaussians[b];
                    let C = &sto_3g.contracted_gaussians[c];
                    let D = &sto_3g.contracted_gaussians[d];

                    let mut sum = 0.0;

                    for i in 0..3 {
                        for j in 0..3 {
                            for k in 0..3 {
                                for l in 0..3 {
                                    let d_i = A.coefficients[i];
                                    let d_j = B.coefficients[j];
                                    let d_k = C.coefficients[k];
                                    let d_l = D.coefficients[l];

                                    sum += d_i
                                        * d_j
                                        * d_k
                                        * d_l
                                        * primitive_eri(
                                            &A.primitives[i],
                                            &B.primitives[j],
                                            &C.primitives[k],
                                            &D.primitives[l],
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
                    println!("({a}{b}|{c}{d}) = {}", eri[a][b][c][d]);
                }
            }
        }
    }

    for a in 0..sto_3g.contracted_gaussians.len() {
        for b in 0..sto_3g.contracted_gaussians.len() {
            for c in 0..sto_3g.contracted_gaussians.len() {
                for d in 0..sto_3g.contracted_gaussians.len() {
                    assert!(
                        (eri[a][b][c][d] - eri[b][a][c][d]).abs() < 1.0e-6,
                        "Expected eri[{a}][{b}][{c}][{d}] = {} and eri[{b}][{a}][{c}][{d}] = {} to be equal but they weren't.",
                        eri[a][b][c][d],
                        eri[b][a][c][d]
                    );
                    assert!(
                        (eri[a][b][c][d] - eri[a][b][d][c]).abs() < 1.0e-6,
                        "Expected eri[{a}][{b}][{c}][{d}] = {} and eri[{a}][{b}][{d}][{c}] = {} to be equal but they weren't.",
                        eri[a][b][c][d],
                        eri[a][b][d][c]
                    );
                    assert!(
                        (eri[a][b][c][d] - eri[c][d][a][b]).abs() < 1.0e-6,
                        "Expected eri[{a}][{b}][{c}][{d}] = {} and eri[{c}][{d}][{a}][{b}] = {} to be equal but they weren't.",
                        eri[a][b][c][d],
                        eri[c][d][a][b]
                    );
                }
            }
        }
    }
}
