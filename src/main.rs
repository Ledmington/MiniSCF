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
        for A in &self.contracted_gaussians {
            for B in &self.contracted_gaussians {
                s += compute_contracted_gaussians_overlap(&A, &B);
            }
        }
        s
    }
}

fn compute_contracted_gaussians_overlap(A: &ContractedGaussian, B: &ContractedGaussian) -> f64 {
    let mut s = 0.0;
    for (i, prim_a) in A.primitives.iter().enumerate() {
        let d_i = A.coefficients[i];
        for (j, prim_b) in B.primitives.iter().enumerate() {
            let d_j = B.coefficients[j];
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

    println!("Overlap:");
    println!(
        "S = | {} {} |",
        compute_contracted_gaussians_overlap(&phi_1, &phi_1),
        compute_contracted_gaussians_overlap(&phi_1, &phi_2)
    );
    println!(
        "    | {} {} |",
        compute_contracted_gaussians_overlap(&phi_2, &phi_1),
        compute_contracted_gaussians_overlap(&phi_2, &phi_2)
    );
}
