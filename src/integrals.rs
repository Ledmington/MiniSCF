use std::f64::consts::PI;

use crate::basis::{BasisFunction, PrimitiveGaussian};

pub(crate) fn overlap(a: &BasisFunction, b: &BasisFunction) -> f64 {
    contracted_pair(a, b, primitive_overlap)
}

pub(crate) fn kinetic_energy(a: &BasisFunction, b: &BasisFunction) -> f64 {
    contracted_pair(a, b, primitive_kinetic_energy)
}

pub(crate) fn nuclear_attraction(a: &BasisFunction, b: &BasisFunction) -> f64 {
    contracted_pair(a, b, primitive_nuclear_attraction)
}

pub(crate) fn electron_repulsion(
    a: &BasisFunction,
    b: &BasisFunction,
    c: &BasisFunction,
    d: &BasisFunction,
) -> f64 {
    let mut sum = 0.0;
    for prim_a in &a.shell.primitives {
        for prim_b in &b.shell.primitives {
            for prim_c in &c.shell.primitives {
                for prim_d in &d.shell.primitives {
                    sum += prim_a.contraction_coefficient()
                        * prim_b.contraction_coefficient()
                        * prim_c.contraction_coefficient()
                        * prim_d.contraction_coefficient()
                        * primitive_eri(&prim_a, &prim_b, &prim_c, &prim_d);
                }
            }
        }
    }
    sum
}

// ------ contracted helper -----------------------------------------------

fn contracted_pair(
    a: &BasisFunction,
    b: &BasisFunction,
    f: impl Fn(&PrimitiveGaussian, &PrimitiveGaussian) -> f64,
) -> f64 {
    let mut sum = 0.0;

    for pa in &a.shell.primitives {
        for pb in &b.shell.primitives {
            sum += pa.contraction_coefficient() * pb.contraction_coefficient() * f(pa, pb);
        }
    }

    sum
}

// ------ primitive integrals ---------------------------------------------

fn primitive_overlap(a: &PrimitiveGaussian, b: &PrimitiveGaussian) -> f64 {
    let (p, mu, r2) = gaussian_pair_params(a, b);
    (PI / p).powf(1.5) * (-mu * r2).exp()
}

fn primitive_kinetic_energy(a: &PrimitiveGaussian, b: &PrimitiveGaussian) -> f64 {
    let (p, mu, r2) = gaussian_pair_params(a, b);
    (PI / p).powf(1.5) * (-mu * r2).exp() * mu * (3.0 - 2.0 * mu * r2)
}

fn primitive_nuclear_attraction(a: &PrimitiveGaussian, b: &PrimitiveGaussian) -> f64 {
    let (p, mu, r2) = gaussian_pair_params(a, b);
    -((2.0 * PI / p) * (-mu * r2).exp() * boys_0(p * r2))
}

fn primitive_eri(
    a: &PrimitiveGaussian,
    b: &PrimitiveGaussian,
    c: &PrimitiveGaussian,
    d: &PrimitiveGaussian,
) -> f64 {
    let r_ab_2 = a.center().sub(&b.center()).norm_squared();
    let r_cd_2 = c.center().sub(&d.center()).norm_squared();
    let p = a.alpha() + b.alpha();
    let q = c.alpha() + d.alpha();
    let mu = (a.alpha() * b.alpha()) / p;
    let v = (c.alpha() * d.alpha()) / q;
    let p_center = weighted_center(a, b, p);
    let q_center = weighted_center(c, d, q);
    let t = (p * q / (p + q)) * p_center.sub(&q_center).norm_squared();

    a.contraction_coefficient()
        * b.contraction_coefficient()
        * c.contraction_coefficient()
        * d.contraction_coefficient()
        * (2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt()))
        * (-mu * r_ab_2).exp()
        * (-v * r_cd_2).exp()
        * boys_0(t)
}

// ------ shared geometry helpers -----------------------------------------

/// Returns (p, μ, |R_A - R_B|²) for a primitive pair.
fn gaussian_pair_params(a: &PrimitiveGaussian, b: &PrimitiveGaussian) -> (f64, f64, f64) {
    let p = a.alpha() + b.alpha();
    let mu = a.alpha() * b.alpha() / p;
    let r2 = a.center().sub(&b.center()).norm_squared();
    (p, mu, r2)
}

fn weighted_center(a: &PrimitiveGaussian, b: &PrimitiveGaussian, p: f64) -> crate::point::Point {
    crate::point::Point {
        x: (a.alpha() * a.center().x + b.alpha() * b.center().x) / p,
        y: (a.alpha() * a.center().y + b.alpha() * b.center().y) / p,
        z: (a.alpha() * a.center().z + b.alpha() * b.center().z) / p,
    }
}

// ------ special functions -----------------------------------------------

fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
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
