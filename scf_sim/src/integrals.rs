use scf_core::Point;
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
                        * primitive_eri(prim_a, prim_b, prim_c, prim_d);
                }
            }
        }
    }
    sum
}

/// Contracted helper
fn contracted_pair(
    a: &BasisFunction,
    b: &BasisFunction,
    f: impl Fn(&PrimitiveGaussian, &PrimitiveGaussian, &(u8, u8, u8), &(u8, u8, u8)) -> f64,
) -> f64 {
    let mut sum = 0.0;

    for pa in &a.shell.primitives {
        for pb in &b.shell.primitives {
            sum += pa.contraction_coefficient()
                * pb.contraction_coefficient()
                * f(pa, pb, &a.angular_momentum, &b.angular_momentum);
        }
    }

    sum
}

/// McMurchie-Davidson recurrence
fn overlap_1d(ia: u8, ib: u8, pa: f64, pb: f64, p: f64) -> f64 {
    fn e(i: i32, j: i32, t: i32, pa: f64, pb: f64, p: f64) -> f64 {
        if t < 0 || t > i + j {
            return 0.0;
        }

        if i == 0 && j == 0 {
            return if t == 0 { 1.0 } else { 0.0 };
        }

        if i > 0 {
            return pa * e(i - 1, j, t, pa, pb, p)
                + (1.0 / (2.0 * p)) * e(i - 1, j, t - 1, pa, pb, p)
                + ((t + 1) as f64) * e(i - 1, j, t + 1, pa, pb, p);
        }

        pb * e(i, j - 1, t, pa, pb, p)
            + (1.0 / (2.0 * p)) * e(i, j - 1, t - 1, pa, pb, p)
            + ((t + 1) as f64) * e(i, j - 1, t + 1, pa, pb, p)
    }

    (PI / p).sqrt() * e(ia as i32, ib as i32, 0, pa, pb, p)
}

fn primitive_overlap(
    a: &PrimitiveGaussian,
    b: &PrimitiveGaussian,
    angular_momentum_a: &(u8, u8, u8),
    angular_momentum_b: &(u8, u8, u8),
) -> f64 {
    let (p, mu, r2) = gaussian_pair_params(a, b);
    let center = weighted_center(a, b, p);
    let pa = center.sub(&a.center()).coordinates();
    let pb = center.sub(&b.center()).coordinates();

    let ex = overlap_1d(angular_momentum_a.0, angular_momentum_b.0, pa[0], pb[0], p);
    let ey = overlap_1d(angular_momentum_a.1, angular_momentum_b.1, pa[1], pb[1], p);
    let ez = overlap_1d(angular_momentum_a.2, angular_momentum_b.2, pa[2], pb[2], p);
    ex * ey * ez * (-mu * r2).exp()
}

fn shift(l: &(u8, u8, u8), axis: usize, delta: i8) -> Option<(u8, u8, u8)> {
    let mut out = [l.0 as i16, l.1 as i16, l.2 as i16];
    out[axis] += delta as i16;

    if out.iter().any(|&x| x < 0) {
        return None;
    }

    Some((out[0] as u8, out[1] as u8, out[2] as u8))
}

fn primitive_kinetic_energy(
    a: &PrimitiveGaussian,
    b: &PrimitiveGaussian,
    la: &(u8, u8, u8),
    lb: &(u8, u8, u8),
) -> f64 {
    let beta = b.alpha();

    let mut value = 0.0;

    for axis in 0..3 {
        let l = match axis {
            0 => lb.0,
            1 => lb.1,
            _ => lb.2,
        } as f64;

        // j(j-1) S(j-2)
        if l >= 2.0 {
            if let Some(lb2) = shift(lb, axis, -2) {
                value += l * (l - 1.0) * primitive_overlap(a, b, la, &lb2);
            }
        }

        // -2β(2j+1) S(j)
        value += -2.0 * beta * (2.0 * l + 1.0) * primitive_overlap(a, b, la, lb);

        // 4β² S(j+2)
        if let Some(lb2) = shift(lb, axis, 2) {
            value += 4.0 * beta * beta * primitive_overlap(a, b, la, &lb2);
        }
    }

    -0.5 * value
}

fn primitive_nuclear_attraction(
    a: &PrimitiveGaussian,
    b: &PrimitiveGaussian,
    _angular_momentum_a: &(u8, u8, u8),
    _angular_momentum_b: &(u8, u8, u8),
) -> f64 {
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

/// Returns (p, μ, |R_A - R_B|²) for a primitive pair.
fn gaussian_pair_params(a: &PrimitiveGaussian, b: &PrimitiveGaussian) -> (f64, f64, f64) {
    let p = a.alpha() + b.alpha();
    let mu = (a.alpha() * b.alpha()) / p;
    let r2 = a.center().sub(&b.center()).norm_squared();
    (p, mu, r2)
}

fn weighted_center(a: &PrimitiveGaussian, b: &PrimitiveGaussian, p: f64) -> Point {
    Point {
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
