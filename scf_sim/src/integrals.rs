use std::f64::consts::PI;

use scf_core::{Atom, point::Point};

use crate::basis::{BasisFunction, PrimitiveGaussian};

pub(crate) fn overlap(a: &BasisFunction, b: &BasisFunction) -> f64 {
    contracted_pair(a, b, primitive_overlap)
}

pub(crate) fn kinetic_energy(a: &BasisFunction, b: &BasisFunction) -> f64 {
    contracted_pair(a, b, primitive_kinetic_energy)
}

pub(crate) fn nuclear_attraction(a: &BasisFunction, b: &BasisFunction, nuclei: &[Atom]) -> f64 {
    let mut sum = 0.0;
    for nucleus in nuclei {
        let z = nucleus.charge as f64;
        sum +=
            z * contracted_pair_with_nucleus(a, b, &nucleus.position, primitive_nuclear_attraction);
    }
    sum
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
                    sum += a.normalized_coefficient(prim_a)
                        * b.normalized_coefficient(prim_b)
                        * c.normalized_coefficient(prim_c)
                        * d.normalized_coefficient(prim_d)
                        * primitive_eri(
                            (prim_a, &a.angular_momentum),
                            (prim_b, &b.angular_momentum),
                            (prim_c, &c.angular_momentum),
                            (prim_d, &d.angular_momentum),
                        );
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
            sum += a.normalized_coefficient(pa)
                * b.normalized_coefficient(pb)
                * f(pa, pb, &a.angular_momentum, &b.angular_momentum);
        }
    }
    sum
}

/// Contracted helper
fn contracted_pair_with_nucleus(
    a: &BasisFunction,
    b: &BasisFunction,
    nucleus: &Point,
    f: impl Fn(&PrimitiveGaussian, &PrimitiveGaussian, &Point, &(u8, u8, u8), &(u8, u8, u8)) -> f64,
) -> f64 {
    let mut sum = 0.0;
    for pa in &a.shell.primitives {
        for pb in &b.shell.primitives {
            sum += a.normalized_coefficient(pa)
                * b.normalized_coefficient(pb)
                * f(pa, pb, nucleus, &a.angular_momentum, &b.angular_momentum);
        }
    }
    sum
}

struct RCoulomb {
    data: Vec<Vec<Vec<Vec<f64>>>>,
}

impl RCoulomb {
    fn new(
        max_n: usize,
        max_t: usize,
        max_u: usize,
        max_v: usize,
        rho: f64,
        t: f64,
        p: (f64, f64, f64),
    ) -> Self {
        let mut data = vec![vec![vec![vec![0.0; max_v + 1]; max_u + 1]; max_t + 1]; max_n + 2];

        //
        // Base:
        //
        // R^n_000 = (-2rho)^n F_n(T)
        //
        for (n, x) in data.iter_mut().enumerate().take(max_n + 1 + 1) {
            x[0][0][0] = (-2.0 * rho).powi(n as i32) * boys(n, t);
        }

        //
        // Build R^n_{tuv}
        //
        for n in (0..=max_n).rev() {
            for tx in 0..=max_t {
                for ty in 0..=max_u {
                    for tz in 0..=max_v {
                        if tx == 0 && ty == 0 && tz == 0 {
                            continue;
                        }

                        let value = if tx > 0 {
                            //
                            // R^n_{tuv}
                            // from x recursion
                            //
                            let mut v = p.0 * data[n + 1][tx - 1][ty][tz];

                            if tx > 1 {
                                v += (tx - 1) as f64 * data[n + 1][tx - 2][ty][tz];
                            }

                            v
                        } else if ty > 0 {
                            let mut v = p.1 * data[n + 1][tx][ty - 1][tz];

                            if ty > 1 {
                                v += (ty - 1) as f64 * data[n + 1][tx][ty - 2][tz];
                            }

                            v
                        } else {
                            let mut v = p.2 * data[n + 1][tx][ty][tz - 1];

                            if tz > 1 {
                                v += (tz - 1) as f64 * data[n + 1][tx][ty][tz - 2];
                            }

                            v
                        };

                        data[n][tx][ty][tz] = value;
                    }
                }
            }
        }

        Self { data }
    }

    fn get(&self, n: usize, t: usize, u: usize, v: usize) -> f64 {
        self.data[n][t][u][v]
    }
}

struct HermitePair {
    e_x: Vec<f64>,
    e_y: Vec<f64>,
    e_z: Vec<f64>,
    prefactor: f64,
}

fn hermite_pair(
    a: &PrimitiveGaussian,
    b: &PrimitiveGaussian,
    la: &(u8, u8, u8),
    lb: &(u8, u8, u8),
) -> HermitePair {
    let (p, mu, r2) = gaussian_pair_params(a, b);

    let center = weighted_center(a, b, p);

    let pa = center.sub(&a.center()).coordinates();
    let pb = center.sub(&b.center()).coordinates();

    HermitePair {
        e_x: hermite_coefficients(la.0, lb.0, pa[0], pb[0], p),
        e_y: hermite_coefficients(la.1, lb.1, pa[1], pb[1], p),
        e_z: hermite_coefficients(la.2, lb.2, pa[2], pb[2], p),
        prefactor: (-mu * r2).exp(),
    }
}

fn hermite_coefficient(ia: u8, ib: u8, t: u8, pa: f64, pb: f64, p: f64) -> f64 {
    fn recurse(i: i32, j: i32, t: i32, pa: f64, pb: f64, p: f64) -> f64 {
        if t < 0 || t > i + j {
            return 0.0;
        }

        if i == 0 && j == 0 {
            return if t == 0 { 1.0 } else { 0.0 };
        }

        if i > 0 {
            return pa * recurse(i - 1, j, t, pa, pb, p)
                + recurse(i - 1, j, t - 1, pa, pb, p) / (2.0 * p)
                + (t + 1) as f64 * recurse(i - 1, j, t + 1, pa, pb, p);
        }

        pb * recurse(i, j - 1, t, pa, pb, p)
            + recurse(i, j - 1, t - 1, pa, pb, p) / (2.0 * p)
            + (t + 1) as f64 * recurse(i, j - 1, t + 1, pa, pb, p)
    }

    recurse(ia as i32, ib as i32, t as i32, pa, pb, p)
}

fn hermite_coefficients(ia: u8, ib: u8, pa: f64, pb: f64, p: f64) -> Vec<f64> {
    (0..=ia + ib)
        .map(|t| hermite_coefficient(ia, ib, t, pa, pb, p))
        .collect()
}

/// McMurchie-Davidson recurrence
fn overlap_1d(ia: u8, ib: u8, pa: f64, pb: f64, p: f64) -> f64 {
    (PI / p).sqrt() * hermite_coefficient(ia, ib, 0, pa, pb, p)
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

pub(crate) fn primitive_kinetic_energy(
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
            2 => lb.2,
            _ => unreachable!(),
        } as f64;

        // j(j-1) S(j-2)
        if l >= 2.0
            && let Some(lb2) = shift(lb, axis, -2)
        {
            value += l * (l - 1.0) * primitive_overlap(a, b, la, &lb2);
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

pub(crate) fn primitive_nuclear_attraction(
    a: &PrimitiveGaussian,
    b: &PrimitiveGaussian,
    nucleus: &Point,
    la: &(u8, u8, u8),
    lb: &(u8, u8, u8),
) -> f64 {
    let (p, mu, r2) = gaussian_pair_params(a, b);

    let center = weighted_center(a, b, p);

    let pa = center.sub(&a.center()).coordinates();
    let pb = center.sub(&b.center()).coordinates();

    let e_x = hermite_coefficients(la.0, lb.0, pa[0], pb[0], p);
    let e_y = hermite_coefficients(la.1, lb.1, pa[1], pb[1], p);
    let e_z = hermite_coefficients(la.2, lb.2, pa[2], pb[2], p);

    let rpc2 = center.sub(nucleus).norm_squared();

    let t = p * rpc2;

    let prefactor = -2.0 * PI / p * (-mu * r2).exp();

    let mut sum = 0.0;

    for (tx, tx_elem) in e_x.iter().enumerate() {
        for (ty, ty_elem) in e_y.iter().enumerate() {
            for (tz, tz_elem) in e_z.iter().enumerate() {
                sum += tx_elem
                    * ty_elem
                    * tz_elem
                    * boys(tx + ty + tz, t)
                    * (-2.0 * p).powi((tx + ty + tz) as i32);
            }
        }
    }

    prefactor * sum
}

fn primitive_eri(
    a: (&PrimitiveGaussian, &(u8, u8, u8)),
    b: (&PrimitiveGaussian, &(u8, u8, u8)),
    c: (&PrimitiveGaussian, &(u8, u8, u8)),
    d: (&PrimitiveGaussian, &(u8, u8, u8)),
) -> f64 {
    let (prim_a, la) = a;
    let (prim_b, lb) = b;
    let (prim_c, lc) = c;
    let (prim_d, ld) = d;

    let p = prim_a.alpha() + prim_b.alpha();
    let q = prim_c.alpha() + prim_d.alpha();

    let ab = hermite_pair(prim_a, prim_b, la, lb);
    let cd = hermite_pair(prim_c, prim_d, lc, ld);

    let p_center = weighted_center(prim_a, prim_b, p);
    let q_center = weighted_center(prim_c, prim_d, q);
    let pq = p_center.sub(&q_center);
    let pq2 = pq.norm_squared();

    let rho = p * q / (p + q);

    let boys_argument = rho * pq2;

    // Maximum auxiliary angular momentum
    let max_n = (la.0 + lb.0 + la.1 + lb.1 + la.2 + lb.2 + lc.0 + ld.0 + lc.1 + ld.1 + lc.2 + ld.2)
        as usize;

    let max_tx = ab.e_x.len() + cd.e_x.len() - 2;
    let max_ty = ab.e_y.len() + cd.e_y.len() - 2;
    let max_tz = ab.e_z.len() + cd.e_z.len() - 2;

    let r = RCoulomb::new(
        max_n,
        max_tx,
        max_ty,
        max_tz,
        rho,
        boys_argument,
        (pq.x, pq.y, pq.z),
    );

    let mut sum = 0.0;

    for t in 0..ab.e_x.len() {
        for u in 0..ab.e_y.len() {
            for v in 0..ab.e_z.len() {
                let e1 = ab.e_x[t] * ab.e_y[u] * ab.e_z[v];

                for tau in 0..cd.e_x.len() {
                    for nu in 0..cd.e_y.len() {
                        for phi in 0..cd.e_z.len() {
                            let e2 = cd.e_x[tau] * cd.e_y[nu] * cd.e_z[phi];
                            sum += e1 * e2 * r.get(0, t + tau, u + nu, v + phi);
                        }
                    }
                }
            }
        }
    }

    // Overall ERI prefactor
    let prefactor = 2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt());

    prefactor * ab.prefactor * cd.prefactor * sum
}

/// Returns (p, μ, |R_A - R_B|²) for a primitive pair.
fn gaussian_pair_params(a: &PrimitiveGaussian, b: &PrimitiveGaussian) -> (f64, f64, f64) {
    let p = a.alpha() + b.alpha();
    let mu = (a.alpha() * b.alpha()) / p;
    let r2 = a.center().sub(&b.center()).norm_squared();
    (p, mu, r2)
}

fn weighted_center(a: &PrimitiveGaussian, b: &PrimitiveGaussian, p: f64) -> Point {
    Point::new(
        (a.alpha() * a.center().x + b.alpha() * b.center().x) / p,
        (a.alpha() * a.center().y + b.alpha() * b.center().y) / p,
        (a.alpha() * a.center().z + b.alpha() * b.center().z) / p,
    )
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
    let t = 1.0 / (1.0 + p * x.abs());
    let y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * (-x * x).exp();
    sign * y
}

fn boys_0(t: f64) -> f64 {
    if t < 1.0e-8 {
        return 1.0;
    }

    0.5 * (PI / t).sqrt() * erf(t.sqrt())
}

fn boys(n: usize, t: f64) -> f64 {
    if t < 1.0e-8 {
        // Small-T expansion
        return 1.0 / (2 * n + 1) as f64;
    }

    let mut f = boys_0(t);

    for m in 0..n {
        f = (((2 * m + 1) as f64) * f - (-t).exp()) / (2.0 * t);
    }

    f
}

#[cfg(test)]
mod tests {
    use rand::{RngExt, SeedableRng, rngs::ChaCha8Rng};
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(0, 0)]
    #[case(0, 1)]
    #[case(1, 1)]
    #[case(0, 2)]
    fn test_symmetric_hermite_coefficients(#[case] ia: u8, #[case] ib: u8) {
        let seed = rand::rng().random();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let pa = rng.random_range(0.0..10.0);
        let pb = rng.random_range(0.0..10.0);
        let p = rng.random_range(0.0..10.0);

        let ab = hermite_coefficients(ia, ib, pa, pb, p);
        let ba = hermite_coefficients(ib, ia, pb, pa, p);
        assert_eq!(
            ab, ba,
            "Expected hermite coefficients for AB ({ab:?}) to be equal to those for BA ({ba:?}) (seed: {seed})."
        );
    }
}
