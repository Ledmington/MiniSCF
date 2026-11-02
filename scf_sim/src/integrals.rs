use std::f64::consts::PI;

use scf_core::{Atom, point::Point};

use crate::basis::{BasisFunction, PrimitiveGaussian};

pub fn overlap(a: &BasisFunction, b: &BasisFunction) -> f64 {
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

/// The (mu nu | 1/|r - point|) integral: the same math as a nuclear attraction integral,
/// but evaluated at an arbitrary point in space rather than at a real nucleus. Used to
/// build up the electronic contribution to the electrostatic potential on a grid.
///
/// Unlike [`nuclear_attraction`], this is the bare (positive) Coulomb integral: the
/// primitive nuclear attraction includes the minus sign of an attractive interaction,
/// which is undone here.
pub(crate) fn coulomb_potential(a: &BasisFunction, b: &BasisFunction, point: &Point) -> f64 {
    -contracted_pair_with_nucleus(a, b, point, primitive_nuclear_attraction)
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
/// Source: One- and Two-electron integrals over cartesian gaussian functions
/// DOI: https://doi.org/10.1016/0021-9991(78)90092-X
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

    let pc = center.sub(nucleus);

    let prefactor = -2.0 * PI / p * (-mu * r2).exp();

    // Hermite Coulomb integrals R_tuv(p, P - C): for t + u + v >= 2 (or P != C) these are
    // not just (-2p)^(t+u+v) F_{t+u+v}, they need the full recursion.
    let max_tx = e_x.len() - 1;
    let max_ty = e_y.len() - 1;
    let max_tz = e_z.len() - 1;
    let r = RCoulomb::new(
        max_tx + max_ty + max_tz,
        max_tx,
        max_ty,
        max_tz,
        p,
        p * pc.norm_squared(),
        (pc.x, pc.y, pc.z),
    );

    let mut sum = 0.0;

    for (tx, tx_elem) in e_x.iter().enumerate() {
        for (ty, ty_elem) in e_y.iter().enumerate() {
            for (tz, tz_elem) in e_z.iter().enumerate() {
                sum += tx_elem * ty_elem * tz_elem * r.get(0, tx, ty, tz);
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
                            // The ket Hermite coefficients pick up a (-1)^(tau+nu+phi) sign,
                            // since R_tuv is defined via derivatives w.r.t. the bra center P
                            // and the Coulomb kernel depends only on P - Q.
                            let ket_sign = if (tau + nu + phi) % 2 == 0 { 1.0 } else { -1.0 };
                            sum += e1 * e2 * ket_sign * r.get(0, t + tau, u + nu, v + phi);
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

/// Above this argument, `erf(sqrt(t))` equals 1 to double precision, so `F_0` has a closed
/// form and the upward recursion is numerically stable.
const BOYS_ASYMPTOTIC_THRESHOLD: f64 = 40.0;

/// The Boys function `F_n(t) = integral_0^1 x^(2n) exp(-t x^2) dx`.
///
/// For small and moderate `t`, `F_n` is computed from its power series
/// `F_n(t) = exp(-t) * sum_k (2t)^k / ((2n+1)(2n+3)...(2n+2k+1))`, which has only positive
/// terms and hence no cancellation. For large `t`, `F_0(t) = sqrt(pi/t)/2` and the upward
/// recursion `F_{m+1} = ((2m+1) F_m - exp(-t)) / (2t)` is used, which is stable there.
///
/// Higher angular momenta (d functions) need `n` up to 8 in the ERIs: a naive upward
/// recursion from `F_0` amplifies rounding errors by `~1/(2t)` at every step, which is
/// catastrophic for small `t`.
fn boys(n: usize, t: f64) -> f64 {
    if t >= BOYS_ASYMPTOTIC_THRESHOLD {
        let exp_t = (-t).exp();
        let mut f = 0.5 * (PI / t).sqrt();
        for m in 0..n {
            f = (((2 * m + 1) as f64) * f - exp_t) / (2.0 * t);
        }
        return f;
    }

    let mut term = 1.0 / (2 * n + 1) as f64;
    let mut sum = term;
    let mut k = 1;
    while term > sum * f64::EPSILON {
        term *= 2.0 * t / (2 * n + 2 * k + 1) as f64;
        sum += term;
        k += 1;
    }
    (-t).exp() * sum
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rand::{RngExt, SeedableRng, rngs::ChaCha8Rng};
    use rstest::rstest;

    use crate::basis::Shell;

    use super::*;

    /// Reference value of the Boys function by composite Simpson quadrature of its
    /// defining integral.
    fn boys_by_quadrature(n: usize, t: f64) -> f64 {
        const STEPS: usize = 20_000;
        let h = 1.0 / STEPS as f64;
        let f = |x: f64| x.powi(2 * n as i32) * (-t * x * x).exp();
        let mut sum = f(0.0) + f(1.0);
        for i in 1..STEPS {
            let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
            sum += weight * f(i as f64 * h);
        }
        sum * h / 3.0
    }

    #[rstest]
    fn test_boys_function_matches_quadrature(
        #[values(0, 1, 2, 4, 6, 8, 12)] n: usize,
        #[values(0.0, 1e-12, 1e-6, 1e-3, 0.1, 1.0, 5.0, 20.0, 39.9, 40.0, 60.0, 200.0)] t: f64,
    ) {
        let actual = boys(n, t);
        let expected = boys_by_quadrature(n, t);
        assert!(
            ((actual - expected) / expected).abs() < 1e-11,
            "F_{n}({t}): expected {expected} but was {actual}."
        );
    }

    #[test]
    fn test_boys_function_closed_forms() {
        for n in 0..10 {
            let expected = 1.0 / (2 * n + 1) as f64;
            assert_eq!(boys(n, 0.0), expected, "F_{n}(0)");
        }
        // F_0(t) = sqrt(pi/t) erf(sqrt(t)) / 2, with erf(1) to 16 digits.
        let expected = 0.5 * PI.sqrt() * 0.842_700_792_949_714_9;
        assert!((boys(0, 1.0) - expected).abs() < 1e-15);
    }

    #[rstest]
    #[case(0, 0)]
    #[case(0, 1)]
    #[case(1, 1)]
    #[case(0, 2)]
    #[case(1, 2)]
    #[case(2, 2)]
    #[case(0, 3)]
    #[case(2, 3)]
    #[case(3, 3)]
    fn test_symmetric_hermite_coefficients(#[case] ia: u8, #[case] ib: u8) {
        let seed = rand::rng().random();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let pa = rng.random_range(0.0..10.0);
        let pb = rng.random_range(0.0..10.0);
        let p = rng.random_range(0.0..10.0);

        let ab = hermite_coefficients(ia, ib, pa, pb, p);
        let ba = hermite_coefficients(ib, ia, pb, pa, p);
        assert_eq!(ab.len(), ba.len());
        // Relative tolerance: with f functions the coefficients reach ~1e7, and the two
        // recursion orders round differently in the last digits.
        assert!(
            (0..ab.len())
                .all(|i| (ab[i] - ba[i]).abs() <= 1e-12 * ab[i].abs().max(ba[i].abs()).max(1.0)),
            "Expected hermite coefficients for AB ({ab:?}) to be equal to those for BA ({ba:?}) (seed: {seed})."
        );
    }

    #[test]
    fn test_rcoulomb_parity() {
        let rho = 0.7;
        let t = 1.2;

        let x = 0.8;
        let y = -0.3;
        let z = 1.1;

        let r1 = RCoulomb::new(2, 2, 2, 2, rho, t, (x, y, z));
        let r2 = RCoulomb::new(2, 2, 2, 2, rho, t, (-x, -y, -z));

        let eps = 1e-12;

        for tx in 0..=2 {
            for ty in 0..=2 {
                for tz in 0..=2 {
                    let sign = if (tx + ty + tz) % 2 == 0 { 1.0 } else { -1.0 };

                    assert!(
                        (r1.get(0, tx, ty, tz) - sign * r2.get(0, tx, ty, tz)).abs() < eps,
                        "({tx}, {ty}, {tz}) failed"
                    );
                }
            }
        }
    }

    #[test]
    fn test_hermite_pair_swap_symmetry() {
        let a = PrimitiveGaussian::new(1.0, 0.5, Point::new(0.2, -0.3, 0.7));
        let b = PrimitiveGaussian::new(1.0, 1.1, Point::new(-0.4, 0.6, -0.2));

        let ab = hermite_pair(&a, &b, &(0, 0, 0), &(1, 0, 1));
        let ba = hermite_pair(&b, &a, &(1, 0, 1), &(0, 0, 0));

        assert_eq!(ab.e_x, ba.e_x);
        assert_eq!(ab.e_y, ba.e_y);
        assert_eq!(ab.e_z, ba.e_z);
        assert_eq!(ab.prefactor, ba.prefactor);
    }

    #[test]
    fn test_primitive_eri_pair_exchange_symmetry() {
        let a = PrimitiveGaussian::new(1.0, 0.5, Point::new(0.0, 0.0, 0.0));
        let b = PrimitiveGaussian::new(1.0, 0.8, Point::new(0.0, 0.0, 0.0));
        let c = PrimitiveGaussian::new(1.0, 1.2, Point::new(1.0, 0.0, 0.0));
        let d = PrimitiveGaussian::new(1.0, 0.7, Point::new(1.0, 0.0, 0.0));

        let ab = primitive_eri(
            (&a, &(0, 0, 0)),
            (&b, &(0, 0, 0)),
            (&c, &(1, 0, 0)),
            (&d, &(1, 1, 0)),
        );

        let cd = primitive_eri(
            (&c, &(1, 0, 0)),
            (&d, &(1, 1, 0)),
            (&a, &(0, 0, 0)),
            (&b, &(0, 0, 0)),
        );

        assert!(
            (ab - cd).abs() < 1e-12,
            "primitive ERI symmetry broken: {ab} vs {cd}"
        );
    }

    #[test]
    fn test_contracted_eri_pair_exchange_symmetry() {
        let center1 = Point::new(0.0, 0.0, 0.0);
        let center2 = Point::new(1.0, 0.0, 0.0);

        let shell1 = Shell {
            center: center1,
            primitives: vec![
                PrimitiveGaussian::new(0.1543289673, 3.425250914, center1),
                PrimitiveGaussian::new(0.5353281423, 0.6239137298, center1),
                PrimitiveGaussian::new(0.4446345422, 0.168855404, center1),
            ],
        };

        let shell2 = Shell {
            center: center2,
            primitives: vec![
                PrimitiveGaussian::new(0.1559162750, 2.941249355, center2),
                PrimitiveGaussian::new(0.6076837186, 0.6834830964, center2),
                PrimitiveGaussian::new(0.3919573931, 0.2222899159, center2),
            ],
        };

        let s = BasisFunction {
            shell: Arc::new(shell1),
            angular_momentum: (0, 0, 0),
        };

        let d = BasisFunction {
            shell: Arc::new(shell2),
            angular_momentum: (2, 0, 0),
        };

        let sd = electron_repulsion(&s, &s, &d, &d);
        let ds = electron_repulsion(&d, &d, &s, &s);

        assert!((sd - ds).abs() < 1e-12, "(ss|dd) != (dd|ss): {sd} {ds}");
    }

    #[test]
    fn test_all_pair_exchange_cartesian() {
        let functions = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (2, 0, 0),
            (1, 1, 0),
            (0, 0, 2),
        ];

        let a = PrimitiveGaussian::new(1.0, 1.0, Point::new(0.0, 0.0, 0.0));
        let b = PrimitiveGaussian::new(1.0, 1.0, Point::new(0.0, 0.0, 0.0));
        let c = PrimitiveGaussian::new(1.0, 1.0, Point::new(0.0, 0.0, 0.0));
        let d = PrimitiveGaussian::new(1.0, 1.0, Point::new(0.0, 0.0, 0.0));

        for &la in &functions {
            for &lb in &functions {
                for &lc in &functions {
                    for &ld in &functions {
                        let x = primitive_eri((&a, &la), (&b, &lb), (&c, &lc), (&d, &ld));
                        let y = primitive_eri((&c, &lc), (&d, &ld), (&a, &la), (&b, &lb));

                        assert!(
                            (x - y).abs() < 1e-10,
                            "failed ({la:?}{lb:?}|{lc:?}{ld:?}) {x} {y}"
                        );
                    }
                }
            }
        }
    }

    /// The angular momenta raised by one along each axis by the raising checks: the p ones
    /// give the d integrals, the d ones the f integrals.
    const LOWER_COMPONENTS: [(u8, u8, u8); 9] = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (2, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 2, 0),
        (0, 1, 1),
        (0, 0, 2),
    ];

    /// A representative set of (other) angular momenta, up to f.
    const OTHER_COMPONENTS: [(u8, u8, u8); 7] = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 1),
        (0, 2, 0),
        (1, 1, 0),
        (1, 1, 1),
        (0, 0, 3),
    ];

    fn moved(p: &Point, axis: usize, delta: f64) -> Point {
        let mut c = p.coordinates();
        c[axis] += delta;
        Point::new(c[0], c[1], c[2])
    }

    /// Checks that the d- and f-type integrals computed by `g` agree with the ones
    /// obtained from lower angular momenta via the identity, for an unnormalized
    /// Cartesian Gaussian `G_l` with exponent `alpha` centered at `A`,
    ///
    ///   d/dA_i G_l = 2 alpha G_{l + e_i} - l_i G_{l - e_i}
    ///
    /// where the center derivative is taken by central finite differences. Since `g` is
    /// linear in its first function, the same identity holds for the integrals. This
    /// validates the d-type integrals independently of the McMurchie-Davidson recursion,
    /// using only the (already validated) s- and p-type ones, and in turn the f-type ones
    /// from the d-type ones.
    fn check_angular_momentum_raising(
        name: &str,
        g: impl Fn(&PrimitiveGaussian, &(u8, u8, u8)) -> f64,
    ) {
        const H: f64 = 1e-4;
        let alpha = 0.8;
        let center = Point::new(0.3, -0.2, 0.5);
        let at = |c: Point| PrimitiveGaussian::new(1.0, alpha, c);

        for lower in LOWER_COMPONENTS {
            for axis in 0..3 {
                let target = shift(&lower, axis, 1).unwrap();

                let derivative = (g(&at(moved(&center, axis, H)), &lower)
                    - g(&at(moved(&center, axis, -H)), &lower))
                    / (2.0 * H);
                let l_axis = lower_component(&lower, axis);
                let below = match shift(&lower, axis, -1) {
                    Some(l) => l_axis * g(&at(center), &l),
                    None => 0.0,
                };
                let expected = (derivative + below) / (2.0 * alpha);

                let actual = g(&at(center), &target);
                assert!(
                    (actual - expected).abs() < 1e-7 * expected.abs().max(1.0),
                    "{name} for {target:?}: expected {expected} but was {actual}."
                );
            }
        }
    }

    fn lower_component(l: &(u8, u8, u8), axis: usize) -> f64 {
        [l.0, l.1, l.2][axis] as f64
    }

    #[test]
    fn test_overlap_by_raising() {
        let b = PrimitiveGaussian::new(1.0, 1.3, Point::new(-0.4, 0.6, -0.1));
        for lb in OTHER_COMPONENTS {
            check_angular_momentum_raising(&format!("overlap with {lb:?}"), |a, la| {
                primitive_overlap(a, &b, la, &lb)
            });
        }
    }

    #[test]
    fn test_kinetic_energy_by_raising() {
        let b = PrimitiveGaussian::new(1.0, 1.3, Point::new(-0.4, 0.6, -0.1));
        for lb in OTHER_COMPONENTS {
            check_angular_momentum_raising(&format!("kinetic energy with {lb:?}"), |a, la| {
                primitive_kinetic_energy(a, &b, la, &lb)
            });
        }
    }

    #[test]
    fn test_nuclear_attraction_by_raising() {
        let b = PrimitiveGaussian::new(1.0, 1.3, Point::new(-0.4, 0.6, -0.1));
        let nucleus = Point::new(0.7, 0.2, -0.6);
        for lb in OTHER_COMPONENTS {
            check_angular_momentum_raising(&format!("nuclear attraction with {lb:?}"), |a, la| {
                primitive_nuclear_attraction(a, &b, &nucleus, la, &lb)
            });
        }
    }

    #[test]
    fn test_electron_repulsion_by_raising() {
        let b = PrimitiveGaussian::new(1.0, 1.3, Point::new(-0.4, 0.6, -0.1));
        let c = PrimitiveGaussian::new(1.0, 0.6, Point::new(0.9, -0.5, 0.2));
        let d = PrimitiveGaussian::new(1.0, 2.1, Point::new(0.1, 0.8, -0.7));
        for lb in OTHER_COMPONENTS {
            for (lc, ld) in [
                ((0, 0, 0), (0, 0, 0)),
                ((2, 0, 0), (0, 1, 1)),
                ((0, 0, 1), (1, 0, 1)),
                ((0, 3, 0), (1, 1, 1)),
            ] {
                check_angular_momentum_raising(
                    &format!("ERI with {lb:?}, {lc:?}, {ld:?}"),
                    |a, la| primitive_eri((a, la), (&b, &lb), (&c, &lc), (&d, &ld)),
                );
            }
        }
    }
}
