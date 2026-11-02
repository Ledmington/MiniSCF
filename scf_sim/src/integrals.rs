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
#[cfg(test)]
pub(crate) fn coulomb_potential(a: &BasisFunction, b: &BasisFunction, point: &Point) -> f64 {
    -contracted_pair_with_nucleus(a, b, point, primitive_nuclear_attraction)
}

#[cfg(test)]
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

#[cfg(test)]
struct HermitePair {
    e_x: Vec<f64>,
    e_y: Vec<f64>,
    e_z: Vec<f64>,
    prefactor: f64,
}

#[cfg(test)]
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

#[cfg(test)]
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

// ------ electron repulsion tensor --------------------------------------

/// One primitive pair of a `ShellPair`.
struct PrimitivePair {
    exponent: f64,
    center: Point,
    /// For each function pair of the shell pair, its Hermite expansion coefficients
    /// (normalization, contraction coefficients and `exp(-mu |A - B|^2)` included) in a
    /// dense cube of side `l + 1`, indexed by `(t * dim + u) * dim + v`.
    expansions: Vec<f64>,
}

/// The products of the functions of two shells (`a` and `b`, ranges of function indices).
struct ShellPair {
    a: std::ops::Range<usize>,
    b: std::ops::Range<usize>,
    /// Total angular momentum `la + lb`.
    l: usize,
    primitives: Vec<PrimitivePair>,
}

impl ShellPair {
    fn new(
        functions: &[BasisFunction],
        a: std::ops::Range<usize>,
        b: std::ops::Range<usize>,
    ) -> Self {
        let total_l = |i: usize| {
            let (x, y, z) = functions[i].angular_momentum;
            (x + y + z) as usize
        };
        let l = total_l(a.start) + total_l(b.start);
        let dim = l + 1;
        let shell_a = &functions[a.start].shell;
        let shell_b = &functions[b.start].shell;
        let mut primitives = Vec::new();
        for prim_a in &shell_a.primitives {
            for prim_b in &shell_b.primitives {
                let (p, mu, r2) = gaussian_pair_params(prim_a, prim_b);
                let overlap_prefactor = (-mu * r2).exp();
                if overlap_prefactor < PAIR_SCREENING_THRESHOLD {
                    continue;
                }
                let center = weighted_center(prim_a, prim_b, p);
                let pa = center.sub(&prim_a.center()).coordinates();
                let pb = center.sub(&prim_b.center()).coordinates();
                let mut expansions = vec![0.0; a.len() * b.len() * dim * dim * dim];
                for (pair, (i, j)) in a
                    .clone()
                    .flat_map(|i| b.clone().map(move |j| (i, j)))
                    .enumerate()
                {
                    let (fa, fb) = (&functions[i], &functions[j]);
                    let weight = overlap_prefactor
                        * fa.normalized_coefficient(prim_a)
                        * fb.normalized_coefficient(prim_b);
                    let (la, lb) = (fa.angular_momentum, fb.angular_momentum);
                    let e_x = hermite_coefficients(la.0, lb.0, pa[0], pb[0], p);
                    let e_y = hermite_coefficients(la.1, lb.1, pa[1], pb[1], p);
                    let e_z = hermite_coefficients(la.2, lb.2, pa[2], pb[2], p);
                    let cube = &mut expansions[pair * dim * dim * dim..];
                    for (t, ex) in e_x.iter().enumerate() {
                        for (u, ey) in e_y.iter().enumerate() {
                            for (v, ez) in e_z.iter().enumerate() {
                                cube[(t * dim + u) * dim + v] = weight * ex * ey * ez;
                            }
                        }
                    }
                }
                primitives.push(PrimitivePair {
                    exponent: p,
                    center,
                    expansions,
                });
            }
        }
        Self {
            a,
            b,
            l,
            primitives,
        }
    }

    fn num_function_pairs(&self) -> usize {
        self.a.len() * self.b.len()
    }

    fn function_pairs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.a
            .clone()
            .flat_map(|i| self.b.clone().map(move |j| (i, j)))
    }
}

/// The electron repulsion integrals `(ij|kl)` of all the basis functions, computed shell
/// quartet by shell quartet with the McMurchie-Davidson scheme: the Hermite expansions of
/// every primitive pair are computed once, and each primitive quartet needs a single
/// evaluation of the Boys functions and of the Hermite Coulomb integrals.
pub(crate) fn electron_repulsion_tensor(functions: &[BasisFunction]) -> ndarray::Array4<f64> {
    let n = functions.len();
    let groups = crate::basis::shell_groups(functions);
    let mut pairs = Vec::new();
    for (ga, a) in groups.iter().enumerate() {
        for b in &groups[..=ga] {
            pairs.push(ShellPair::new(functions, a.clone(), b.clone()));
        }
    }
    let max_l = pairs.iter().map(|p| p.l).max().unwrap_or(0);
    let boys_table = BoysTable::new(2 * max_l);
    let mut boys = vec![0.0; 2 * max_l + 1];
    let mut r = vec![0.0; (2 * max_l + 1).pow(4)];
    let mut ket_contracted = Vec::new();
    let mut values = Vec::new();

    let mut eri = ndarray::Array4::zeros((n, n, n, n));
    for (index, bra) in pairs.iter().enumerate() {
        for ket in &pairs[..=index] {
            let total_l = bra.l + ket.l;
            let dim = total_l + 1;
            let (bra_dim, ket_dim) = (bra.l + 1, ket.l + 1);
            let (bra_cube, ket_cube) = (bra_dim.pow(3), ket_dim.pow(3));
            values.clear();
            values.resize(bra.num_function_pairs() * ket.num_function_pairs(), 0.0);
            ket_contracted.resize(ket.num_function_pairs() * bra_cube, 0.0);

            for pb in &bra.primitives {
                for pk in &ket.primitives {
                    let (p, q) = (pb.exponent, pk.exponent);
                    let rho = p * q / (p + q);
                    let pq = pb.center.sub(&pk.center);
                    hermite_coulomb_scalar(
                        total_l,
                        rho,
                        [pq.x, pq.y, pq.z],
                        &boys_table,
                        &mut boys,
                        &mut r,
                    );
                    let prefactor = 2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt());

                    // X_kl(t,u,v) = sum_(tau,nu,phi) (-1)^(tau+nu+phi) E_kl(tau,nu,phi)
                    //               * R(t+tau, u+nu, v+phi)
                    // (the ket picks up the sign since R is a derivative w.r.t. P).
                    for (kl, x) in ket_contracted.chunks_exact_mut(bra_cube).enumerate() {
                        let e_kl = &pk.expansions[kl * ket_cube..][..ket_cube];
                        x.fill(0.0);
                        for tau in 0..=ket.l {
                            for nu in 0..=(ket.l - tau) {
                                for phi in 0..=(ket.l - tau - nu) {
                                    let e = e_kl[(tau * ket_dim + nu) * ket_dim + phi];
                                    if e == 0.0 {
                                        continue;
                                    }
                                    let signed = if (tau + nu + phi) % 2 == 0 { e } else { -e };
                                    for t in 0..=bra.l {
                                        for u in 0..=(bra.l - t) {
                                            let row = &r[((t + tau) * dim + u + nu) * dim + phi..];
                                            let out = &mut x[(t * bra_dim + u) * bra_dim..];
                                            for v in 0..=(bra.l - t - u) {
                                                out[v] += signed * row[v];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for (ij, e_ij) in pb.expansions.chunks_exact(bra_cube).enumerate() {
                        let row = &mut values[ij * ket.num_function_pairs()..];
                        for (kl, x) in ket_contracted.chunks_exact(bra_cube).enumerate() {
                            let dot: f64 = e_ij.iter().zip(x).map(|(e, xv)| e * xv).sum();
                            row[kl] += prefactor * dot;
                        }
                    }
                }
            }

            let mut value = values.iter();
            for (i, j) in bra.function_pairs() {
                for (k, l) in ket.function_pairs() {
                    let v = *value.next().expect("one value per function quartet");
                    for [w, x, y, z] in [
                        [i, j, k, l],
                        [j, i, k, l],
                        [i, j, l, k],
                        [j, i, l, k],
                        [k, l, i, j],
                        [l, k, i, j],
                        [k, l, j, i],
                        [l, k, j, i],
                    ] {
                        eri[[w, x, y, z]] = v;
                    }
                }
            }
        }
    }
    eri
}

/// Fills `r` with the Hermite Coulomb integrals `R^n_tuv(alpha, pc)` for every
/// `n + t + u + v <= l` at a single point, stored like in `hermite_coulomb_impl`.
fn hermite_coulomb_scalar(
    l: usize,
    alpha: f64,
    pc: [f64; 3],
    boys_table: &BoysTable,
    boys: &mut [f64],
    r: &mut [f64],
) {
    let t = alpha * (pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2]);
    boys_table.fill(l, t, boys);
    let dim = l + 1;
    let block = dim * dim * dim;
    let mut factor = 1.0;
    for (n, f) in boys.iter().enumerate().take(l + 1) {
        r[n * block] = factor * f;
        factor *= -2.0 * alpha;
    }
    for n in (0..l).rev() {
        let (lower, upper) = r.split_at_mut((n + 1) * block);
        let here = &mut lower[n * block..];
        let max = l - n;
        for tx in 0..=max {
            for ty in 0..=(max - tx) {
                for tz in 0..=(max - tx - ty) {
                    here[(tx * dim + ty) * dim + tz] = if tx > 0 {
                        let index = ((tx - 1) * dim + ty) * dim + tz;
                        let second = if tx > 1 {
                            (tx - 1) as f64 * upper[index - dim * dim]
                        } else {
                            0.0
                        };
                        pc[0] * upper[index] + second
                    } else if ty > 0 {
                        let index = (ty - 1) * dim + tz;
                        let second = if ty > 1 {
                            (ty - 1) as f64 * upper[index - dim]
                        } else {
                            0.0
                        };
                        pc[1] * upper[index] + second
                    } else if tz > 0 {
                        let second = if tz > 1 {
                            (tz - 1) as f64 * upper[tz - 2]
                        } else {
                            0.0
                        };
                        pc[2] * upper[tz - 1] + second
                    } else {
                        continue;
                    };
                }
            }
        }
    }
}

// ------ electrostatic potential of a density ----------------------------

/// Primitive pairs whose Gaussian overlap prefactor `exp(-mu |A - B|^2)` is below this
/// value contribute nothing measurable to the potential and are skipped.
const PAIR_SCREENING_THRESHOLD: f64 = 1e-15;

/// The product of two primitive shells, contracted with the density matrix and expanded in
/// Hermite Gaussians centered at `center`: its Coulomb potential at a point C is
/// `sum_(t,u,v) coefficient * R_tuv(exponent, center - C)`.
struct HermiteDistribution {
    exponent: f64,
    center: Point,
    /// Highest `t + u + v` of the expansion.
    max_l: usize,
    /// `(index in the R table of side max_l + 1, coefficient)` of the nonzero terms.
    terms: Vec<(usize, f64)>,
}

/// Evaluates `sum_(mu,nu) P[mu,nu] (mu nu | 1/|r - C|)` at arbitrary points C, with all the
/// point-independent work (Hermite expansions, normalization, density weights) done once.
pub(crate) struct CoulombPotentialEvaluator {
    /// The spherical (`max_l == 0`) distributions as `(exponent, center, coefficient,
    /// charge)`, evaluated by a dedicated fast path; far away they are point charges.
    spherical: Vec<(f64, Point, f64, f64)>,
    distributions: Vec<HermiteDistribution>,
    boys_table: BoysTable,
    boys: Vec<f64>,
    /// The Hermite Coulomb integrals of one distribution at a batch of points.
    r: Vec<Lanes>,
}

/// How many points `CoulombPotentialEvaluator::evaluate_batch` evaluates together.
pub(crate) const POTENTIAL_BATCH: usize = 16;

/// One value per point of a batch.
type Lanes = [f64; POTENTIAL_BATCH];

impl CoulombPotentialEvaluator {
    pub(crate) fn new(functions: &[BasisFunction], density: &ndarray::Array2<f64>) -> Self {
        // Functions sharing a shell are expanded together.
        let groups = crate::basis::shell_groups(functions);

        let total_l = |f: &BasisFunction| {
            let (x, y, z) = f.angular_momentum;
            (x + y + z) as usize
        };

        // Primitive pairs with the same exponent and center (e.g. (ab) and (ba) within a
        // shell, or the S and P shells of an SP block) are merged into one distribution.
        // Coefficients are accumulated in cubes of side `cube` and repacked at the end.
        let cube = 2 * functions.iter().map(total_l).max().unwrap_or(0) + 1;
        let mut merged: Vec<(f64, Point, usize, Vec<f64>)> = Vec::new();
        let mut index_of = std::collections::HashMap::new();
        for (ga, group_a) in groups.iter().enumerate() {
            for (gb, group_b) in groups.iter().enumerate().skip(ga) {
                let shell_a = &functions[group_a.start].shell;
                let shell_b = &functions[group_b.start].shell;
                let max_l = group_a
                    .clone()
                    .map(|i| total_l(&functions[i]))
                    .max()
                    .unwrap()
                    + group_b
                        .clone()
                        .map(|j| total_l(&functions[j]))
                        .max()
                        .unwrap();
                // (ij) and (ji) both appear in the sum over the density matrix.
                let symmetry = if ga == gb { 1.0 } else { 2.0 };

                for prim_a in &shell_a.primitives {
                    for prim_b in &shell_b.primitives {
                        let (p, mu, r2) = gaussian_pair_params(prim_a, prim_b);
                        let overlap_prefactor = (-mu * r2).exp();
                        if overlap_prefactor < PAIR_SCREENING_THRESHOLD {
                            continue;
                        }
                        let center = weighted_center(prim_a, prim_b, p);
                        let pa = center.sub(&prim_a.center()).coordinates();
                        let pb = center.sub(&prim_b.center()).coordinates();

                        let key = [p, center.x, center.y, center.z].map(f64::to_bits);
                        let slot = *index_of.entry(key).or_insert_with(|| {
                            merged.push((p, center, 0, vec![0.0; cube * cube * cube]));
                            merged.len() - 1
                        });
                        let scale = 2.0 * PI / p * overlap_prefactor;
                        let (_, _, merged_l, coefficients) = &mut merged[slot];
                        *merged_l = (*merged_l).max(max_l);
                        for i in group_a.clone() {
                            for j in group_b.clone() {
                                let (fa, fb) = (&functions[i], &functions[j]);
                                let weight = symmetry
                                    * density[[i, j]]
                                    * fa.normalized_coefficient(prim_a)
                                    * fb.normalized_coefficient(prim_b);
                                if weight == 0.0 {
                                    continue;
                                }
                                let (la, lb) = (fa.angular_momentum, fb.angular_momentum);
                                let e_x = hermite_coefficients(la.0, lb.0, pa[0], pb[0], p);
                                let e_y = hermite_coefficients(la.1, lb.1, pa[1], pb[1], p);
                                let e_z = hermite_coefficients(la.2, lb.2, pa[2], pb[2], p);
                                for (t, ex) in e_x.iter().enumerate() {
                                    for (u, ey) in e_y.iter().enumerate() {
                                        for (v, ez) in e_z.iter().enumerate() {
                                            coefficients[(t * cube + u) * cube + v] +=
                                                scale * weight * ex * ey * ez;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let distributions: Vec<HermiteDistribution> = merged
            .into_iter()
            .filter_map(|(exponent, center, pair_l, coefficients)| {
                // Only the nonzero terms matter: cancellations can lower the order.
                let nonzero =
                    |t: usize, u: usize, v: usize| coefficients[(t * cube + u) * cube + v] != 0.0;
                let max_l = (0..=pair_l)
                    .flat_map(|t| {
                        (0..=pair_l).flat_map(move |u| (0..=pair_l).map(move |v| (t, u, v)))
                    })
                    .filter(|&(t, u, v)| t + u + v <= pair_l && nonzero(t, u, v))
                    .map(|(t, u, v)| t + u + v)
                    .max()?;
                let dim = max_l + 1;
                let mut terms = Vec::new();
                for t in 0..dim {
                    for u in 0..dim {
                        for v in 0..dim {
                            let c = coefficients[(t * cube + u) * cube + v];
                            if c != 0.0 {
                                terms.push(((t * dim + u) * dim + v, c));
                            }
                        }
                    }
                }
                Some(HermiteDistribution {
                    exponent,
                    center,
                    max_l,
                    terms,
                })
            })
            .collect();

        let (spherical_distributions, mut other_distributions): (Vec<_>, Vec<_>) =
            distributions.into_iter().partition(|d| d.max_l == 0);
        // Grouping equal orders keeps the dispatch on `max_l` predictable.
        other_distributions.sort_by_key(|d| d.max_l);
        let spherical = spherical_distributions
            .into_iter()
            .map(|d| {
                let c = d.terms[0].1;
                (d.exponent, d.center, c, c * 0.5 * (PI / d.exponent).sqrt())
            })
            .collect();
        let max_l = other_distributions
            .iter()
            .map(|d| d.max_l)
            .max()
            .unwrap_or(0);
        Self {
            spherical,
            distributions: other_distributions,
            boys_table: BoysTable::new(max_l),
            boys: vec![0.0; max_l + 1],
            r: vec![[0.0; POTENTIAL_BATCH]; (max_l + 1).pow(4)],
        }
    }

    /// The (positive) electronic Coulomb potential at `point`.
    #[cfg(test)]
    pub(crate) fn evaluate(&mut self, point: &Point) -> f64 {
        let mut value = [0.0];
        self.evaluate_batch(&[*point], &mut value);
        value[0]
    }

    /// Writes the (positive) electronic Coulomb potential at each of `points` (at most
    /// `POTENTIAL_BATCH`) into `out`. Points are processed together, so that the work on
    /// each distribution runs over fixed-size arrays the compiler can vectorize.
    pub(crate) fn evaluate_batch(&mut self, points: &[Point], out: &mut [f64]) {
        assert!(!points.is_empty() && points.len() <= POTENTIAL_BATCH);
        // Unused lanes repeat the last point; their values are discarded.
        let last = points[points.len() - 1];
        let lane = |k: usize| points.get(k).unwrap_or(&last);
        let px: Lanes = std::array::from_fn(|k| lane(k).x);
        let py: Lanes = std::array::from_fn(|k| lane(k).y);
        let pz: Lanes = std::array::from_fn(|k| lane(k).z);

        let mut sum = [0.0; POTENTIAL_BATCH];
        for &(exponent, center, c, charge) in &self.spherical {
            let r2: Lanes = std::array::from_fn(|k| {
                let (dx, dy, dz) = (center.x - px[k], center.y - py[k], center.z - pz[k]);
                dx * dx + dy * dy + dz * dz
            });
            if r2
                .iter()
                .all(|r2| exponent * r2 >= BOYS_ASYMPTOTIC_THRESHOLD)
            {
                // Far from the whole batch, F_0(t) = sqrt(pi / t) / 2 exactly: the
                // distribution is a point charge.
                for k in 0..POTENTIAL_BATCH {
                    sum[k] += charge / r2[k].sqrt();
                }
            } else {
                for k in 0..POTENTIAL_BATCH {
                    sum[k] += c * self.boys_table.f0(exponent * r2[k]);
                }
            }
        }
        for d in &self.distributions {
            let pc: [Lanes; 3] = [
                std::array::from_fn(|k| d.center.x - px[k]),
                std::array::from_fn(|k| d.center.y - py[k]),
                std::array::from_fn(|k| d.center.z - pz[k]),
            ];
            hermite_coulomb(
                d.max_l,
                d.exponent,
                &pc,
                &self.boys_table,
                &mut self.boys,
                &mut self.r,
            );
            for &(index, c) in &d.terms {
                let r = &self.r[index];
                for k in 0..POTENTIAL_BATCH {
                    sum[k] += c * r[k];
                }
            }
        }
        out.copy_from_slice(&sum[..out.len()]);
    }
}

/// Fills `r` with the Hermite Coulomb integrals `R^n_tuv(alpha, pc)` for every
/// `n + t + u + v <= l` and every point of a batch (`pc` holds the components of `P - C`
/// for each point), stored at `((n * dim + t) * dim + u) * dim + v` with `dim = l + 1` (so
/// the `n = 0` ones come first, at `(t * dim + u) * dim + v`).
fn hermite_coulomb(
    l: usize,
    alpha: f64,
    pc: &[Lanes; 3],
    boys_table: &BoysTable,
    boys: &mut [f64],
    r: &mut [Lanes],
) {
    // Dispatching to a compile-time `l` lets the compiler unroll the recursion.
    let f = match l {
        0 => hermite_coulomb_fixed::<0>,
        1 => hermite_coulomb_fixed::<1>,
        2 => hermite_coulomb_fixed::<2>,
        3 => hermite_coulomb_fixed::<3>,
        4 => hermite_coulomb_fixed::<4>,
        5 => hermite_coulomb_fixed::<5>,
        6 => hermite_coulomb_fixed::<6>,
        _ => return hermite_coulomb_impl(l, alpha, pc, boys_table, boys, r),
    };
    f(alpha, pc, boys_table, boys, r)
}

fn hermite_coulomb_fixed<const L: usize>(
    alpha: f64,
    pc: &[Lanes; 3],
    boys_table: &BoysTable,
    boys: &mut [f64],
    r: &mut [Lanes],
) {
    hermite_coulomb_impl(L, alpha, pc, boys_table, boys, r)
}

#[inline(always)]
fn hermite_coulomb_impl(
    l: usize,
    alpha: f64,
    pc: &[Lanes; 3],
    boys_table: &BoysTable,
    boys: &mut [f64],
    r: &mut [Lanes],
) {
    let dim = l + 1;
    let block = dim * dim * dim;
    for k in 0..POTENTIAL_BATCH {
        let t = alpha * (pc[0][k] * pc[0][k] + pc[1][k] * pc[1][k] + pc[2][k] * pc[2][k]);
        boys_table.fill(l, t, boys);
        let mut factor = 1.0;
        for (n, f) in boys.iter().enumerate().take(l + 1) {
            r[n * block][k] = factor * f;
            factor *= -2.0 * alpha;
        }
    }
    for n in (0..l).rev() {
        let (lower, upper) = r.split_at_mut((n + 1) * block);
        let here = &mut lower[n * block..];
        let max = l - n;
        for tx in 0..=max {
            for ty in 0..=(max - tx) {
                for tz in 0..=(max - tx - ty) {
                    // R^n_tuv from R^(n+1) of the lowered index along the first nonzero axis.
                    let (axis, order, lowered) = if tx > 0 {
                        (0, tx, ((tx - 1) * dim + ty) * dim + tz)
                    } else if ty > 0 {
                        (1, ty, (ty - 1) * dim + tz)
                    } else if tz > 0 {
                        (2, tz, tz - 1)
                    } else {
                        continue;
                    };
                    let step = match axis {
                        0 => dim * dim,
                        1 => dim,
                        _ => 1,
                    };
                    let target = &mut here[(tx * dim + ty) * dim + tz];
                    let first = &upper[lowered];
                    let x = &pc[axis];
                    if order > 1 {
                        let second = &upper[lowered - step];
                        let factor = (order - 1) as f64;
                        for k in 0..POTENTIAL_BATCH {
                            target[k] = x[k] * first[k] + factor * second[k];
                        }
                    } else {
                        for k in 0..POTENTIAL_BATCH {
                            target[k] = x[k] * first[k];
                        }
                    }
                }
            }
        }
    }
}

// ------ special functions -----------------------------------------------

/// Fills `out[0..=n_max]` with `F_0(t), ..., F_n_max(t)`: the highest order from the power
/// series (or the closed form for large `t`) and the others by the downward recursion
/// `F_m = (2t F_(m+1) + exp(-t)) / (2m + 1)`, which is stable.
fn boys_all(n_max: usize, t: f64, out: &mut [f64]) {
    if t >= BOYS_ASYMPTOTIC_THRESHOLD {
        let exp_t = (-t).exp();
        let mut f = 0.5 * (PI / t).sqrt();
        out[0] = f;
        for m in 0..n_max {
            f = (((2 * m + 1) as f64) * f - exp_t) / (2.0 * t);
            out[m + 1] = f;
        }
        return;
    }
    let exp_t = (-t).exp();
    let mut term = 1.0 / (2 * n_max + 1) as f64;
    let mut sum = term;
    let mut k = 1;
    while term > sum * f64::EPSILON {
        term *= 2.0 * t / (2 * n_max + 2 * k + 1) as f64;
        sum += term;
        k += 1;
    }
    let mut f = exp_t * sum;
    out[n_max] = f;
    for m in (0..n_max).rev() {
        f = (2.0 * t * f + exp_t) / (2 * m + 1) as f64;
        out[m] = f;
    }
}

/// Spacing of the tabulated arguments of `BoysTable`.
const BOYS_TABLE_STEP: f64 = 0.1;

/// The index of the tabulated argument nearest to `t >= 0`. Adding 0.5 and truncating is
/// a plain conversion, unlike `f64::round`, which is a library call on baseline x86-64.
fn nearest_table_index(t: f64) -> usize {
    (t * (1.0 / BOYS_TABLE_STEP) + 0.5) as usize
}

/// Number of Taylor terms used by `BoysTable`.
const BOYS_TABLE_TERMS: usize = 8;

/// The Boys functions `F_0..=F_n_max`, tabulated below `BOYS_ASYMPTOTIC_THRESHOLD` and
/// interpolated with a Taylor expansion, since `d/dt F_n = -F_(n+1)`:
/// `F_n(t0 + d) = sum_k F_(n+k)(t0) (-d)^k / k!`. With `|d| <= step / 2 = 0.05` and
/// `K = 8` terms, the truncation error is at most
/// `F_n(t) * e^|d| * |d|^K / K! < 1e-15 * F_n(t)` (because `F_(n+K) <= F_n` and
/// `|d/dt ln F_n| <= 1`), so this is as accurate as the power series, only much faster.
pub(crate) struct BoysTable {
    n_max: usize,
    /// `F_(n+k)(t0) / k!` for `n` in `0..=n_max` and `k` in `0..K`, at every tabulated
    /// argument `t0`, row by row.
    coefficients: Vec<f64>,
}

impl BoysTable {
    pub(crate) fn new(n_max: usize) -> Self {
        let width = n_max + BOYS_TABLE_TERMS;
        let points = (BOYS_ASYMPTOTIC_THRESHOLD / BOYS_TABLE_STEP).round() as usize + 1;
        let mut boys = vec![0.0; width];
        let mut coefficients = Vec::with_capacity(points * (n_max + 1) * BOYS_TABLE_TERMS);
        for i in 0..points {
            boys_all(width - 1, i as f64 * BOYS_TABLE_STEP, &mut boys);
            for n in 0..=n_max {
                let mut factorial = 1.0;
                for k in 0..BOYS_TABLE_TERMS {
                    if k > 0 {
                        factorial *= k as f64;
                    }
                    coefficients.push(boys[n + k] / factorial);
                }
            }
        }
        Self {
            n_max,
            coefficients,
        }
    }

    /// `F_0(t)`. Above `BOYS_ASYMPTOTIC_THRESHOLD` it is exactly `sqrt(pi / t) / 2` in double
    /// precision.
    pub(crate) fn f0(&self, t: f64) -> f64 {
        if t >= BOYS_ASYMPTOTIC_THRESHOLD {
            return 0.5 * (PI / t).sqrt();
        }
        let i = nearest_table_index(t);
        let minus_d = i as f64 * BOYS_TABLE_STEP - t;
        let row_len = (self.n_max + 1) * BOYS_TABLE_TERMS;
        self.coefficients[i * row_len..][..BOYS_TABLE_TERMS]
            .iter()
            .rev()
            .fold(0.0, |sum, c_k| c_k + sum * minus_d)
    }

    /// Fills `out[0..=n_max]` with `F_0(t), ..., F_n_max(t)`, for `n_max` up to the one the
    /// table was built for.
    pub(crate) fn fill(&self, n_max: usize, t: f64, out: &mut [f64]) {
        debug_assert!(n_max <= self.n_max);
        if t >= BOYS_ASYMPTOTIC_THRESHOLD + 3.0 * n_max as f64 {
            // F_n(t) = Gamma(n + 1/2) / (2 t^(n + 1/2)) - Gamma(n + 1/2, t) / (2 t^(n + 1/2)),
            // and the incomplete gamma tail is below exp(-t) t^(n - 1/2) / Gamma(n + 1/2)
            // relative to F_n, i.e. < 1e-17 for t >= 40 + 3n: drop it (and exp(-t)).
            let inverse_2t = 0.5 / t;
            let mut f = 0.5 * (PI / t).sqrt();
            for (m, value) in out.iter_mut().enumerate().take(n_max + 1) {
                *value = f;
                f *= (2 * m + 1) as f64 * inverse_2t;
            }
            return;
        }
        if t >= BOYS_ASYMPTOTIC_THRESHOLD {
            boys_all(n_max, t, out);
            return;
        }
        let i = nearest_table_index(t);
        let minus_d = i as f64 * BOYS_TABLE_STEP - t;
        let row_len = (self.n_max + 1) * BOYS_TABLE_TERMS;
        let row = &self.coefficients[i * row_len..][..(n_max + 1) * BOYS_TABLE_TERMS];
        for (f, c) in out.iter_mut().zip(row.as_chunks::<BOYS_TABLE_TERMS>().0) {
            // Horner's scheme on sum_k F_(n+k)(t0) / k! * (-d)^k.
            *f = c.iter().rev().fold(0.0, |sum, c_k| c_k + sum * minus_d);
        }
    }
}

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

    use crate::basis::{AngularMomentum, Shell};

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
    fn test_electron_repulsion_tensor_matches_single_integrals() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut functions = Vec::new();
        for (angular, center) in [
            (AngularMomentum::S, Point::new(0.0, 0.0, 0.0)),
            (AngularMomentum::P, Point::new(0.0, 0.0, 0.0)),
            (AngularMomentum::D, Point::new(1.1, -0.4, 0.3)),
            (AngularMomentum::F, Point::new(-0.5, 0.9, -1.2)),
            (AngularMomentum::S, Point::new(-0.5, 0.9, -1.2)),
        ] {
            let shell = Arc::new(Shell {
                center,
                primitives: vec![
                    PrimitiveGaussian::new(0.4, rng.random_range(2.0..6.0), center),
                    PrimitiveGaussian::new(0.7, rng.random_range(0.1..1.5), center),
                ],
            });
            for angular_momentum in angular.cartesian_components() {
                functions.push(BasisFunction {
                    shell: Arc::clone(&shell),
                    angular_momentum,
                });
            }
        }
        let n = functions.len();
        let eri = electron_repulsion_tensor(&functions);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        // A sample of the quadruplets: the reference is slow.
                        if (i + 3 * j + 7 * k + 13 * l) % 29 != 0 {
                            continue;
                        }
                        let expected = electron_repulsion(
                            &functions[i],
                            &functions[j],
                            &functions[k],
                            &functions[l],
                        );
                        let actual = eri[[i, j, k, l]];
                        assert!(
                            (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                            "({i}{j}|{k}{l}): expected {expected} but was {actual}."
                        );
                    }
                }
            }
        }
    }

    #[rstest]
    fn test_boys_table_matches_series(
        #[values(0, 1, 3, 6, 12)] n_max: usize,
        #[values(
            0.0, 1e-9, 0.049, 0.05, 0.3, 2.0, 15.0, 15.04, 39.96, 39.99, 40.0, 42.9, 43.0, 57.9,
            58.0, 75.0, 76.0, 300.0
        )]
        t: f64,
    ) {
        let table = BoysTable::new(12);
        let mut all = vec![0.0; n_max + 1];
        table.fill(n_max, t, &mut all);
        assert_eq!(all[0], table.f0(t), "F_0({t})");
        for (n, actual) in all.iter().enumerate() {
            let expected = boys(n, t);
            assert!(
                ((actual - expected) / expected).abs() < 1e-14,
                "F_{n}({t}): expected {expected} but was {actual}."
            );
        }
    }

    #[rstest]
    fn test_all_boys_functions_match_single_ones(
        #[values(0, 1, 3, 6, 12)] n_max: usize,
        #[values(0.0, 1e-9, 0.3, 2.0, 15.0, 39.9, 40.0, 75.0)] t: f64,
    ) {
        let mut all = vec![0.0; n_max + 1];
        boys_all(n_max, t, &mut all);
        for (n, actual) in all.iter().enumerate() {
            let expected = boys(n, t);
            assert!(
                ((actual - expected) / expected).abs() < 1e-13,
                "F_{n}({t}): expected {expected} but was {actual}."
            );
        }
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
