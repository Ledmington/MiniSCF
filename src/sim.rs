use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};

use crate::{Atom, basis::BasisSet};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn assert_matrix_approx_eq(a: &Array2<f64>, b: &Array2<f64>, tol: f64) {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    for ((i, j), val) in a.indexed_iter() {
        assert!(
            approx_eq(*val, b[[i, j]], tol),
            "matrices differ at [{i},{j}]: {} vs {}",
            val,
            b[[i, j]]
        );
    }
}

fn assert_symmetric(m: &Array2<f64>, tol: f64) {
    assert_matrix_approx_eq(m, &m.t().to_owned(), tol);
}

fn identity(n: usize) -> Array2<f64> {
    Array2::from_diag(&Array1::ones(n))
}

fn compute_density_matrix(n: usize, n_occ: usize, c: &Array2<f64>, p: &mut Array2<f64>) {
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;
            for i in 0..n_occ {
                sum += c[[mu, i]] * c[[nu, i]];
            }

            p[[mu, nu]] = 2.0 * sum;
        }
    }
}

// TODO: what is G?
fn compute_g(n: usize, p: &Array2<f64>, eri: &[Vec<Vec<Vec<f64>>>], g: &mut Array2<f64>) {
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;

            for lambda in 0..n {
                for sigma in 0..n {
                    sum += p[[lambda, sigma]]
                        * (eri[mu][nu][lambda][sigma] - 0.5 * eri[mu][lambda][nu][sigma]);
                }
            }

            g[[mu, nu]] = sum;
        }
    }
}

fn compute_electronic_energy(n: usize, p: &Array2<f64>, h: &Array2<f64>, f: &Array2<f64>) -> f64 {
    let mut e_elec = 0.0;
    for mu in 0..n {
        for nu in 0..n {
            e_elec += p[[mu, nu]] * (h[[mu, nu]] + f[[mu, nu]]);
        }
    }
    e_elec *= 0.5;
    e_elec
}

fn nuclear_repulsion_energy(atoms: &[Atom]) -> f64 {
    let mut e = 0.0;

    for a in 0..atoms.len() {
        for b in (a + 1)..atoms.len() {
            let dx = atoms[a].position.x - atoms[b].position.x;
            let dy = atoms[a].position.y - atoms[b].position.y;
            let dz = atoms[a].position.z - atoms[b].position.z;

            let r = (dx * dx + dy * dy + dz * dz).sqrt();

            e += atoms[a].z * atoms[b].z / r;
        }
    }

    e
}

pub(crate) struct OptimizationParameters {
    max_iterations: usize,
    e_tol: f64,
    p_tol: f64,
}

impl OptimizationParameters {
    pub(crate) fn new(max_iterations: usize, e_tol: f64, p_tol: f64) -> Self {
        assert!(e_tol > 0.0);
        assert!(p_tol > 0.0);
        OptimizationParameters {
            max_iterations,
            e_tol,
            p_tol,
        }
    }
}

fn setup_rhf_simulation(
    basis: &BasisSet,
    eri: &mut [Vec<Vec<Vec<f64>>>],
    h: &mut Array2<f64>,
    x: &mut Array2<f64>,
) -> Array2<f64> {
    let n = basis.num_contracted_gaussians();

    // Build S, T, V as Array2
    let mut s = Array2::<f64>::zeros((n, n));
    let mut t = Array2::<f64>::zeros((n, n));
    let mut v = Array2::<f64>::zeros((n, n));

    basis.compute_contracted_gaussians_overlap(&mut s);
    basis.compute_contracted_gaussians_kinetic_energy(&mut t);
    basis.compute_contracted_gaussians_nuclear_attraction(&mut v);

    // diagonal must be 1, and S must be symmetric
    for i in 0..n {
        assert!(approx_eq(s[[i, i]], 1.0, 1e-6), "S[{i},{i}] != 1");
    }
    assert_symmetric(&s, 1e-6);

    assert_symmetric(&t, 1e-6);

    assert_symmetric(&v, 1e-6);

    *h = &t + &v;

    assert_symmetric(h, 1e-6);

    basis.compute_electron_repulsion(eri);

    for (a, eri_a) in eri.iter().enumerate().take(n) {
        for (b, eri_a_b) in eri_a.iter().enumerate().take(n) {
            for (c, eri_a_b_c) in eri_a_b.iter().enumerate().take(n) {
                for (d, eri_a_b_c_d) in eri_a_b_c.iter().enumerate().take(n) {
                    assert!(
                        approx_eq(*eri_a_b_c_d, eri[b][a][c][d], 1e-6),
                        "ERI: ⟨{a}{b}|{c}{d}⟩ != ⟨{b}{a}|{c}{d}⟩"
                    );
                    assert!(
                        approx_eq(*eri_a_b_c_d, eri[a][b][d][c], 1e-6),
                        "ERI: ⟨{a}{b}|{c}{d}⟩ != ⟨{a}{b}|{d}{c}⟩"
                    );
                    assert!(
                        approx_eq(*eri_a_b_c_d, eri[c][d][a][b], 1e-6),
                        "ERI: ⟨{a}{b}|{c}{d}⟩ != ⟨{c}{d}|{a}{b}⟩"
                    );
                }
            }
        }
    }

    // Symmetric eigendecomposition of S: S = U * diag(d) * U^T
    let (eigenvalues, u): (Array1<f64>, Array2<f64>) = s.eigh(UPLO::Lower).unwrap();

    // X = U * D^(-1/2) * U^T  — the canonical orthogonalization matrix
    let d_inv_sqrt = Array2::from_diag(&eigenvalues.mapv(|e| 1.0 / e.sqrt()));

    *x = u.dot(&d_inv_sqrt).dot(&u.t());

    // X must be symmetric
    assert_symmetric(x, 1e-6);

    // X^T * S * X must equal the identity (canonical orthogonalization check)
    let should_be_identity = x.t().dot(&s).dot(x);
    assert_matrix_approx_eq(&should_be_identity, &identity(n), 1e-6);

    // H' = X^T * H * X
    let h_prime = x.t().dot(h).dot(x);

    // H' must be symmetric (since H and X are both symmetric, X^T * H * X is too)
    assert_symmetric(&h_prime, 1e-6);

    let (_epsilon, c_prime) = h_prime.eigh(UPLO::Lower).unwrap();

    // C = X * C'
    x.dot(&c_prime)
}

pub(crate) fn run_rhf_simulation(
    atoms: &[Atom],
    basis: &BasisSet,
    opt_params: &OptimizationParameters,
) -> Array2<f64> {
    let n = basis.num_contracted_gaussians();

    let mut eri: Vec<Vec<Vec<Vec<f64>>>> =
        vec![
            vec![
                vec![vec![0.0; basis.num_contracted_gaussians()]; basis.num_contracted_gaussians()];
                basis.num_contracted_gaussians()
            ];
            basis.num_contracted_gaussians()
        ];
    let mut h = Array2::<f64>::zeros((n, n));
    let mut x = Array2::<f64>::zeros((n, n));

    let c = setup_rhf_simulation(basis, &mut eri, &mut h, &mut x);

    // initial guess density
    let mut p = Array2::<f64>::zeros((n, n));
    let mut p_new = Array2::<f64>::zeros((n, n));

    let mut g = Array2::<f64>::zeros((n, n));

    let n_occ = basis.num_occupied_orbitals();

    let mut e_old = 0.0;

    let max_iterations = 100;
    let e_tol = 1e-10;
    let p_tol = 1e-8;

    println!(" ### Optimization parameters ### ");
    println!(" Max Iterations : {}", opt_params.max_iterations);
    println!(" dE tolerance   : {:.6e}", opt_params.e_tol);
    println!(" dP tolerance   : {:.6e}", opt_params.p_tol);
    println!();

    for iter in 0..max_iterations {
        // Build G(P)
        compute_g(n, &p, &eri, &mut g);

        // F = H + G
        let f = &h + &g;

        // F' = X^T F X
        let f_prime = x.t().dot(&f).dot(&x);

        // diagonalize
        let (_eps, c_prime) = f_prime.eigh(UPLO::Lower).unwrap();

        // AO coefficients
        let c = x.dot(&c_prime);

        // density
        compute_density_matrix(n, n_occ, &c, &mut p_new);

        // RHF energy
        let e_elec = compute_electronic_energy(n, &p_new, &h, &f);
        let e_nuclear = nuclear_repulsion_energy(atoms);
        let e_total = e_elec + e_nuclear;

        let delta_e = (e_total - e_old).abs();

        let delta_p = (&p_new - &p).mapv(|x| x * x).sum().sqrt();

        println!("iter {iter:3} E = {e_total:20.12} dE = {delta_e:12.5e} dP = {delta_p:12.5e}");

        if delta_e < e_tol && delta_p < p_tol {
            break;
        }

        p = p_new.clone();
        e_old = e_total;
    }

    c
}
