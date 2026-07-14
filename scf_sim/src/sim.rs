use crate::basis::BasisSet;
use ndarray::{Array1, Array2, Array4};
use ndarray_linalg::{Eigh, UPLO};
use scf_core::Atom;
use std::time::Instant;

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
fn compute_g(n: usize, p: &Array2<f64>, eri: &Array4<f64>) -> Array2<f64> {
    let mut g = Array2::<f64>::zeros((n, n));
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;

            for lambda in 0..n {
                for sigma in 0..n {
                    sum += p[[lambda, sigma]]
                        * (eri[[mu, nu, lambda, sigma]] - 0.5 * eri[[mu, lambda, nu, sigma]]);
                }
            }

            g[[mu, nu]] = sum;
        }
    }
    g
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
            let r = atoms[a].position.distance(&atoms[b].position);
            e += (atoms[a].charge as f64).powi(2) / r;
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
    h: &mut Array2<f64>,
    x: &mut Array2<f64>,
) -> (Array2<f64>, Array4<f64>) {
    let n = basis.num_contracted_gaussians();

    let mut s = basis.overlap_matrix();
    let t = basis.kinetic_energy_matrix();
    let v = basis.nuclear_attraction_matrix();

    // diagonal must be 1, and S must be symmetric
    for i in 0..n {
        assert!(approx_eq(s[[i, i]], 1.0, 1e-6), "S[{i},{i}] != 1");
    }
    // Force diagonal entries of S to be exactly 1
    for i in 0..n {
        s[[i, i]] = 1.0;
    }
    assert_symmetric(&s, 1e-6);
    assert_symmetric(&t, 1e-6);
    assert_symmetric(&v, 1e-6);

    *h = &t + &v;

    assert_symmetric(h, 1e-6);

    // Symmetric eigendecomposition of S: S = U * diag(d) * U^T
    let (eigenvalues, u): (Array1<f64>, Array2<f64>) = s.eigh(UPLO::Lower).unwrap();

    let max_eigenvalue = *eigenvalues
        .iter()
        .filter(|v| **v > 1e-10)
        .reduce(|a, b| return if a > b { a } else { b })
        .unwrap();
    let min_eigenvalue = *eigenvalues
        .iter()
        .filter(|v| **v > 1e-10)
        .reduce(|a, b| return if a < b { a } else { b })
        .unwrap();
    log::info!("condition number : {}", max_eigenvalue / min_eigenvalue);

    // for e in eigenvalues.iter_mut() {
    //     if *e < 0.0 && e.abs() < 1e-10 {
    //         // *e = 0.0;
    //         *e = e.abs();
    //     }
    // }

    // X = U * D^(-1/2) * U^T  — the canonical orthogonalization matrix
    let d_inv_sqrt = Array2::from_diag(&eigenvalues.mapv(|e| 1.0 / e.sqrt()));

    *x = u.dot(&d_inv_sqrt).dot(&u.t());

    // X must be symmetric
    assert_symmetric(x, 1e-6);

    // X^T * S * X must equal the identity (canonical orthogonalization check)
    let should_be_identity = x.t().dot(&s).dot(x);
    println!("{should_be_identity:?}");
    assert_matrix_approx_eq(&should_be_identity, &identity(n), 1e-6);

    // H' = X^T * H * X
    let h_prime = x.t().dot(h).dot(x);

    // H' must be symmetric (since H and X are both symmetric, X^T * H * X is too)
    assert_symmetric(&h_prime, 1e-6);

    let (_epsilon, c_prime) = h_prime.eigh(UPLO::Lower).unwrap();

    let c = x.dot(&c_prime);

    let eri = basis.electron_repulsion_tensor();
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                for d in 0..n {
                    let abcd = eri[[a, b, c, d]];
                    assert!(
                        approx_eq(abcd, eri[[b, a, c, d]], 1e-6),
                        "ERI: ⟨{a}{b}|{c}{d}⟩ != ⟨{b}{a}|{c}{d}⟩"
                    );
                    assert!(
                        approx_eq(abcd, eri[[a, b, d, c]], 1e-6),
                        "ERI: ⟨{a}{b}|{c}{d}⟩ != ⟨{a}{b}|{d}{c}⟩"
                    );
                    assert!(
                        approx_eq(abcd, eri[[c, d, a, b]], 1e-6),
                        "ERI: ⟨{a}{b}|{c}{d}⟩ != ⟨{c}{d}|{a}{b}⟩"
                    );
                }
            }
        }
    }

    (c, eri)
}

pub(crate) fn run_rhf_simulation(
    atoms: &[Atom],
    basis: &BasisSet,
    opt_params: &OptimizationParameters,
) -> Array2<f64> {
    log::info!("Starting optimization");
    let beginning = Instant::now();

    let n = basis.num_contracted_gaussians();

    let mut h = Array2::<f64>::zeros((n, n));
    let mut x = Array2::<f64>::zeros((n, n));

    let (c, eri) = setup_rhf_simulation(basis, &mut h, &mut x);

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Optimization setup completed in {elapsed:?}.");
    }

    // initial guess density
    let mut p = Array2::<f64>::zeros((n, n));
    let mut p_new = Array2::<f64>::zeros((n, n));

    let n_electrons: usize = atoms.iter().map(|a| a.charge as usize).sum();
    let n_occ = basis.num_occupied_orbitals(n_electrons);

    let mut e_old = 0.0;

    let max_iterations = 100;
    let e_tol = 1e-10;
    let p_tol = 1e-8;

    log::info!(" ### Optimization parameters ### ");
    log::info!(" Max Iterations : {}", opt_params.max_iterations);
    log::info!(" dE tolerance   : {:.6e}", opt_params.e_tol);
    log::info!(" dP tolerance   : {:.6e}", opt_params.p_tol);
    log::info!(" ### Optimization parameters ### ");

    for iter in 0..max_iterations {
        let loop_beginning = Instant::now();

        // Build G(P)
        let g = compute_g(n, &p, &eri);

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

        {
            let elapsed = Instant::now() - loop_beginning;
            log::info!(
                "iter {iter:3} E = {e_total:20.12} dE = {delta_e:12.5e} dP = {delta_p:12.5e} time = {elapsed:?}"
            );
        }

        if delta_e < e_tol && delta_p < p_tol {
            break;
        }

        p = p_new.clone();
        e_old = e_total;
    }

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Optimization completed in {elapsed:?}.");
    }

    c
}
