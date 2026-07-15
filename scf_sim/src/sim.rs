use crate::basis::BasisSet;
use ndarray::{Array1, Array2, Array4};
use ndarray_linalg::{Eigh, Norm, UPLO};
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
    // FIXME: could exploit symmetry and compute only half the coefficients
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

fn compute_two_electron_contribution(p: &Array2<f64>, eri: &Array4<f64>) -> Array2<f64> {
    let n = p.dim().0;
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
            e += ((atoms[a].charge as f64) * (atoms[b].charge as f64)) / r;
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

struct RhfSetup {
    /// Hamiltonian
    h: Array2<f64>,

    /// Orthogonalizer
    x: Array2<f64>,

    /// Orbital overlap matrix
    s: Array2<f64>,

    /// First guess of molecular orbital coefficients
    c0: Array2<f64>,
}

fn setup_rhf_simulation(basis: &BasisSet) -> RhfSetup {
    let n = basis.num_contracted_gaussians();

    let s = basis.overlap_matrix();
    let t = basis.kinetic_energy_matrix();
    let v = basis.nuclear_attraction_matrix();

    // diagonal must be 1, and S must be symmetric
    for i in 0..n {
        assert!(approx_eq(s[[i, i]], 1.0, 1e-6), "S[{i},{i}] != 1");
    }
    assert_symmetric(&s, 1e-6);
    log::debug!(
        "||S - S^T||                    : {:.6e}",
        (s.to_owned() - s.t()).norm()
    );

    assert_symmetric(&t, 1e-6);
    log::debug!(
        "||T - T^T||                    : {:.6e}",
        (t.to_owned() - t.t()).norm()
    );

    assert_symmetric(&v, 1e-6);
    log::debug!(
        "||V - V^T||                    : {:.6e}",
        (v.to_owned() - v.t()).norm()
    );

    let h = &t + &v;

    assert_symmetric(&h, 1e-6);
    log::debug!(
        "||H - H^T||                    : {:.6e}",
        (h.to_owned() - h.t()).norm()
    );

    // Symmetric eigendecomposition of S: S = U * diag(d) * U^T
    let (eigenvalues, u): (Array1<f64>, Array2<f64>) = s.eigh(UPLO::Lower).unwrap();
    log::debug!(
        "||(U * diag(d) * U^T) - S||    : {:.6e}",
        ((u.clone().dot(&Array2::from_diag(&eigenvalues)).dot(&u.t())) - s.clone()).norm()
    );

    {
        let max_eigenvalue = *eigenvalues
            .iter()
            .reduce(|a, b| if a > b { a } else { b })
            .unwrap();
        let min_eigenvalue = *eigenvalues
            .iter()
            .reduce(|a, b| if a < b { a } else { b })
            .unwrap();
        // Print condition number
        log::debug!(
            "k(S)                           : {:.6e}",
            max_eigenvalue / min_eigenvalue
        );
    }

    // X = U * D^(-1/2) * U^T  — the canonical orthogonalization matrix
    let d_inv_sqrt = Array2::from_diag(&eigenvalues.mapv(|e| 1.0 / e.sqrt()));

    let x = u.dot(&d_inv_sqrt).dot(&u.t());

    // X must be symmetric
    assert_symmetric(&x, 1e-6);
    log::debug!(
        "||X - X^T||                    : {:.6e}",
        (x.to_owned() - x.t()).norm()
    );

    // X^T * S * X must equal the identity (canonical orthogonalization check)
    let should_be_identity = x.t().dot(&s).dot(&x);
    assert_matrix_approx_eq(&should_be_identity, &identity(n), 1e-6);
    log::debug!(
        "||(X^T * S * X) - I||          : {:.6e}",
        ((x.t().dot(&s).dot(&x)) - &identity(n)).norm()
    );

    // H' = X^T * H * X
    let h_prime = x.t().dot(&h).dot(&x);

    // H' must be symmetric (since H and X are both symmetric, X^T * H * X is too)
    assert_symmetric(&h_prime, 1e-6);
    log::debug!(
        "||H' - H'^T||                  : {:.6e}",
        (h_prime.to_owned() - h_prime.t()).norm()
    );

    // We ignore both eigenvalues and condition number since matrix H' is NOT positive definite
    let (orbital_energies, c_prime) = h_prime.eigh(UPLO::Lower).unwrap();
    log::debug!(
        "||(C' * diag(e) * C'^T) - H'|| : {:.6e}",
        ((c_prime
            .dot(&Array2::from_diag(&orbital_energies))
            .dot(&c_prime.t()))
            - h_prime)
            .norm()
    );

    let c = x.dot(&c_prime);

    RhfSetup { h, x, s, c0: c }
}

fn check_electron_repulsion_integrals(eri: &Array4<f64>, tolerance: f64) {
    let n = eri.dim().0;
    assert_eq!(n, eri.dim().1);
    assert_eq!(n, eri.dim().2);
    assert_eq!(n, eri.dim().3);
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                for d in 0..n {
                    let abcd = eri[[a, b, c, d]];
                    assert!(
                        approx_eq(abcd, eri[[b, a, c, d]], tolerance),
                        "ERI: ⟨{a}{b}|{c}{d}⟩ != ⟨{b}{a}|{c}{d}⟩"
                    );
                    assert!(
                        approx_eq(abcd, eri[[a, b, d, c]], tolerance),
                        "ERI: ⟨{a}{b}|{c}{d}⟩ != ⟨{a}{b}|{d}{c}⟩"
                    );
                    assert!(
                        approx_eq(abcd, eri[[c, d, a, b]], tolerance),
                        "ERI: ⟨{a}{b}|{c}{d}⟩ != ⟨{c}{d}|{a}{b}⟩"
                    );
                }
            }
        }
    }
}

pub(crate) fn run_rhf_simulation(
    atoms: &[Atom],
    basis: &BasisSet,
    opt_params: &OptimizationParameters,
) -> Array2<f64> {
    log::info!("Starting optimization");
    let beginning = Instant::now();

    let n = basis.num_contracted_gaussians();
    let n_electrons: usize = atoms.iter().map(|a| a.charge as usize).sum();
    let n_occ = basis.num_occupied_orbitals(n_electrons);

    let setup = setup_rhf_simulation(basis);

    let eri = basis.electron_repulsion_tensor();
    check_electron_repulsion_integrals(&eri, 1e-6);

    // initial guess density
    let mut p = Array2::<f64>::zeros((n, n));
    let mut p_new = Array2::<f64>::zeros((n, n));

    compute_density_matrix(n, n_occ, &setup.c0, &mut p);

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Optimization setup completed in {elapsed:?}.");
    }

    log::info!(" ### Optimization parameters ### ");
    log::info!(" Max Iterations : {}", opt_params.max_iterations);
    log::info!(" dE tolerance   : {:.6e}", opt_params.e_tol);
    log::info!(" dP tolerance   : {:.6e}", opt_params.p_tol);
    log::info!(" ### Optimization parameters ### ");

    let mut c;

    let mut e_old = 0.0;

    let mut iter = 0;
    let mut delta_e;
    let mut delta_p;

    let loop_beginning = Instant::now();
    loop {
        let iteration_beginning = Instant::now();

        // Build two-electron contribution G(P) (must be symmetric)
        let g = compute_two_electron_contribution(&p, &eri);
        assert_symmetric(&g, 1e-10);
        log::debug!(
            "||G - G^T||                    : {:.6e}",
            (g.to_owned() - g.t()).norm()
        );

        // F = H + G
        let f = &setup.h + &g;

        // The Fock (F) matrix must be symmetric
        assert_symmetric(&f, 1e-10);
        log::debug!(
            "||F - F^T||                    : {:.6e}",
            (f.to_owned() - f.t()).norm()
        );

        // F' = X^T F X
        let f_prime = setup.x.t().dot(&f).dot(&setup.x);

        // The transformed/orthogonalized Fock (F') matrix must be symmetric
        assert_symmetric(&f_prime, 1e-10);
        log::debug!(
            "||F' - F'^T||                  : {:.6e}",
            (f_prime.to_owned() - f_prime.t()).norm()
        );

        // diagonalize
        let (orbital_energies, c_prime) = f_prime.eigh(UPLO::Lower).unwrap();

        // Check that orbital energies are sorted
        for i in 0..(n - 1) {
            assert!(orbital_energies[i] <= orbital_energies[i + 1]);
        }

        // Double-check on reconstruction of the orthogonal Fock (F') matrix
        let reconstructed_f_prime = c_prime
            .dot(&Array2::from_diag(&orbital_energies))
            .dot(&c_prime.t());
        assert_matrix_approx_eq(&reconstructed_f_prime, &f_prime, 1e-10);
        log::debug!(
            "||(C' * diag(e) * F'^T) - F'|| : {:.6e}",
            (reconstructed_f_prime - f_prime).norm()
        );

        // AO coefficients
        c = setup.x.dot(&c_prime);

        // Double-check on the orbital coefficients
        let csc = c.t().dot(&setup.s).dot(&c);
        assert_matrix_approx_eq(&csc, &identity(n), 1e-10);
        log::debug!(
            "||(C^T * S * C) - I||          : {:.6e}",
            (csc - &identity(n)).norm()
        );

        let residual = &f.dot(&c) - &setup.s.dot(&c).dot(&Array2::from_diag(&orbital_energies));
        let residual_norm = residual.norm();
        log::debug!("||FC - SCE||                   : {:.6e}", residual_norm);
        assert!(
            residual_norm < 1e-8,
            "Roothaan residual too large: {}",
            residual_norm
        );

        // Density (must be symmetric)
        compute_density_matrix(n, n_occ, &c, &mut p_new);
        assert_symmetric(&p_new, 1e-6);
        log::debug!(
            "||P - P^T||                    : {:.6e}",
            (p_new.to_owned() - p_new.t()).norm()
        );

        // Density idempotency check: PSP = 2P
        log::debug!(
            "||P * S * P - 2 * P||          : {:.6e}",
            (p_new.dot(&setup.s).dot(&p_new) - 2.0 * &p_new).norm()
        );

        // Double-check on the electron count
        let electron_count = (&p_new * &setup.s).sum();
        log::debug!("N_e                            : {:.6}", electron_count);
        assert!(
            approx_eq(electron_count, n_electrons as f64, 1e-8),
            "Wrong electron count: expected {} but was {}.",
            electron_count,
            n_electrons
        );

        // RHF energy
        let e_elec = compute_electronic_energy(n, &p_new, &setup.h, &f);
        let e_nuclear = nuclear_repulsion_energy(atoms);
        let e_total = e_elec + e_nuclear;

        delta_e = (e_total - e_old).abs();
        delta_p = (&p_new - &p).mapv(|x| x * x).sum().sqrt();

        {
            let elapsed = (Instant::now() - iteration_beginning).as_secs_f64();
            let throughput = ((iter + 1) as f64) / (Instant::now() - loop_beginning).as_secs_f64();
            log::info!(
                "iter = {iter:3} | Ee = {e_elec:18.12} | En = {e_nuclear:18.12} | E = {e_total:18.12} | dE = {delta_e:.6e} | dP = {delta_p:.6e} | dt = {elapsed:.6e}s | thr. = {throughput:.6e} it/s"
            );
        }

        std::mem::swap(&mut p, &mut p_new);
        e_old = e_total;

        if iter >= opt_params.max_iterations
            || (delta_e < opt_params.e_tol && delta_p < opt_params.p_tol)
        {
            break;
        }

        iter += 1;
    }

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Optimization completed in {elapsed:?}.");
    }

    let reason;
    if iter >= opt_params.max_iterations {
        reason = "max iterations reached";
    } else if delta_e < opt_params.e_tol && delta_p < opt_params.p_tol {
        reason = "energy/density tolerance";
    } else {
        reason = "UNKNOWN";
    }
    log::info!("Optimization stopped because: {reason}.");

    c
}
