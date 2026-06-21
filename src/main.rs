#![forbid(unsafe_code)]

mod basis;

use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};

use crate::basis::{BasisSet, Point};

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

fn main() {
    const R: f64 = 1.4; // bohr

    // Prepare the STO-3G basis
    let sto_3g = BasisSet::new(
        &[0.15432897, 0.53532814, 0.44463454],
        &[3.42525091, 0.62391373, 0.16885540],
        // 2 Hydrogen atoms
        &[
            Point {
                x: 0.0,
                y: 0.0,
                z: -R / 2.0,
            },
            Point {
                x: 0.0,
                y: 0.0,
                z: R / 2.0,
            },
        ],
    );

    let n = sto_3g.num_contracted_gaussians();

    // Build S, T, V as Array2
    let mut s = Array2::<f64>::zeros((n, n));
    let mut t = Array2::<f64>::zeros((n, n));
    let mut v = Array2::<f64>::zeros((n, n));

    sto_3g.compute_contracted_gaussians_overlap(&mut s);
    sto_3g.compute_contracted_gaussians_kinetic_energy(&mut t);
    sto_3g.compute_contracted_gaussians_nuclear_attraction(&mut v);

    println!("Overlap (S):\n{s:?}\n");

    // diagonal must be 1, and S must be symmetric
    for i in 0..n {
        assert!(approx_eq(s[[i, i]], 1.0, 1e-6), "S[{i},{i}] != 1");
    }
    assert_symmetric(&s, 1e-6);

    println!("Kinetic energy (T):\n{t:?}\n");

    assert_symmetric(&t, 1e-6);

    println!("Nuclear attraction (V):\n{v:?}\n");

    assert_symmetric(&v, 1e-6);

    let h = &t + &v;
    println!("Hamiltonian (H):\n{h:?}\n");

    assert_symmetric(&h, 1e-6);

    println!("Electron Repulsion Integrals:");
    let mut eri: Vec<Vec<Vec<Vec<f64>>>> = vec![
        vec![
            vec![
                vec![0.0; sto_3g.num_contracted_gaussians()];
                sto_3g.num_contracted_gaussians()
            ];
            sto_3g.num_contracted_gaussians()
        ];
        sto_3g.num_contracted_gaussians()
    ];

    sto_3g.compute_electron_repulsion(&mut eri);

    for (a, eri_a) in eri
        .iter()
        .enumerate()
        .take(sto_3g.num_contracted_gaussians())
    {
        for (b, eri_a_b) in eri_a
            .iter()
            .enumerate()
            .take(sto_3g.num_contracted_gaussians())
        {
            for (c, eri_a_b_c) in eri_a_b
                .iter()
                .enumerate()
                .take(sto_3g.num_contracted_gaussians())
            {
                for (d, eri_a_b_c_d) in eri_a_b_c
                    .iter()
                    .enumerate()
                    .take(sto_3g.num_contracted_gaussians())
                {
                    println!("⟨{a}{b}|{c}{d}⟩ = {eri_a_b_c_d}");
                }
            }
        }
    }

    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                for d in 0..n {
                    let abcd = eri[a][b][c][d];
                    assert!(
                        approx_eq(abcd, eri[b][a][c][d], 1e-6),
                        "ERI: ⟨{a}{b}|{c}{d}⟩ != ⟨{b}{a}|{c}{d}⟩"
                    );
                    assert!(
                        approx_eq(abcd, eri[a][b][d][c], 1e-6),
                        "ERI: ⟨{a}{b}|{c}{d}⟩ != ⟨{a}{b}|{d}{c}⟩"
                    );
                    assert!(
                        approx_eq(abcd, eri[c][d][a][b], 1e-6),
                        "ERI: ⟨{a}{b}|{c}{d}⟩ != ⟨{c}{d}|{a}{b}⟩"
                    );
                }
            }
        }
    }

    println!();

    // Symmetric eigendecomposition of S: S = U * diag(d) * U^T
    let (eigenvalues, u): (Array1<f64>, Array2<f64>) = s.eigh(UPLO::Lower).unwrap();

    // X = U * D^(-1/2) * U^T  — the canonical orthogonalization matrix
    let d_inv_sqrt = Array2::from_diag(&eigenvalues.mapv(|e| 1.0 / e.sqrt()));

    let x = u.dot(&d_inv_sqrt).dot(&u.t());
    println!("X:\n{x:?}\n");

    // X must be symmetric
    assert_symmetric(&x, 1e-6);

    // X^T * S * X must equal the identity (canonical orthogonalization check)
    let should_be_identity = x.t().dot(&s).dot(&x);
    assert_matrix_approx_eq(&should_be_identity, &identity(n), 1e-6);

    // H' = X^T * H * X
    let h_prime = x.t().dot(&h).dot(&x);
    println!("H':\n{h_prime:?}\n");

    // H' must be symmetric (since H and X are both symmetric, X^T * H * X is too)
    assert_symmetric(&h_prime, 1e-6);

    let (epsilon, c_prime) = h_prime.eigh(UPLO::Lower).unwrap();

    println!("Molecular Orbital coefficients (C'):\n{c_prime:?}\n");
    println!("Molecular Orbital energies (epsilon):\n{epsilon:?}\n");

    // Build the density matrix
    let mut p = Array2::<f64>::zeros((n, n));

    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;
            for i in 0..n {
                sum += c_prime[[mu, i]] * c_prime[[nu, i]];
            }

            p[[mu, nu]] = 2.0 * sum;
        }
    }

    println!("Density (P):\n{p:?}\n");

    let mut g = Array2::<f64>::zeros((n, n));
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

    println!("G:\n{g:?}\n");

    let f = h_prime + g;

    println!("Fock (F):\n{f:?}\n");
}
