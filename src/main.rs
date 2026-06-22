#![forbid(unsafe_code)]

mod basis;
mod cube_writer;
mod integrals;
mod point;
mod sim;

use std::time::Instant;

use clap::Parser;

use simple_logger::SimpleLogger;

use crate::{
    basis::BasisSet,
    cube_writer::dump_molecular_orbital,
    point::Point,
    sim::{OptimizationParameters, run_rhf_simulation},
};

#[derive(Clone)]
struct Atom {
    z: f64,
    position: Point,
}

/// A small and simple simulator of Hartree-Fock method
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Maximum number of iterations for the SCF method
    #[arg(long, default_value_t = 100)]
    max_iterations: usize,

    /// Tolerance value for the SCF energy
    #[arg(long, default_value_t = 1.0e-10)]
    e_tol: f64,

    /// Tolerance value for the SCF density
    #[arg(long, default_value_t = 1.0e-8)]
    p_tol: f64,

    /// Prefix of the file where to write molecular orbitals
    #[arg(short, long, default_value = "mo_")]
    output_prefix: String,
}

fn main() -> std::io::Result<()> {
    let beginning = Instant::now();

    SimpleLogger::new().init().unwrap();

    let args = Args::parse();

    const R: f64 = 1.4; // bohr

    // 2 Hydrogen atoms
    let atoms = vec![
        Atom {
            z: 1.0,
            position: Point {
                x: 0.0,
                y: 0.0,
                z: -R / 2.0,
            },
        },
        Atom {
            z: 1.0,
            position: Point {
                x: 0.0,
                y: 0.0,
                z: R / 2.0,
            },
        },
    ];

    // Prepare the STO-3G basis
    let sto_3g = BasisSet::new(
        &[0.15432897, 0.53532814, 0.44463454],
        &[3.42525091, 0.62391373, 0.16885540],
        &atoms.iter().map(|a| a.position).collect::<Vec<Point>>(),
    );

    let opt_params = OptimizationParameters::new(args.max_iterations, args.e_tol, args.p_tol);

    let c = run_rhf_simulation(&atoms, &sto_3g, &opt_params);

    for mo_index in 0..c.ncols() {
        dump_molecular_orbital(
            &atoms,
            &sto_3g,
            mo_index,
            &c,
            format!("{}{}.cube", args.output_prefix, mo_index),
        )?;
    }

    log::info!("All done!");

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Total execution time = {elapsed:?}");
    }

    Ok(())
}
