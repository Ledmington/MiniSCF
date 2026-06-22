#![forbid(unsafe_code)]

mod atom;
mod basis;
mod cube_writer;
mod integrals;
mod point;
mod sim;

use std::time::Instant;

use clap::Parser;

use ndarray::Array2;
use simple_logger::SimpleLogger;

use crate::{
    atom::Atom,
    basis::BasisSet,
    cube_writer::dump_all_molecular_orbitals,
    point::Point,
    sim::{OptimizationParameters, run_rhf_simulation},
};

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

pub struct SCF {
    pub basis: BasisSet,
    pub n_electrons: usize,
    pub density: Array2<f64>,
}

fn main() -> std::io::Result<()> {
    let beginning = Instant::now();

    SimpleLogger::new().init().unwrap();

    let args = Args::parse();

    const R: f64 = 1.4; // bohr

    // 2 Hydrogen atoms
    let atoms = vec![
        Atom {
            symbol: "H".to_string(),
            charge: 1,
            position: Point {
                x: 0.0,
                y: 0.0,
                z: -R / 2.0,
            },
        },
        Atom {
            symbol: "H".to_string(),
            charge: 1,
            position: Point {
                x: 0.0,
                y: 0.0,
                z: R / 2.0,
            },
        },
    ];

    log::info!(" ### Input system ### ");
    for atom in atoms.iter() {
        log::info!(
            " {} {} {} {} {}",
            atom.symbol,
            atom.charge,
            atom.position.x,
            atom.position.y,
            atom.position.z
        );
    }
    log::info!(" ### Input system ### ");

    // Prepare the STO-3G basis
    let sto_3g = BasisSet::new(
        &[0.15432897, 0.53532814, 0.44463454],
        &[3.42525091, 0.62391373, 0.16885540],
        &atoms.iter().map(|a| a.position).collect::<Vec<Point>>(),
    );

    let opt_params = OptimizationParameters::new(args.max_iterations, args.e_tol, args.p_tol);

    let c = run_rhf_simulation(&atoms, &sto_3g, &opt_params);

    dump_all_molecular_orbitals(&atoms, &sto_3g, &c, args.output_prefix)?;

    log::info!("All done!");

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Total execution time = {elapsed:?}");
    }

    Ok(())
}
