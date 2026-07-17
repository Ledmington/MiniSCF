#![forbid(unsafe_code)]

mod basis;
mod basis_reader;
mod cube_writer;
mod integrals;
mod sim;

use crate::{
    basis_reader::{build_basis, parse_nwchem_basis},
    cube_writer::dump_all_molecular_orbitals,
    sim::{OptimizationParameters, run_rhf_simulation},
};
use clap::Parser;
use simple_logger::SimpleLogger;
use std::time::Instant;
use xyz::read_xyz;

/// A small and simple simulator of Hartree-Fock method
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// File to read molecule from (coordinates are assumed to be expressed in Bohr units)
    #[arg(short, long, default_value = "input.xyz")]
    input_xyz: String,

    /// File to read basis set from
    #[arg(short, long, default_value = "basis.gbs")]
    basis_file: String,

    /// Prefix of the file where to write molecular orbitals
    #[arg(short, long, default_value = "mo_")]
    output_prefix: String,

    /// Maximum number of iterations for the SCF method
    #[arg(long, default_value_t = 100)]
    max_iterations: usize,

    /// Tolerance value for the SCF energy (Hartree)
    #[arg(long, default_value_t = 1.0e-10)]
    e_tol: f64,

    /// Tolerance value for the SCF density
    #[arg(long, default_value_t = 1.0e-8)]
    p_tol: f64,

    /// Minimum residual to reach
    #[arg(long, default_value_t = 1.0e-15)]
    min_residual: f64,
}

fn main() -> std::io::Result<()> {
    let beginning = Instant::now();

    SimpleLogger::new()
        .env()
        .without_timestamps()
        .init()
        .unwrap();

    let args = Args::parse();

    let input_file = read_xyz(&args.input_xyz).unwrap_or_else(|err| {
        panic!(
            "Could not read input file '{}' because:\n{}.",
            args.input_xyz, err
        )
    });

    log::info!(" ### Input system ### ");
    for atom in input_file.atoms.iter() {
        log::info!(
            " {} {} {} {} {}",
            atom.element.symbol,
            atom.charge,
            atom.position.x,
            atom.position.y,
            atom.position.z
        );
    }
    log::info!(" ### Input system ### ");

    let basis_library = parse_nwchem_basis(&args.basis_file).unwrap_or_else(|err| {
        panic!(
            "Could not read input file '{}' because:\n{}.",
            args.basis_file, err
        )
    });

    let basis = build_basis(&input_file.atoms, &basis_library);

    log::info!("Checking that the basis set is normalized...");
    for bf in basis.functions.iter() {
        let actual_overlap = integrals::overlap(bf, bf);
        let expected_overlap = 1.0;
        if (actual_overlap - expected_overlap).abs() > 1e-10 {
            log::warn!(
                "The basis function {bf:?} is not normalized: expected overlap with itself to be {expected_overlap} but was {actual_overlap}."
            );
        }
    }
    log::info!("Check complete.");

    let opt_params = OptimizationParameters::new(
        args.max_iterations,
        args.e_tol,
        args.p_tol,
        args.min_residual,
    );

    let c = run_rhf_simulation(&input_file.atoms, &basis, &opt_params);

    dump_all_molecular_orbitals(&input_file.atoms, &basis, &c, args.output_prefix)?;

    log::info!("All done!");

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Total execution time = {elapsed:?}");
    }

    Ok(())
}
