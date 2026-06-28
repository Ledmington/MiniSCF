#![forbid(unsafe_code)]

mod atom;
mod basis;
mod basis_reader;
mod cube_writer;
mod integrals;
mod point;
mod sim;

use crate::basis::{AngularMomentum, PrimitiveGaussian};
use crate::{
    atom::Atom,
    basis::{BasisSet, Shell},
    basis_reader::{build_basis, parse_nwchem_basis},
    cube_writer::dump_all_molecular_orbitals,
    point::Point,
    sim::{OptimizationParameters, run_rhf_simulation},
};
use clap::Parser;
use ndarray::Array2;
use simple_logger::SimpleLogger;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    sync::Arc,
    time::Instant,
};

/// A small and simple simulator of Hartree-Fock method
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// File to read molecule from
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

    /// Tolerance value for the SCF energy
    #[arg(long, default_value_t = 1.0e-10)]
    e_tol: f64,

    /// Tolerance value for the SCF density
    #[arg(long, default_value_t = 1.0e-8)]
    p_tol: f64,
}

// TODO: move this into sim.rs
pub(crate) struct SCF {
    pub(crate) basis: BasisSet,
    pub(crate) n_electrons: usize,
    pub(crate) density: Array2<f64>,
}

// TODO: move this into Atom
fn atomic_number(symbol: &str) -> Result<u8, String> {
    match symbol {
        "H" => Ok(1),
        "He" => Ok(2),
        "Li" => Ok(3),
        "Be" => Ok(4),
        "B" => Ok(5),
        "C" => Ok(6),
        "N" => Ok(7),
        "O" => Ok(8),
        "F" => Ok(9),
        "Ne" => Ok(10),
        _ => Err(format!("unknown element: {}", symbol)),
    }
}

fn main() -> std::io::Result<()> {
    let beginning = Instant::now();

    SimpleLogger::new().init().unwrap();

    let args = Args::parse();

    let atoms = read_xyz(&args.input_xyz).unwrap_or_else(|err| {
        panic!(
            "Could not read input file '{}' because:\n{}.",
            args.input_xyz, err
        )
    });

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

    let basis_library = parse_nwchem_basis(&args.basis_file).unwrap_or_else(|err| {
        panic!(
            "Could not read input file '{}' because:\n{}.",
            args.basis_file, err
        )
    });

    let basis = build_basis(&atoms, &basis_library);

    log::info!("basis: {:#?}", basis);
    log::info!("{} basis functions", basis.functions.len());
    log::info!("{} shells", basis.shells.len());

    let opt_params = OptimizationParameters::new(args.max_iterations, args.e_tol, args.p_tol);

    let c = run_rhf_simulation(&atoms, &basis, &opt_params);

    dump_all_molecular_orbitals(&atoms, &basis, &c, args.output_prefix)?;

    log::info!("All done!");

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Total execution time = {elapsed:?}");
    }

    Ok(())
}
