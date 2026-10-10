#![forbid(unsafe_code)]

use clap::Parser;
use scf_sim::{
    basis_reader::{build_basis, parse_nwchem_basis},
    cube_writer::{CubeFormat, dump, output_filenames},
    integrals,
    sim::{OptimizationParameters, run_rhf_simulation},
};
use simple_logger::SimpleLogger;
use std::path::Path;
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

    /// Cube file format(s) to write: orbitals, density, esp. Repeat the flag or
    /// comma-separate to request more than one.
    #[arg(
        short = 'f',
        long = "format",
        value_delimiter = ',',
        default_value = "orbitals,density"
    )]
    formats: Vec<CubeFormat>,

    /// Overwrite output files if they already exist
    #[arg(long, default_value_t = false)]
    force: bool,

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

/// Logs `message` as an error and exits, for user errors that do not deserve a stack trace.
fn fail(message: &str) -> ! {
    log::error!("{message}");
    std::process::exit(1);
}

fn check_input_file(path: &str, description: &str) {
    if !Path::new(path).is_file() {
        fail(&format!(
            "The {description} '{path}' does not exist or is not a file."
        ));
    }
}

/// Checks that every output file can be written: its directory must exist, and the file
/// itself must not exist unless `force` is set.
fn check_output_files(filenames: &[String], force: bool) {
    let mut existing = Vec::new();
    for filename in filenames {
        let path = Path::new(filename);
        if let Some(dir) = path.parent()
            && !dir.as_os_str().is_empty()
            && !dir.is_dir()
        {
            fail(&format!(
                "Cannot write output file '{filename}': directory '{}' does not exist.",
                dir.display()
            ));
        }
        if path.is_dir() {
            fail(&format!(
                "Cannot write output file '{filename}': it is a directory."
            ));
        }
        if path.exists() {
            existing.push(filename.as_str());
        }
    }

    if existing.is_empty() {
        return;
    }
    if force {
        log::warn!(
            "{} output file(s) already exist and will be overwritten.",
            existing.len()
        );
    } else {
        const MAX_SHOWN: usize = 5;
        let mut shown = existing
            .iter()
            .take(MAX_SHOWN)
            .map(|f| format!("'{f}'"))
            .collect::<Vec<String>>()
            .join(", ");
        if existing.len() > MAX_SHOWN {
            shown.push_str(&format!(" and {} more", existing.len() - MAX_SHOWN));
        }
        fail(&format!(
            "{} output file(s) already exist: {shown}. Use --force to overwrite them.",
            existing.len()
        ));
    }
}

fn main() -> std::io::Result<()> {
    let beginning = Instant::now();

    SimpleLogger::new()
        .env()
        .without_timestamps()
        .init()
        .unwrap();

    let args = Args::parse();

    check_input_file(&args.input_xyz, "input file");
    check_input_file(&args.basis_file, "basis set file");

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

    check_output_files(
        &output_filenames(&args.formats, &args.output_prefix, basis.functions.len()),
        args.force,
    );

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

    dump(
        &input_file.atoms,
        &basis,
        &c,
        &args.formats,
        args.output_prefix,
    )?;

    log::info!("All done!");

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Total execution time = {elapsed:?}");
    }

    Ok(())
}
