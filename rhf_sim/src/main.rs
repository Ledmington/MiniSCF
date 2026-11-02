#![forbid(unsafe_code)]

use clap::{Parser, ValueEnum, builder::NonEmptyStringValueParser};
use scf_core::{Atom, units::ANGSTROM_TO_BOHR};
use scf_sim::{
    basis_reader::{build_basis, parse_nwchem_basis},
    cube_writer::{CubeFormat, dump, output_filenames},
    integrals,
    sim::{OptimizationParameters, run_rhf_simulation},
};
use std::fs::File;
use std::path::Path;
use std::time::Instant;
use xyz::read_xyz;

mod logger;

/// A small and simple simulator of Hartree-Fock method
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// File to read molecule from (coordinates are expressed in the units given by --units)
    #[arg(short, long, default_value = "input.xyz")]
    input_xyz: String,

    /// Length units of the coordinates in the input file
    #[arg(short, long, value_enum, default_value_t = LengthUnit::Angstrom)]
    units: LengthUnit,

    /// File to read basis set from
    #[arg(short, long, default_value = "basis.gbs")]
    basis_file: String,

    /// Prefix of the output files: writes `<PREFIX>_mo_<N>.cube`, `<PREFIX>_density.cube`
    /// and `<PREFIX>_esp.cube`
    #[arg(short, long, default_value = "rhf", value_parser = NonEmptyStringValueParser::new())]
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

    /// Write the log to this file instead of the terminal (errors are still shown on the
    /// terminal)
    #[arg(short, long)]
    log_file: Option<String>,

    /// Overwrite output files (cube files and log file) if they already exist
    #[arg(long, default_value_t = false)]
    force: bool,

    /// Maximum number of iterations for the SCF method
    #[arg(long, default_value_t = 100)]
    max_iterations: usize,

    /// Tolerance value for the SCF energy (Hartree)
    #[arg(long, default_value = "1e-10")]
    e_tol: f64,

    /// Tolerance value for the SCF density
    #[arg(long, default_value = "1e-8")]
    p_tol: f64,

    /// Minimum residual to reach
    #[arg(long, default_value = "1e-15")]
    min_residual: f64,
}

/// Length units of the input coordinates. The simulation itself always works in bohr.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum LengthUnit {
    /// Bohr (atomic units)
    #[value(alias = "au")]
    Bohr,
    /// Ångström
    #[value(alias = "ang")]
    Angstrom,
}

impl LengthUnit {
    /// The factor that converts a length in these units to bohr.
    fn to_bohr_factor(self) -> f64 {
        match self {
            LengthUnit::Bohr => 1.0,
            LengthUnit::Angstrom => ANGSTROM_TO_BOHR,
        }
    }
}

/// The atoms with their positions converted from `unit` to bohr.
fn atoms_in_bohr(atoms: &[Atom], unit: LengthUnit) -> Vec<Atom> {
    let factor = unit.to_bohr_factor();
    atoms
        .iter()
        .map(|atom| Atom {
            position: atom.position.scale(factor),
            ..atom.clone()
        })
        .collect()
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

    logger::init();

    let args = Args::parse();

    if let Some(log_file) = &args.log_file {
        check_output_files(std::slice::from_ref(log_file), args.force);
        match File::create(log_file) {
            Ok(file) => logger::redirect_to_file(file),
            Err(err) => fail(&format!("Cannot create log file '{log_file}': {err}.")),
        }
    }

    check_input_file(&args.input_xyz, "input file");
    check_input_file(&args.basis_file, "basis set file");

    let input_file = read_xyz(&args.input_xyz).unwrap_or_else(|err| {
        fail(&format!(
            "Could not read input file '{}': {err}.",
            args.input_xyz
        ))
    });
    let atoms = atoms_in_bohr(&input_file.atoms, args.units);

    log::info!(" ### Input system (bohr, read as {:?}) ### ", args.units);
    for atom in atoms.iter() {
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
        fail(&format!(
            "Could not read basis set file '{}': {err}.",
            args.basis_file
        ))
    });

    let basis = build_basis(&atoms, &basis_library);

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

    let c = run_rhf_simulation(&atoms, &basis, &opt_params);

    dump(&atoms, &basis, &c, &args.formats, args.output_prefix)?;

    log::info!("All done!");

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Total execution time = {elapsed:?}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use scf_core::point::Point;

    use super::*;

    fn hydrogen_at(x: f64, y: f64, z: f64) -> Atom {
        Atom {
            element: element::HYDROGEN,
            position: Point::new(x, y, z),
            charge: 1,
        }
    }

    #[test]
    fn bohr_coordinates_are_unchanged() {
        let atoms = [hydrogen_at(0.0, 0.0, 0.0), hydrogen_at(1.4, -0.2, 3.0)];
        let converted = atoms_in_bohr(&atoms, LengthUnit::Bohr);
        for (original, actual) in atoms.iter().zip(&converted) {
            assert_eq!(original.position, actual.position);
        }
    }

    #[test]
    fn angstrom_coordinates_are_converted_to_bohr() {
        let atoms = [hydrogen_at(0.74, 0.0, -1.0)];
        let converted = atoms_in_bohr(&atoms, LengthUnit::Angstrom);
        let expected = Point::new(0.74 * 1.889_726_124_626, 0.0, -1.889_726_124_626);
        assert!(
            converted[0].position.distance(&expected) < 1e-10,
            "Expected {expected:?} but was {:?}.",
            converted[0].position
        );
        assert_eq!(converted[0].element, atoms[0].element);
        assert_eq!(converted[0].charge, atoms[0].charge);
    }

    #[test]
    fn output_prefix_defaults_to_rhf_and_rejects_empty() {
        let args = Args::try_parse_from(["rhf_sim"]).unwrap();
        assert_eq!(args.output_prefix, "rhf");
        let custom = Args::try_parse_from(["rhf_sim", "-o", "mysim"]).unwrap();
        assert_eq!(custom.output_prefix, "mysim");
        assert!(Args::try_parse_from(["rhf_sim", "--output-prefix", ""]).is_err());
    }

    #[test]
    fn tolerance_defaults_parse_from_scientific_notation() {
        let args = Args::try_parse_from(["rhf_sim"]).unwrap();
        assert_eq!(args.e_tol, 1.0e-10);
        assert_eq!(args.p_tol, 1.0e-8);
        assert_eq!(args.min_residual, 1.0e-15);
    }

    #[test]
    fn log_file_flag_is_optional() {
        assert_eq!(Args::try_parse_from(["rhf_sim"]).unwrap().log_file, None);
        let args = Args::try_parse_from(["rhf_sim", "--log-file", "run.log"]).unwrap();
        assert_eq!(args.log_file.as_deref(), Some("run.log"));
        let short = Args::try_parse_from(["rhf_sim", "-l", "run.log"]).unwrap();
        assert_eq!(short.log_file.as_deref(), Some("run.log"));
    }

    #[test]
    fn units_flag_defaults_to_angstrom() {
        let args = Args::try_parse_from(["rhf_sim"]).unwrap();
        assert_eq!(args.units, LengthUnit::Angstrom);
    }

    #[test]
    fn units_flag_accepts_names_and_aliases() {
        for (value, expected) in [
            ("bohr", LengthUnit::Bohr),
            ("au", LengthUnit::Bohr),
            ("angstrom", LengthUnit::Angstrom),
            ("ang", LengthUnit::Angstrom),
        ] {
            let long = Args::try_parse_from(["rhf_sim", "--units", value]).unwrap();
            assert_eq!(long.units, expected, "--units {value}");
            let short = Args::try_parse_from(["rhf_sim", "-u", value]).unwrap();
            assert_eq!(short.units, expected, "-u {value}");
        }
        assert!(Args::try_parse_from(["rhf_sim", "--units", "nanometer"]).is_err());
    }
}
