#![forbid(unsafe_code)]

mod atom;
mod basis;
mod cube_writer;
mod integrals;
mod point;
mod sim;

use crate::{
    atom::Atom,
    basis::{AngularMomentum, BasisSet, PrimitiveGaussian, Shell},
    cube_writer::dump_all_molecular_orbitals,
    point::Point,
    sim::{OptimizationParameters, run_rhf_simulation},
};
use clap::Parser;
use ndarray::Array2;
use simple_logger::SimpleLogger;
use std::collections::HashMap;
use std::{
    fs::File,
    io::{BufRead, BufReader},
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

fn read_xyz(path: &str) -> Result<Vec<Atom>, String> {
    let beginning = Instant::now();
    log::info!("Started reading input system from file '{path}'");

    let file = File::open(path).map_err(|e| format!("failed to open file: {}", e))?;

    let mut lines = BufReader::new(file).lines();

    let natoms: usize = lines
        .next()
        .ok_or("missing atom count")?
        .map_err(|e| e.to_string())?
        .trim()
        .parse()
        .map_err(|e| format!("invalid atom count: {}", e))?;

    // comment line
    lines
        .next()
        .ok_or("missing comment line")?
        .map_err(|e| e.to_string())?;

    let mut atoms = Vec::with_capacity(natoms);

    for _ in 0..natoms {
        let line = lines
            .next()
            .ok_or("unexpected end of file")?
            .map_err(|e| e.to_string())?;

        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() != 4 {
            return Err(format!("invalid atom line: {}", line));
        }

        let symbol = parts[0].to_string();

        let x: f64 = parts[1]
            .parse()
            .map_err(|_| format!("invalid x coordinate: {}", parts[1]))?;

        let y: f64 = parts[2]
            .parse()
            .map_err(|_| format!("invalid y coordinate: {}", parts[2]))?;

        let z: f64 = parts[3]
            .parse()
            .map_err(|_| format!("invalid z coordinate: {}", parts[3]))?;

        atoms.push(Atom {
            charge: atomic_number(&symbol)?,
            symbol,
            position: Point { x, y, z },
        });
    }

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Completed reading input system from file '{path}' in {elapsed:?}");
    }

    Ok(atoms)
}

struct ShellTemplate {
    angular: AngularMomentum::S,
    primitives: vec![
        (71.6168370, 0.15432897),
        (13.0450960, 0.53532814),
        (3.5305122, 0.44463454),
    ],
}

pub(crate) type BasisLibrary = HashMap<String, Vec<ShellTemplate>>;

pub(crate) fn parse_nwchem_basis(text: &str) -> BasisLibrary {
    let mut library = HashMap::new();

    let mut current_element: Option<String> = None;
    let mut current_shell: Option<ShellTemplate> = None;

    for line in text.lines() {
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') || line.starts_with("BASIS") {
            continue;
        }

        if line == "END" {
            break;
        }

        let fields: Vec<_> = line.split_whitespace().collect();

        // Shell header
        if fields.len() == 2 {
            if let (Some(element), Some(shell)) = (current_element.take(), current_shell.take()) {
                library.entry(element).or_default().push(shell);
            }

            current_element = Some(fields[0].to_string());

            let angular = match fields[1] {
                "S" => AngularMomentum::S,
                "P" => AngularMomentum::P,
                _ => continue,
            };

            current_shell = Some(ShellTemplate {
                angular,
                primitives: Vec::new(),
            });

            continue;
        }

        // Primitive
        if fields.len() >= 2 {
            let exponent = fields[0].parse::<f64>().unwrap();
            let coeff = fields[1].parse::<f64>().unwrap();

            current_shell
                .as_mut()
                .unwrap()
                .primitives
                .push((exponent, coeff));
        }
    }

    if let (Some(element), Some(shell)) = (current_element.take(), current_shell.take()) {
        library.entry(element).or_default().push(shell);
    }

    library
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

    // Prepare the STO-3G basis
    let mut shells = Vec::new();
    for atom in &atoms {
        let primitives = vec![
            PrimitiveGaussian::new(0.1543289673, 0.3425250914e1, atom.position, (0, 0, 0)),
            PrimitiveGaussian::new(0.5353281423, 0.6239137298, atom.position, (0, 0, 0)),
            PrimitiveGaussian::new(0.4446345422, 0.1688554040, atom.position, (0, 0, 0)),
        ];
        shells.push(Shell {
            center: atom.position,
            angular: AngularMomentum::S,
            primitives,
        });
    }
    let sto_3g = BasisSet::new(shells);

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
