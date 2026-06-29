#![forbid(unsafe_code)]

use clap::Parser;
use scf_core::Atom;
use std::{
    fs::File,
    io::{self, BufWriter, Write},
};
use xyz::{XYZFile, read_xyz};

#[derive(Parser, Debug)]
#[command(author, version, about = "Normalize an XYZ molecule")]
struct Args {
    /// Input XYZ file
    #[arg(short, long)]
    input: String,

    /// Output XYZ file
    #[arg(short, long)]
    output: String,
}

fn normalize_xyz(atoms: &mut [Atom]) {
    let n = atoms.len();

    // First atom becomes the origin
    for i in 1..n {
        atoms[i].position.x -= atoms[0].position.x;
        atoms[i].position.y -= atoms[0].position.y;
        atoms[i].position.z -= atoms[0].position.z;
    }
    atoms[0].position.x = 0.0;
    atoms[0].position.y = 0.0;
    atoms[0].position.z = 0.0;

    if n >= 2 {
        // Rotate first around the z axis, then around the y axis, so that the second atom lies on the x axis
        let p = (atoms[1].position.x.powi(2) + atoms[1].position.y.powi(2)).sqrt();
        let phi = f64::atan2(atoms[1].position.y, atoms[1].position.x);
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();
        for i in 1..n {
            let x = atoms[i].position.x;
            let y = atoms[i].position.y;
            atoms[i].position.x = x * cos_phi + y * sin_phi;
            atoms[i].position.y = -x * sin_phi + y * cos_phi;
        }
        let psi = f64::atan2(atoms[1].position.z, p);
        let cos_psi = psi.cos();
        let sin_psi = psi.sin();
        for i in 1..n {
            let x = atoms[i].position.x;
            let z = atoms[i].position.z;
            atoms[i].position.x = x * cos_psi + z * sin_psi;
            atoms[i].position.z = -x * sin_psi + z * cos_psi;
        }

        if n >= 3 {
            // Rotate around x axis in order to have third atom on the xy plane
            let theta = -f64::atan2(atoms[2].position.z, atoms[2].position.y);
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();
            for i in 2..n {
                let y = atoms[i].position.y;
                let z = atoms[i].position.z;
                atoms[i].position.y = y * cos_theta - z * sin_theta;
                atoms[i].position.z = y * sin_theta + z * cos_theta;
            }
        }
    }
}

fn write_xyz(path: &String, content: &XYZFile) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);

    writeln!(writer, "{}", content.atoms.len())?;
    writeln!(writer, "{}", content.comment)?;

    for a in &content.atoms {
        writeln!(
            writer,
            "{:<2} {:>16.8} {:>16.8} {:>16.8}",
            a.symbol, a.position.x, a.position.y, a.position.z
        )?;
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut xyz_file = read_xyz(&args.input)?;

    if xyz_file.atoms.is_empty() {
        return Err("XYZ file contains no atoms".into());
    }

    normalize_xyz(&mut xyz_file.atoms);

    write_xyz(&args.output, &xyz_file)?;

    Ok(())
}
