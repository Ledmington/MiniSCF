#![forbid(unsafe_code)]

use clap::Parser;
use std::{
    fs::File,
    io::{self, BufWriter, Write},
};
use xyz::{XYZFile, normalize_xyz, read_xyz};

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

fn write_xyz(path: &String, content: &XYZFile) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);

    writeln!(writer, "{}", content.atoms.len())?;
    writeln!(writer, "{}", content.comment)?;

    for a in &content.atoms {
        writeln!(
            writer,
            "{:<2} {:>16.8} {:>16.8} {:>16.8}",
            a.element.symbol, a.position.x, a.position.y, a.position.z
        )?;
    }

    Ok(())
}

fn run(args: &Args) -> Result<(), String> {
    let mut xyz_file = read_xyz(&args.input)
        .map_err(|e| format!("Could not read input file '{}': {e}", args.input))?;

    if xyz_file.atoms.is_empty() {
        return Err(format!("Input file '{}' contains no atoms.", args.input));
    }

    normalize_xyz(&mut xyz_file.atoms);

    write_xyz(&args.output, &xyz_file)
        .map_err(|e| format!("Could not write output file '{}': {e}.", args.output))
}

fn main() {
    let args = Args::parse();

    if let Err(e) = run(&args) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
