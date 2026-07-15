use anyhow::{Result, anyhow};
use clap::Parser;
use std::fs::File;
use std::io::Write;

#[derive(Parser, Debug)]
#[command(author, about)]
struct Args {
    /// Atomic numbers or symbols (e.g. H 6 8 C)
    #[arg(short, long, num_args = 1..)]
    elements: Vec<String>,

    /// Basis set name (e.g. STO-3G, def2-SVP, cc-pV*Z)
    #[arg(short, long)]
    basis: String,

    /// Output file
    #[arg(short, long)]
    output: String,
}

fn normalize_basis(name: &str) -> String {
    name.to_lowercase().replace('*', "_st_")
}

fn atomic_number(element: &str) -> Result<u8> {
    if let Ok(num) = element.parse::<u8>() {
        return Ok(num);
    }

    let symbol = element.to_ascii_lowercase();

    let number = match symbol.as_str() {
        "H" | "h" => 1,
        "He" | "he" => 2,
        "Li" | "li" => 3,
        "Be" | "be" => 4,
        "B" | "b" => 5,
        "C" | "c" => 6,
        "N" | "n" => 7,
        "O" | "o" => 8,
        "F" | "f" => 9,
        "Ne" | "ne" => 10,
        _ => return Err(anyhow!("Unknown element: {element}")),
    };

    Ok(number)
}

fn main() -> Result<()> {
    let args = Args::parse();

    let basis = normalize_basis(&args.basis);

    let elements: Vec<String> = args
        .elements
        .iter()
        .map(|e| atomic_number(e).map(|n| n.to_string()))
        .collect::<Result<Vec<_>>>()?;

    let elements = elements.join(",");

    let url = format!(
        "http://www.basissetexchange.org/api/basis/{}/format/nwchem/?version=1&elements={}",
        basis, elements
    );

    println!("Requesting: {url}");

    let response = reqwest::blocking::get(&url)?;

    if !response.status().is_success() {
        return Err(anyhow!("Request failed with status {}", response.status()));
    }

    let body = response.text()?;

    let mut file = File::create(&args.output)?;
    file.write_all(body.as_bytes())?;

    println!("Written basis set to {}", args.output);

    Ok(())
}
