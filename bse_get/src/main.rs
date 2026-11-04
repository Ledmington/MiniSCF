#![forbid(unsafe_code)]

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

fn atomic_number(symbol_or_number: &str) -> Result<u8> {
    if let Ok(num) = symbol_or_number.parse::<usize>() {
        return num
            .checked_sub(1)
            .and_then(|index| element::ELEMENTS.get(index))
            .map(|element| element.number)
            .ok_or_else(|| {
                anyhow!(
                    "Invalid atomic number {num}: expected 1 to {}",
                    element::ELEMENTS.len()
                )
            });
    }

    element::from_symbol(symbol_or_number)
        .map(|element| element.number)
        .ok_or_else(|| anyhow!("Unknown element symbol '{symbol_or_number}'"))
}

fn run(args: &Args) -> Result<()> {
    let basis = normalize_basis(&args.basis);

    let elements: Vec<String> = args
        .elements
        .iter()
        .map(|e| atomic_number(e).map(|n| n.to_string()))
        .collect::<Result<Vec<_>>>()?;

    let url = format!(
        "http://www.basissetexchange.org/api/basis/{}/format/nwchem/?version=1&elements={}",
        basis,
        elements.join(",")
    );

    println!("Requesting: {url}");

    let response = reqwest::blocking::get(&url)?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Request failed with status '{}', is the basis set name correct?",
            response.status()
        ));
    }

    let body = response.text()?;

    let mut file = File::create(&args.output)?;
    file.write_all(body.as_bytes())?;

    println!("Written basis set to {}", args.output);

    Ok(())
}

fn main() {
    let args = Args::parse();

    if let Err(e) = run(&args) {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_number_accepts_symbols_and_numbers() {
        assert_eq!(atomic_number("H").unwrap(), 1);
        assert_eq!(atomic_number("cl").unwrap(), 17);
        assert_eq!(atomic_number("8").unwrap(), 8);
        assert_eq!(atomic_number("118").unwrap(), 118);
    }

    #[test]
    fn atomic_number_rejects_invalid_input() {
        for input in ["0", "119", "300", "Xx", ""] {
            assert!(atomic_number(input).is_err(), "{input}");
        }
    }
}
