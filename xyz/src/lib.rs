use scf_core::{Atom, Point, atomic_number};
use std::{
    fs::File,
    io::{BufRead, BufReader},
    time::Instant,
};

pub fn read_xyz(path: &str) -> Result<Vec<Atom>, String> {
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
