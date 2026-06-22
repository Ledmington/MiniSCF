use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use ndarray::Array2;

use crate::Atom;
use crate::basis::BasisSet;
use crate::point::Point;

#[derive(Clone)]
struct Grid {
    origin: Point,
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
}

struct CubeWriter {
    atoms: Vec<Atom>,
    grid: Grid,
}

impl CubeWriter {
    fn new(atoms: Vec<Atom>, grid: Grid) -> Self {
        Self { atoms, grid }
    }

    fn write(&self, path: &str, values: &[f64]) -> std::io::Result<()> {
        let mut f = BufWriter::new(File::create(path)?);

        // -------------------------
        // 1. Header (2 comment lines)
        // -------------------------
        writeln!(f, "MO Cube generated from SCF")?;
        writeln!(f, "Orbitals / density")?;

        // -------------------------
        // 2. Atom count + origin
        // -------------------------
        writeln!(
            f,
            "{:5} {:10.6} {:10.6} {:10.6}",
            self.atoms.len(),
            self.grid.origin.x,
            self.grid.origin.y,
            self.grid.origin.z
        )?;

        // -------------------------
        // 3. Grid vectors
        // -------------------------
        writeln!(
            f,
            "{:5} {:10.6} {:10.6} {:10.6}",
            self.grid.nx, self.grid.dx, 0.0, 0.0
        )?;

        writeln!(
            f,
            "{:5} {:10.6} {:10.6} {:10.6}",
            self.grid.ny, 0.0, self.grid.dy, 0.0
        )?;

        writeln!(
            f,
            "{:5} {:10.6} {:10.6} {:10.6}",
            self.grid.nz, 0.0, 0.0, self.grid.dz
        )?;

        // -------------------------
        // 4. Atoms
        // -------------------------
        for atom in &self.atoms {
            writeln!(
                f,
                "{:5} {:10.6} {:10.6} {:10.6} {:10.6}",
                atom.z, atom.z, atom.position.x, atom.position.y, atom.position.z
            )?;
        }

        // -------------------------
        // 5. Volume data
        // -------------------------
        let mut count = 0;

        for v in values {
            write!(f, "{v:13.5e}")?;
            count += 1;

            if count % 6 == 0 {
                writeln!(f)?;
            }
        }

        writeln!(f)?;
        Ok(())
    }
}

fn evaluate_orbital(grid: &Grid, basis: &BasisSet, c: &Array2<f64>, mo_index: usize) -> Vec<f64> {
    let mut values = Vec::new();

    for ix in 0..grid.nx {
        for iy in 0..grid.ny {
            for iz in 0..grid.nz {
                let r = Point {
                    x: grid.origin.x + (ix as f64) * grid.dx,
                    y: grid.origin.y + (iy as f64) * grid.dy,
                    z: grid.origin.z + (iz as f64) * grid.dz,
                };

                let psi = basis.compute(mo_index, &r, c);
                values.push(psi);
            }
        }
    }

    values
}

fn compute_grid(atoms: &[Atom]) -> Grid {
    let beginning = Instant::now();

    log::info!("Pre-computing the orbital grid");

    const PADDING: f64 = 3.0; // bohr

    let min_x = atoms
        .iter()
        .map(|a| a.position.x)
        .fold(f64::INFINITY, f64::min);
    let max_x = atoms
        .iter()
        .map(|a| a.position.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = atoms
        .iter()
        .map(|a| a.position.y)
        .fold(f64::INFINITY, f64::min);
    let max_y = atoms
        .iter()
        .map(|a| a.position.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_z = atoms
        .iter()
        .map(|a| a.position.z)
        .fold(f64::INFINITY, f64::min);
    let max_z = atoms
        .iter()
        .map(|a| a.position.z)
        .fold(f64::NEG_INFINITY, f64::max);

    let origin = Point {
        x: min_x - PADDING,
        y: min_y - PADDING,
        z: min_z - PADDING,
    };

    let dx = 0.1;
    let dy = 0.1;
    let dz = 0.1;

    let nx = ((max_x - min_x + 2.0 * PADDING) / dx).ceil() as usize;
    let ny = ((max_y - min_y + 2.0 * PADDING) / dy).ceil() as usize;
    let nz = ((max_z - min_z + 2.0 * PADDING) / dz).ceil() as usize;

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Completed pre-computing the orbital grid in {elapsed:?}");
    }

    Grid {
        origin,
        nx,
        ny,
        nz,
        dx,
        dy,
        dz,
    }
}

fn dump_molecular_orbital(
    writer: &CubeWriter,
    basis: &BasisSet,
    c: &Array2<f64>,
    mo_index: usize,
    grid: &Grid,
    filename: String,
) -> std::io::Result<()> {
    let beginning = Instant::now();
    log::info!("Starting writing orbital {mo_index} to '{filename}'");

    let values = evaluate_orbital(grid, basis, c, mo_index);
    writer.write(&filename, &values)?;

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Completed writing orbital {mo_index} in {elapsed:?}");
    }

    Ok(())
}

pub(crate) fn dump_all_molecular_orbitals(
    atoms: &[Atom],
    basis: &BasisSet,
    c: &Array2<f64>,
    output_filename_prefix: String,
) -> std::io::Result<()> {
    // Pre-compute the grid
    let grid = compute_grid(atoms);

    let writer = CubeWriter::new(atoms.to_vec(), grid.clone());

    let beginning = Instant::now();
    log::info!("Starting writing {} orbitals", c.ncols());

    for mo_index in 0..c.ncols() {
        dump_molecular_orbital(
            &writer,
            basis,
            c,
            mo_index,
            &grid,
            format!("{}{}.cube", output_filename_prefix, mo_index),
        )?;
    }

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Completed writing {} orbitals in {elapsed:?}", c.ncols());
    }

    Ok(())
}
