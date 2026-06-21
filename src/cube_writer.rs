use std::fs::File;
use std::io::{BufWriter, Write};

use crate::Atom;
use crate::point::Point;

#[derive(Clone)]
pub struct Grid {
    pub origin: Point,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: Point,
    pub dy: Point,
    pub dz: Point,
}

pub struct CubeWriter {
    pub atoms: Vec<Atom>,
    pub grid: Grid,
}

impl CubeWriter {
    pub fn new(atoms: Vec<Atom>, grid: Grid) -> Self {
        Self { atoms, grid }
    }

    pub fn write(&self, path: &str, values: &[f64]) -> std::io::Result<()> {
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
            self.grid.nx, self.grid.dx.x, self.grid.dx.y, self.grid.dx.z
        )?;

        writeln!(
            f,
            "{:5} {:10.6} {:10.6} {:10.6}",
            self.grid.ny, self.grid.dy.x, self.grid.dy.y, self.grid.dy.z
        )?;

        writeln!(
            f,
            "{:5} {:10.6} {:10.6} {:10.6}",
            self.grid.nz, self.grid.dz.x, self.grid.dz.y, self.grid.dz.z
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
