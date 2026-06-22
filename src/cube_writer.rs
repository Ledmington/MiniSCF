use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use ndarray::Array2;

use crate::Atom;
use crate::basis::BasisSet;
use crate::point::Point;

#[derive(Clone)]
pub(crate) struct Grid {
    pub(crate) origin: Point,
    pub(crate) nx: usize,
    pub(crate) ny: usize,
    pub(crate) nz: usize,
    pub(crate) dx: f64,
    pub(crate) dy: f64,
    pub(crate) dz: f64,
}

pub(crate) struct CubeWriter {
    pub(crate) atoms: Vec<Atom>,
    pub(crate) grid: Grid,
}

impl CubeWriter {
    pub(crate) fn new(atoms: Vec<Atom>, grid: Grid) -> Self {
        Self { atoms, grid }
    }

    pub(crate) fn write(&self, path: &str, values: &[f64]) -> std::io::Result<()> {
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

fn build_cube_values(grid: &Grid, mo_index: usize, basis: &BasisSet, c: &Array2<f64>) -> Vec<f64> {
    let mut values = Vec::new();

    for iz in 0..grid.nz {
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let r = Point {
                    x: grid.origin.x + ix as f64 * grid.dx,
                    y: grid.origin.y + iy as f64 * grid.dy,
                    z: grid.origin.z + iz as f64 * grid.dz,
                };

                let psi = basis.compute(mo_index, &r, c);
                values.push(psi);
            }
        }
    }

    values
}

pub(crate) fn dump_molecular_orbital(
    atoms: &[Atom],
    basis: &BasisSet,
    mo_index: usize,
    c: &Array2<f64>,
    filename: String,
) -> std::io::Result<()> {
    let beginning = Instant::now();
    log::info!("Starting dumping orbitals to '{filename}'");

    let grid = Grid {
        origin: Point {
            x: -3.0,
            y: -3.0,
            z: -3.0,
        },

        nx: 60,
        ny: 60,
        nz: 60,

        dx: 0.1,
        dy: 0.1,
        dz: 0.1,
    };

    let cube = CubeWriter::new(atoms.to_vec(), grid.clone());
    let values = build_cube_values(&grid, mo_index, basis, c);
    cube.write(&filename, &values)?;

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Completed dumping orbitals in {elapsed:?}");
    }

    Ok(())
}
