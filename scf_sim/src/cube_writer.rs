use crate::basis::BasisSet;
use crate::sim::density_matrix;
use ndarray::Array2;
use scf_core::Atom;
use scf_core::point::Point;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

/// Which cube file format(s) `dump` should write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeFormat {
    /// One .cube file per molecular orbital.
    Orbitals,
    /// A single .cube file with the total electron density.
    Density,
    /// A single .cube file with the molecular electrostatic potential.
    Esp,
}

impl std::str::FromStr for CubeFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "orbitals" => Ok(CubeFormat::Orbitals),
            "density" => Ok(CubeFormat::Density),
            "esp" => Ok(CubeFormat::Esp),
            other => Err(format!(
                "unknown cube format '{other}' (expected 'orbitals', 'density' or 'esp')"
            )),
        }
    }
}

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
        writeln!(f, "MO Cube generated from MiniSCF")?;
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
                atom.charge, atom.charge, atom.position.x, atom.position.y, atom.position.z
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
                let r = Point::new(
                    grid.origin.x + (ix as f64) * grid.dx,
                    grid.origin.y + (iy as f64) * grid.dy,
                    grid.origin.z + (iz as f64) * grid.dz,
                );

                let psi = basis.evaluate_molecular_orbital(&r, c, mo_index);
                values.push(psi);
            }
        }
    }

    values
}

fn evaluate_density(grid: &Grid, basis: &BasisSet, c: &Array2<f64>, n_occ: usize) -> Vec<f64> {
    let mut values = Vec::new();

    for ix in 0..grid.nx {
        for iy in 0..grid.ny {
            for iz in 0..grid.nz {
                let r = Point::new(
                    grid.origin.x + (ix as f64) * grid.dx,
                    grid.origin.y + (iy as f64) * grid.dy,
                    grid.origin.z + (iz as f64) * grid.dz,
                );

                let rho: f64 = (0..n_occ)
                    .map(|mo_index| {
                        let psi = basis.evaluate_molecular_orbital(&r, c, mo_index);
                        2.0 * psi * psi
                    })
                    .sum();
                values.push(rho);
            }
        }
    }

    values
}

/// A grid point can land exactly on a nucleus (e.g. atoms placed at coordinates that are
/// multiples of the grid spacing), where the 1/r nuclear term is formally infinite. Clamp
/// the distance to keep the cube file finite; this is far below the grid spacing, so it
/// only affects voxels that coincide with a nucleus.
const MIN_NUCLEAR_DISTANCE: f64 = 1e-3;

/// The molecular electrostatic potential at `r`: the potential a unit positive test charge
/// would feel, i.e. the nuclear contribution sum_A Z_A/|r-R_A| minus the electronic
/// contribution sum_{mu,nu} P[mu,nu] * (mu nu | 1/|r-r'|).
fn evaluate_electrostatic_potential(
    grid: &Grid,
    basis: &BasisSet,
    atoms: &[Atom],
    density: &Array2<f64>,
) -> Vec<f64> {
    let mut values = Vec::new();

    for ix in 0..grid.nx {
        for iy in 0..grid.ny {
            for iz in 0..grid.nz {
                let r = Point::new(
                    grid.origin.x + (ix as f64) * grid.dx,
                    grid.origin.y + (iy as f64) * grid.dy,
                    grid.origin.z + (iz as f64) * grid.dz,
                );

                let v_nuclear: f64 = atoms
                    .iter()
                    .map(|a| (a.charge as f64) / r.distance(&a.position).max(MIN_NUCLEAR_DISTANCE))
                    .sum();
                let v_electronic = basis.electronic_potential(&r, density);

                values.push(v_nuclear - v_electronic);
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

    let origin = Point::new(min_x - PADDING, min_y - PADDING, min_z - PADDING);

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

fn dump_all_molecular_orbitals(
    writer: &CubeWriter,
    basis: &BasisSet,
    c: &Array2<f64>,
    grid: &Grid,
    output_filename_prefix: &str,
) -> std::io::Result<()> {
    let beginning = Instant::now();
    log::info!("Starting writing {} orbitals", c.ncols());

    for mo_index in 0..c.ncols() {
        dump_molecular_orbital(
            writer,
            basis,
            c,
            mo_index,
            grid,
            format!("{output_filename_prefix}{mo_index}.cube"),
        )?;
    }

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Completed writing {} orbitals in {elapsed:?}", c.ncols());
    }

    Ok(())
}

/// Dumps the total electron density rho(r) = sum over occupied MOs of 2 * psi_i(r)^2
/// into a single cube file, combining every occupied orbital into one field.
fn dump_electron_density(
    writer: &CubeWriter,
    basis: &BasisSet,
    c: &Array2<f64>,
    grid: &Grid,
    output_filename: &str,
) -> std::io::Result<()> {
    let n_electrons: usize = writer.atoms.iter().map(|a| a.charge as usize).sum();
    let n_occ = basis.num_occupied_orbitals(n_electrons);

    let beginning = Instant::now();
    log::info!("Starting writing electron density to '{output_filename}'");

    let values = evaluate_density(grid, basis, c, n_occ);
    writer.write(output_filename, &values)?;

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Completed writing electron density in {elapsed:?}");
    }

    Ok(())
}

/// Dumps the molecular electrostatic potential into a single cube file.
fn dump_electrostatic_potential(
    writer: &CubeWriter,
    basis: &BasisSet,
    c: &Array2<f64>,
    grid: &Grid,
    output_filename: &str,
) -> std::io::Result<()> {
    let n_electrons: usize = writer.atoms.iter().map(|a| a.charge as usize).sum();
    let n_occ = basis.num_occupied_orbitals(n_electrons);
    let density = density_matrix(c, n_occ);

    let beginning = Instant::now();
    log::info!("Starting writing electrostatic potential to '{output_filename}'");

    let values = evaluate_electrostatic_potential(grid, basis, &writer.atoms, &density);
    writer.write(output_filename, &values)?;

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Completed writing electrostatic potential in {elapsed:?}");
    }

    Ok(())
}

/// Writes the requested cube file format(s), computing the real-space grid once and
/// reusing it across all of them.
pub fn dump(
    atoms: &[Atom],
    basis: &BasisSet,
    c: &Array2<f64>,
    formats: &[CubeFormat],
    output_filename_prefix: String,
) -> std::io::Result<()> {
    let grid = compute_grid(atoms);
    let writer = CubeWriter::new(atoms.to_vec(), grid.clone());

    for format in formats {
        match format {
            CubeFormat::Orbitals => {
                dump_all_molecular_orbitals(&writer, basis, c, &grid, &output_filename_prefix)?
            }
            CubeFormat::Density => dump_electron_density(
                &writer,
                basis,
                c,
                &grid,
                &format!("{output_filename_prefix}density.cube"),
            )?,
            CubeFormat::Esp => dump_electrostatic_potential(
                &writer,
                basis,
                c,
                &grid,
                &format!("{output_filename_prefix}esp.cube"),
            )?,
        }
    }

    Ok(())
}
