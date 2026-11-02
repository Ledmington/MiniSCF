use crate::basis::{BasisEvaluator, BasisSet};
use crate::integrals::{CoulombPotentialEvaluator, POTENTIAL_BATCH};
use crate::sim::density_matrix;
use ndarray::{Array2, s};
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
        let formatter = ValueFormatter::new();
        let mut buffer = Vec::with_capacity(values.len() * 13 + values.len() / 6 + 1);
        for (count, v) in values.iter().enumerate() {
            formatter.format(*v, &mut buffer);
            if (count + 1) % 6 == 0 {
                buffer.push(b'\n');
            }
        }
        buffer.push(b'\n');
        f.write_all(&buffer)?;
        f.flush()
    }
}

/// Formats values exactly as `format!("{v:13.5e}")`, but much faster: the six significant
/// digits are computed in floating point, falling back to the standard formatter whenever
/// the rounding could be ambiguous.
struct ValueFormatter {
    /// `10^k` (correctly rounded) at index `k + POWER_OFFSET`.
    powers_of_ten: Vec<f64>,
}

const POWER_OFFSET: i32 = 310;

impl ValueFormatter {
    fn new() -> Self {
        let powers_of_ten = (-POWER_OFFSET..=POWER_OFFSET)
            .map(|k| format!("1e{k}").parse().expect("valid float literal"))
            .collect();
        Self { powers_of_ten }
    }

    fn power_of_ten(&self, k: i32) -> f64 {
        self.powers_of_ten[(k + POWER_OFFSET) as usize]
    }

    /// Appends `v` formatted as `{v:13.5e}` to `out`.
    fn format(&self, v: f64, out: &mut Vec<u8>) {
        let magnitude = v.abs();
        // Only normal numbers well inside the f64 range take the fast path.
        if (1e-300..1e300).contains(&magnitude) {
            // A first guess from the binary exponent (78913 / 2^18 ~ log10(2)), corrected until
            // the mantissa has six digits before the decimal point.
            let binary_exponent = ((magnitude.to_bits() >> 52) as i32) - 1023;
            let mut exponent = (binary_exponent * 78913) >> 18;
            let mut scaled = magnitude * self.power_of_ten(5 - exponent);
            while scaled >= 1_000_000.0 {
                exponent += 1;
                scaled = magnitude * self.power_of_ten(5 - exponent);
            }
            while scaled < 100_000.0 {
                exponent -= 1;
                scaled = magnitude * self.power_of_ten(5 - exponent);
            }
            // Truncating conversions, unlike `f64::round`/`floor`, are not library calls.
            let whole = scaled as u32;
            let fraction = scaled - f64::from(whole);
            let rounded = whole + u32::from(fraction > 0.5);
            // `scaled` is within a few ulps (~1e-10) of the exact value, so unless it is this
            // close to a rounding tie, it rounds the same way as the exact decimal expansion.
            if (fraction - 0.5).abs() > 1e-6 && (100_000..1_000_000).contains(&rounded) {
                // At most "-d.ddddde-ddd", 13 bytes: fill a right-aligned field backwards.
                let mut field = [b' '; 13];
                let mut position = 13;
                let mut push = |byte: u8| {
                    position -= 1;
                    field[position] = byte;
                };
                let mut exponent_digits = exponent.unsigned_abs();
                loop {
                    push(b'0' + (exponent_digits % 10) as u8);
                    exponent_digits /= 10;
                    if exponent_digits == 0 {
                        break;
                    }
                }
                if exponent < 0 {
                    push(b'-');
                }
                push(b'e');
                let mut digits = rounded;
                for _ in 0..5 {
                    push(b'0' + (digits % 10) as u8);
                    digits /= 10;
                }
                push(b'.');
                push(b'0' + digits as u8);
                if v < 0.0 {
                    push(b'-');
                }
                out.extend_from_slice(&field);
                return;
            }
        }
        out.extend_from_slice(format!("{v:13.5e}").as_bytes());
    }
}

/// The values of the first `num_orbitals` molecular orbitals (columns of `c`) at every
/// point of `grid`, one vector per orbital in cube order. The basis functions are
/// evaluated once per point, then turned into orbitals one x-slab at a time with a single
/// matrix product.
fn evaluate_orbitals(
    grid: &Grid,
    basis: &BasisSet,
    c: &Array2<f64>,
    num_orbitals: usize,
) -> Vec<Vec<f64>> {
    let evaluator = BasisEvaluator::new(&basis.functions);
    let coefficients = c.slice(s![.., ..num_orbitals]);
    let axis = |origin: f64, n: usize, step: f64| -> Vec<f64> {
        (0..n).map(|i| origin + (i as f64) * step).collect()
    };
    let xs = axis(grid.origin.x, grid.nx, grid.dx);
    let ys = axis(grid.origin.y, grid.ny, grid.dy);
    let zs = axis(grid.origin.z, grid.nz, grid.dz);
    let mut orbitals = vec![Vec::with_capacity(grid.nx * grid.ny * grid.nz); num_orbitals];

    evaluator.evaluate_grid(&xs, &ys, &zs, |_, phi| {
        let psi = phi.dot(&coefficients);
        for (values, column) in orbitals.iter_mut().zip(psi.columns()) {
            values.extend(column.iter());
        }
    });

    orbitals
}

/// The electron density 2 * sum_i psi_i^2 from the values of the occupied orbitals.
fn density_from_orbitals(occupied: &[Vec<f64>], num_points: usize) -> Vec<f64> {
    let mut density = vec![0.0; num_points];
    for psi in occupied {
        for (rho, value) in density.iter_mut().zip(psi) {
            *rho += 2.0 * value * value;
        }
    }
    density
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
    let mut electronic = CoulombPotentialEvaluator::new(&basis.functions, density);
    let mut values = Vec::with_capacity(grid.nx * grid.ny * grid.nz);
    let mut v_electronic = [0.0; POTENTIAL_BATCH];
    for ix in 0..grid.nx {
        for iy in 0..grid.ny {
            for start in (0..grid.nz).step_by(POTENTIAL_BATCH) {
                let points: Vec<Point> = (start..grid.nz.min(start + POTENTIAL_BATCH))
                    .map(|iz| {
                        Point::new(
                            grid.origin.x + (ix as f64) * grid.dx,
                            grid.origin.y + (iy as f64) * grid.dy,
                            grid.origin.z + (iz as f64) * grid.dz,
                        )
                    })
                    .collect();
                let batch = &mut v_electronic[..points.len()];
                electronic.evaluate_batch(&points, batch);
                for (r, electronic_value) in points.iter().zip(batch.iter()) {
                    let v_nuclear: f64 = atoms
                        .iter()
                        .map(|a| {
                            (a.charge as f64) / r.distance(&a.position).max(MIN_NUCLEAR_DISTANCE)
                        })
                        .sum();
                    values.push(v_nuclear - electronic_value);
                }
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

/// Runs `f`, logging a start message and an elapsed-time completion message around it.
/// The completion message is skipped if `f` fails, matching an early return from `?`.
fn timed<T>(
    label: impl std::fmt::Display,
    f: impl FnOnce() -> std::io::Result<T>,
) -> std::io::Result<T> {
    log::info!("Starting {label}");
    let beginning = Instant::now();

    let result = f()?;

    let elapsed = Instant::now() - beginning;
    log::info!("Completed {label} in {elapsed:?}");
    Ok(result)
}

fn dump_all_molecular_orbitals(
    writer: &CubeWriter,
    orbitals: &[Vec<f64>],
    output_filename_prefix: &str,
) -> std::io::Result<()> {
    timed(format!("writing {} orbitals", orbitals.len()), || {
        for (mo_index, values) in orbitals.iter().enumerate() {
            let filename = orbital_filename(output_filename_prefix, mo_index, orbitals.len());
            timed(
                format!("writing orbital {mo_index} to '{filename}'"),
                || writer.write(&filename, values),
            )?;
        }
        Ok(())
    })
}

/// Dumps the total electron density rho(r) = sum over occupied MOs of 2 * psi_i(r)^2
/// into a single cube file, combining every occupied orbital into one field.
fn dump_electron_density(
    writer: &CubeWriter,
    occupied: &[Vec<f64>],
    num_points: usize,
    output_filename: &str,
) -> std::io::Result<()> {
    timed(
        format!("writing electron density to '{output_filename}'"),
        || {
            let values = density_from_orbitals(occupied, num_points);
            writer.write(output_filename, &values)
        },
    )
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

    timed(
        format!("writing electrostatic potential to '{output_filename}'"),
        || {
            let values = evaluate_electrostatic_potential(grid, basis, &writer.atoms, &density);
            writer.write(output_filename, &values)
        },
    )
}

/// Number of digits of the largest orbital index, so that orbital file names sort
/// in order (`mo_00` ... `mo_64` for 65 orbitals).
fn orbital_index_width(num_orbitals: usize) -> usize {
    num_orbitals.saturating_sub(1).to_string().len()
}

fn orbital_filename(prefix: &str, mo_index: usize, num_orbitals: usize) -> String {
    let width = orbital_index_width(num_orbitals);
    format!("{prefix}_mo_{mo_index:0width$}.cube")
}

fn density_filename(prefix: &str) -> String {
    format!("{prefix}_density.cube")
}

fn esp_filename(prefix: &str) -> String {
    format!("{prefix}_esp.cube")
}

/// Names of all the files `dump` would write for the given format(s) and number of
/// molecular orbitals, so that callers can check for them before running a simulation.
pub fn output_filenames(formats: &[CubeFormat], prefix: &str, num_orbitals: usize) -> Vec<String> {
    let mut filenames = Vec::new();
    for format in formats {
        match format {
            CubeFormat::Orbitals => filenames
                .extend((0..num_orbitals).map(|i| orbital_filename(prefix, i, num_orbitals))),
            CubeFormat::Density => filenames.push(density_filename(prefix)),
            CubeFormat::Esp => filenames.push(esp_filename(prefix)),
        }
    }
    filenames
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
    let num_points = grid.nx * grid.ny * grid.nz;
    let n_electrons: usize = atoms.iter().map(|a| a.charge as usize).sum();
    let n_occ = basis.num_occupied_orbitals(n_electrons);

    // The orbital values on the grid are shared by the orbital and density cubes.
    let num_orbitals = if formats.contains(&CubeFormat::Orbitals) {
        c.ncols()
    } else if formats.contains(&CubeFormat::Density) {
        n_occ
    } else {
        0
    };
    let orbitals = if num_orbitals > 0 {
        timed(
            format!("evaluating {num_orbitals} orbitals on the grid"),
            || Ok(evaluate_orbitals(&grid, basis, c, num_orbitals)),
        )?
    } else {
        Vec::new()
    };

    for format in formats {
        match format {
            CubeFormat::Orbitals => {
                dump_all_molecular_orbitals(&writer, &orbitals, &output_filename_prefix)?
            }
            CubeFormat::Density => dump_electron_density(
                &writer,
                &orbitals[..n_occ],
                num_points,
                &density_filename(&output_filename_prefix),
            )?,
            CubeFormat::Esp => dump_electrostatic_potential(
                &writer,
                basis,
                c,
                &grid,
                &esp_filename(&output_filename_prefix),
            )?,
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_value_matches_standard_formatter() {
        let mut values = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            1.000005,
            1.000015,
            9.999995,
            9.9999949,
            99999.95,
            1e-5,
            1e-300,
            1e-310,
            1e300,
            123456.5,
            0.1,
            0.3,
            f64::INFINITY,
            f64::NAN,
            f64::MIN_POSITIVE,
        ];
        let mut state: u64 = 0x2545_f491_4f6c_dd1d;
        for _ in 0..200_000 {
            // xorshift64: arbitrary bit patterns cover every exponent and mantissa.
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            values.push(f64::from_bits(state));
            values.push(
                ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
                    * 10f64.powi((state % 40) as i32 - 20),
            );
        }
        let formatter = ValueFormatter::new();
        for v in values {
            let mut actual = Vec::new();
            formatter.format(v, &mut actual);
            assert_eq!(
                String::from_utf8(actual).unwrap(),
                format!("{v:13.5e}"),
                "{v:e}"
            );
        }
    }

    #[test]
    fn output_filenames_match_formats() {
        assert_eq!(
            vec![
                "mysim_mo_0.cube",
                "mysim_mo_1.cube",
                "mysim_density.cube",
                "mysim_esp.cube"
            ],
            output_filenames(
                &[CubeFormat::Orbitals, CubeFormat::Density, CubeFormat::Esp],
                "mysim",
                2
            )
        );
        assert_eq!(
            vec!["out/rhf_esp.cube"],
            output_filenames(&[CubeFormat::Esp], "out/rhf", 7)
        );
        assert!(output_filenames(&[], "rhf", 3).is_empty());
    }

    #[test]
    fn orbital_filenames_are_padded_to_the_largest_index() {
        let names = output_filenames(&[CubeFormat::Orbitals], "rhf", 65);
        assert_eq!("rhf_mo_00.cube", names[0]);
        assert_eq!("rhf_mo_09.cube", names[9]);
        assert_eq!("rhf_mo_64.cube", names[64]);
        assert_eq!(
            vec!["rhf_mo_0.cube"],
            output_filenames(&[CubeFormat::Orbitals], "rhf", 1)
        );
        assert_eq!(
            "rhf_mo_099.cube",
            output_filenames(&[CubeFormat::Orbitals], "rhf", 101)[99]
        );
        // Exactly 10 orbitals still fit in one digit (0..=9).
        assert_eq!(
            "rhf_mo_9.cube",
            output_filenames(&[CubeFormat::Orbitals], "rhf", 10)[9]
        );
    }
}
