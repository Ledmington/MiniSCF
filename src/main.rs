#![forbid(unsafe_code)]

mod basis;
mod cube_writer;
mod sim;

use ndarray::Array2;

use crate::{
    basis::{BasisSet, Point},
    cube_writer::{CubeWriter, Grid},
    sim::run_rhf_simulation,
};

#[derive(Clone)]
struct Atom {
    z: f64,
    position: Point,
}

fn build_cube_values(grid: &Grid, mo_index: usize, basis: &BasisSet, c: &Array2<f64>) -> Vec<f64> {
    let mut values = Vec::new();

    for iz in 0..grid.nz {
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let r = Point {
                    x: grid.origin.x + ix as f64 * grid.dx.x,
                    y: grid.origin.y + iy as f64 * grid.dy.y,
                    z: grid.origin.z + iz as f64 * grid.dz.z,
                };

                let psi = basis.compute(mo_index, &r, c);
                values.push(psi);
            }
        }
    }

    values
}

fn dump_molecular_orbital(
    atoms: &[Atom],
    basis: &BasisSet,
    c: &Array2<f64>,
) -> std::io::Result<()> {
    let grid = Grid {
        origin: Point {
            x: -3.0,
            y: -3.0,
            z: -3.0,
        },

        nx: 60,
        ny: 60,
        nz: 60,

        dx: Point {
            x: 0.1,
            y: 0.0,
            z: 0.0,
        },
        dy: Point {
            x: 0.0,
            y: 0.1,
            z: 0.0,
        },
        dz: Point {
            x: 0.0,
            y: 0.0,
            z: 0.1,
        },
    };

    let cube = CubeWriter::new(atoms.to_vec(), grid.clone());
    let values = build_cube_values(&grid, 0, basis, c);
    cube.write("h2_mo0.cube", &values)?;
    Ok(())
}

fn main() -> std::io::Result<()> {
    const R: f64 = 1.4; // bohr

    // 2 Hydrogen atoms
    let atoms = vec![
        Atom {
            z: 1.0,
            position: Point {
                x: 0.0,
                y: 0.0,
                z: -R / 2.0,
            },
        },
        Atom {
            z: 1.0,
            position: Point {
                x: 0.0,
                y: 0.0,
                z: R / 2.0,
            },
        },
    ];

    // Prepare the STO-3G basis
    let sto_3g = BasisSet::new(
        &[0.15432897, 0.53532814, 0.44463454],
        &[3.42525091, 0.62391373, 0.16885540],
        &atoms.iter().map(|a| a.position).collect::<Vec<Point>>(),
    );

    let c = run_rhf_simulation(&atoms, &sto_3g);

    dump_molecular_orbital(&atoms, &sto_3g, &c)?;

    Ok(())
}
