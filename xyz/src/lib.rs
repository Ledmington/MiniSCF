#![forbid(unsafe_code)]

use scf_core::{Atom, Point, atomic_number};
use std::{
    fs::File,
    io::{BufRead, BufReader},
    time::Instant,
};

pub struct XYZFile {
    pub comment: String,
    pub atoms: Vec<Atom>,
}

pub fn read_xyz(path: &str) -> Result<XYZFile, String> {
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
    let comment = lines
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
            position: Point::new(x, y, z),
        });
    }

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Completed reading input system from file '{path}' in {elapsed:?}");
    }

    Ok(XYZFile { comment, atoms })
}

// TODO: rewrite using BLAS?
pub fn normalize_xyz(atoms: &mut [Atom]) {
    let n = atoms.len();

    if n >= 1 {
        // First atom becomes the origin
        for i in 1..n {
            atoms[i].position = atoms[i].position.sub(&atoms[0].position);
        }
        atoms[0].position.set_x(0.0);
        atoms[0].position.set_y(0.0);
        atoms[0].position.set_z(0.0);

        if n >= 2 {
            // Rotate first around the z axis, then around the y axis, so that the second atom lies on the x axis
            let p = (atoms[1].position.x().powi(2) + atoms[1].position.y().powi(2)).sqrt();
            let phi = f64::atan2(atoms[1].position.y(), atoms[1].position.x());
            let cos_phi = phi.cos();
            let sin_phi = phi.sin();
            for a in atoms.iter_mut().take(n).skip(1) {
                let x = a.position.x();
                let y = a.position.y();
                a.position.set_x(x * cos_phi + y * sin_phi);
                a.position.set_y(-x * sin_phi + y * cos_phi);
            }
            let psi = f64::atan2(atoms[1].position.z(), p);
            let cos_psi = psi.cos();
            let sin_psi = psi.sin();
            for a in atoms.iter_mut().take(n).skip(1) {
                let x = a.position.x();
                let z = a.position.z();
                a.position.set_x(x * cos_psi + z * sin_psi);
                a.position.set_z(-x * sin_psi + z * cos_psi);
            }

            if n >= 3 {
                // Rotate around x axis in order to have third atom on the xy plane
                let theta = -f64::atan2(atoms[2].position.z(), atoms[2].position.y());
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();
                for a in atoms.iter_mut().take(n).skip(2) {
                    let y = a.position.y();
                    let z = a.position.z();
                    a.position.set_y(y * cos_theta - z * sin_theta);
                    a.position.set_z(y * sin_theta + z * cos_theta);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rand::RngExt;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(2)]
    #[case(3)]
    #[case(4)]
    #[case(5)]
    #[case(10)]
    fn xyz_normalization(#[case] n: usize) {
        let mut rng = rand::rng();
        let points = (0..n)
            .map(|_| {
                Point::new(
                    rng.random_range(-10.0..10.0),
                    rng.random_range(-10.0..10.0),
                    rng.random_range(-10.0..10.0),
                )
            })
            .collect::<Vec<Point>>();
        let mut atoms = points
            .iter()
            .map(|p| Atom {
                symbol: "C".to_string(),
                position: *p,
                charge: 6,
            })
            .collect::<Vec<Atom>>();
        let mut original_distances: HashMap<(usize, usize), f64> = HashMap::new();
        for i in 0..n {
            for j in (i + 1)..n {
                original_distances.insert((i, j), points[i].distance(&points[j]));
            }
        }

        normalize_xyz(&mut atoms);

        assert_eq!(n, atoms.len());
        if n >= 1 {
            assert!((atoms[0].position.x() - 0.0).abs() < 1e-12);
            assert!((atoms[0].position.y() - 0.0).abs() < 1e-12);
            assert!((atoms[0].position.z() - 0.0).abs() < 1e-12);
            if n >= 2 {
                assert!((atoms[1].position.y() - 0.0).abs() < 1e-12);
                assert!((atoms[1].position.z() - 0.0).abs() < 1e-12);
                if n >= 3 {
                    assert!((atoms[2].position.z() - 0.0).abs() < 1e-12);
                }
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                assert!(
                    (atoms[i].position.distance(&atoms[j].position)
                        - *original_distances.get(&(i, j)).unwrap())
                    .abs()
                        < 1e-12
                );
            }
        }
    }
}
