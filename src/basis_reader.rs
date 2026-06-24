use std::{collections::HashMap, fs, time::Instant};

use crate::{
    atom::Atom,
    basis::{AngularMomentum, BasisSet, PrimitiveGaussian, Shell},
};

pub(crate) struct ShellTemplate {
    angular: AngularMomentum,
    primitives: Vec<(f64, f64)>,
}

pub(crate) type BasisLibrary = HashMap<String, Vec<ShellTemplate>>;

pub(crate) fn parse_nwchem_basis(path: &str) -> Result<BasisLibrary, String> {
    let beginning = Instant::now();
    log::info!("Started reading basis set from file '{path}'");

    let text = fs::read_to_string(path).map_err(|e| format!("failed to open file: {}", e))?;

    let mut library: HashMap<String, Vec<ShellTemplate>> = HashMap::new();

    let mut current_element: Option<String> = None;
    let mut current_shell: Option<ShellTemplate> = None;

    for line in text.lines() {
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') || line.starts_with("BASIS") {
            continue;
        }

        if line == "END" {
            break;
        }

        let fields: Vec<_> = line.split_whitespace().collect();

        // Shell header
        if fields.len() == 2 && fields.iter().all(|s| s.parse::<f64>().is_err()) {
            if let (Some(element), Some(shell)) = (current_element.take(), current_shell.take()) {
                library.entry(element).or_default().push(shell);
            }

            current_element = Some(fields[0].to_string());

            let angular = match fields[1] {
                "S" => AngularMomentum::S,
                "P" => AngularMomentum::P,
                _ => continue,
            };

            current_shell = Some(ShellTemplate {
                angular,
                primitives: Vec::new(),
            });

            continue;
        }

        // Primitive coefficients
        if fields.len() >= 2 {
            let exponent = fields[0].parse::<f64>().unwrap();
            let coeff = fields[1].parse::<f64>().unwrap();

            current_shell
                .as_mut()
                .unwrap()
                .primitives
                .push((exponent, coeff));
        }
    }

    if let (Some(element), Some(shell)) = (current_element.take(), current_shell.take()) {
        library.entry(element).or_default().push(shell);
    }

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Completed reading basis set from file '{path}' in {elapsed:?}");
    }

    Ok(library)
}

pub(crate) fn build_basis(atoms: &[Atom], basis_library: &BasisLibrary) -> BasisSet {
    let mut shells = Vec::new();

    for atom in atoms {
        let templates = basis_library
            .get(&atom.symbol)
            .unwrap_or_else(|| panic!("No basis functions found for element '{}'", atom.symbol));

        for template in templates {
            match template.angular {
                AngularMomentum::S => {
                    let primitives = template
                        .primitives
                        .iter()
                        .map(|&(exponent, coeff)| {
                            PrimitiveGaussian::new(coeff, exponent, atom.position, (0, 0, 0))
                        })
                        .collect();

                    shells.push(Shell {
                        center: atom.position,
                        angular: AngularMomentum::S,
                        primitives,
                    });
                }

                AngularMomentum::P => {
                    for powers in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] {
                        let primitives = template
                            .primitives
                            .iter()
                            .map(|&(exponent, coeff)| {
                                PrimitiveGaussian::new(coeff, exponent, atom.position, powers)
                            })
                            .collect();

                        shells.push(Shell {
                            center: atom.position,
                            angular: AngularMomentum::P,
                            primitives,
                        });
                    }
                }
            }
        }
    }

    BasisSet::new(shells)
}
