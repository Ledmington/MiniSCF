use element::Element;
use scf_core::Atom;
use std::{collections::HashMap, fs, time::Instant};

use crate::basis::{AngularMomentum, BasisSet, PrimitiveGaussian, Shell};

#[derive(PartialEq, Debug)]
pub(crate) struct ShellTemplate {
    angular: AngularMomentum,
    primitives: Vec<(f64, f64)>,
}

pub(crate) type BasisLibrary = HashMap<Element, Vec<ShellTemplate>>;

pub(crate) fn parse_nwchem_basis(path: &str) -> Result<BasisLibrary, String> {
    let beginning = Instant::now();
    log::info!("Started reading basis set from file '{path}'");

    let text = fs::read_to_string(path).map_err(|e| format!("failed to open file: {}", e))?;

    let result = parse_nwchem_basis_text(&text);

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Completed reading basis set from file '{path}' in {elapsed:?}");
    }

    Ok(result)
}

fn parse_nwchem_basis_text(text: &str) -> BasisLibrary {
    let mut library: HashMap<Element, Vec<ShellTemplate>> = HashMap::new();

    let mut current_element: Option<String> = None;
    let mut current_shell: Option<ShellTemplate> = None;
    let mut current_shell_p: Option<ShellTemplate> = None; // for SP shells

    for line in text.lines() {
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("BASIS") {
            continue;
        }

        if trimmed == "END" {
            break;
        }

        let fields: Vec<_> = trimmed.split_whitespace().collect();

        // Shell header
        if fields.len() == 2 && fields.iter().all(|s| s.parse::<f64>().is_err()) {
            if let Some(symbol) = current_element.take() {
                let element = element::from_symbol(symbol);

                if let Some(shell) = current_shell.take() {
                    library.entry(element.clone()).or_default().push(shell);
                }

                if let Some(shell_p) = current_shell_p.take() {
                    library.entry(element).or_default().push(shell_p);
                }
            }

            current_element = Some(fields[0].to_string());

            match fields[1] {
                "S" => {
                    current_shell = Some(ShellTemplate {
                        angular: AngularMomentum::S,
                        primitives: Vec::new(),
                    });
                }

                "P" => {
                    current_shell = Some(ShellTemplate {
                        angular: AngularMomentum::P,
                        primitives: Vec::new(),
                    });
                }

                "SP" => {
                    current_shell = Some(ShellTemplate {
                        angular: AngularMomentum::S,
                        primitives: Vec::new(),
                    });

                    current_shell_p = Some(ShellTemplate {
                        angular: AngularMomentum::P,
                        primitives: Vec::new(),
                    });
                }

                _ => panic!("Unknown orbital '{}'", fields[1]),
            }

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

            // SP: third column is the P contraction coefficient
            if fields.len() >= 3
                && let Some(ref mut shell_p) = current_shell_p
            {
                let coeff_p = fields[2].parse::<f64>().unwrap();
                shell_p.primitives.push((exponent, coeff_p));
            }
        }
    }

    // Final flush
    if let Some(symbol) = current_element.take() {
        let element = element::from_symbol(symbol);

        if let Some(shell) = current_shell.take() {
            library.entry(element.clone()).or_default().push(shell);
        }

        if let Some(shell_p) = current_shell_p.take() {
            library.entry(element).or_default().push(shell_p);
        }
    }

    library
}

pub(crate) fn build_basis(atoms: &[Atom], basis_library: &BasisLibrary) -> BasisSet {
    let mut shells: Vec<(Shell, (u8, u8, u8))> = Vec::new();

    for atom in atoms {
        let templates = basis_library.get(&atom.element).unwrap_or_else(|| {
            panic!(
                "No basis functions found for element '{}'",
                atom.element.symbol
            )
        });

        for template in templates {
            match template.angular {
                AngularMomentum::S => {
                    let primitives = template
                        .primitives
                        .iter()
                        .map(|&(exponent, coeff)| {
                            PrimitiveGaussian::new(coeff, exponent, atom.position)
                        })
                        .collect();

                    shells.push((
                        Shell {
                            center: atom.position,
                            primitives,
                        },
                        (0, 0, 0),
                    ));
                }

                AngularMomentum::P => {
                    let primitives: Vec<PrimitiveGaussian> = template
                        .primitives
                        .iter()
                        .map(|&(exponent, coeff)| {
                            PrimitiveGaussian::new(coeff, exponent, atom.position)
                        })
                        .collect();

                    // px
                    shells.push((
                        Shell {
                            center: atom.position,
                            primitives: primitives.clone(),
                        },
                        (1, 0, 0),
                    ));
                    // py
                    shells.push((
                        Shell {
                            center: atom.position,
                            primitives: primitives.clone(),
                        },
                        (0, 1, 0),
                    ));
                    // pz
                    shells.push((
                        Shell {
                            center: atom.position,
                            primitives,
                        },
                        (0, 0, 1),
                    ));
                }
            }
        }
    }

    BasisSet::new(shells)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sto_3g_hydrogen() {
        let text = "BASIS \"ao basis\" SPHERICAL PRINT
#BASIS SET: (3s) -> [1s]
H    S
      0.3425250914E+01       0.1543289673E+00
      0.6239137298E+00       0.5353281423E+00
      0.1688554040E+00       0.4446345422E+00
END
";

        let basis = parse_nwchem_basis_text(text);
        let mut expected = HashMap::new();
        expected.insert(
            element::HYDROGEN,
            vec![ShellTemplate {
                angular: AngularMomentum::S,
                primitives: vec![
                    (3.425250914, 0.1543289673),
                    (0.6239137298, 0.5353281423),
                    (0.168855404, 0.4446345422),
                ],
            }],
        );
        assert_eq!(expected, basis);
    }

    #[test]
    fn sto_3g_hydrogen_carbon() {
        let text = "BASIS \"ao basis\" SPHERICAL PRINT
#BASIS SET: (3s) -> [1s]
H    S
      0.3425250914E+01       0.1543289673E+00
      0.6239137298E+00       0.5353281423E+00
      0.1688554040E+00       0.4446345422E+00
#BASIS SET: (6s,3p) -> [2s,1p]
C    S
      0.7161683735E+02       0.1543289673E+00
      0.1304509632E+02       0.5353281423E+00
      0.3530512160E+01       0.4446345422E+00
C    SP
      0.2941249355E+01      -0.9996722919E-01       0.1559162750E+00
      0.6834830964E+00       0.3995128261E+00       0.6076837186E+00
      0.2222899159E+00       0.7001154689E+00       0.3919573931E+00
END
";

        let basis = parse_nwchem_basis_text(text);
        let mut expected = HashMap::new();
        expected.insert(
            element::HYDROGEN,
            vec![ShellTemplate {
                angular: AngularMomentum::S,
                primitives: vec![
                    (3.425250914, 0.1543289673),
                    (0.6239137298, 0.5353281423),
                    (0.168855404, 0.4446345422),
                ],
            }],
        );
        expected.insert(
            element::CARBON,
            vec![
                ShellTemplate {
                    angular: AngularMomentum::S,
                    primitives: vec![
                        (71.61683735, 0.1543289673),
                        (13.04509632, 0.5353281423),
                        (3.530512160, 0.4446345422),
                    ],
                },
                ShellTemplate {
                    angular: AngularMomentum::S,
                    primitives: vec![
                        (2.941249355, -0.09996722919),
                        (0.6834830964, 0.3995128261),
                        (0.2222899159, 0.7001154689),
                    ],
                },
                ShellTemplate {
                    angular: AngularMomentum::P,
                    primitives: vec![
                        (2.941249355, 0.1559162750),
                        (0.6834830964, 0.6076837186),
                        (0.2222899159, 0.3919573931),
                    ],
                },
            ],
        );
        assert_eq!(expected, basis);
    }
}
