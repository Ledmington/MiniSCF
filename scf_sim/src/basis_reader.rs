use element::Element;
use scf_core::{Atom, point::Point};
use std::{collections::HashMap, fs, sync::Arc, time::Instant};

use crate::basis::{AngularMomentum, BasisFunction, BasisSet, PrimitiveGaussian, Shell};
use crate::integrals;

#[derive(PartialEq, Debug)]
pub struct ShellTemplate {
    angular: AngularMomentum,
    primitives: Vec<(f64, f64)>,
}

pub(crate) type BasisLibrary = HashMap<Element, Vec<ShellTemplate>>;

pub fn parse_nwchem_basis(path: &str) -> Result<BasisLibrary, String> {
    let beginning = Instant::now();
    log::info!("Started reading basis set from file '{path}'");

    let text = fs::read_to_string(path).map_err(|e| format!("failed to open file: {e}"))?;

    let result = parse_nwchem_basis_text(&text);

    {
        let elapsed = Instant::now() - beginning;
        log::info!("Completed reading basis set from file '{path}' in {elapsed:?}");
    }

    Ok(result)
}

/// The kind of shell block announced by a shell header such as `C    SP`.
#[derive(Clone)]
enum ShellKind {
    /// A combined SP block: the first coefficient column is the s contraction, the second
    /// one the p contraction, sharing the same exponents.
    Sp,
    /// A block with a single angular momentum. With more than one coefficient column it is
    /// a general contraction (as in the Dunning cc-pVXZ sets): each column is a separate
    /// contracted shell sharing the same exponents.
    Single(AngularMomentum),
}

/// The shell block being read: its element, kind and the shells built so far (one per
/// coefficient column, created when the first primitive line is read).
struct PendingBlock {
    symbol: String,
    kind: ShellKind,
    shells: Vec<ShellTemplate>,
}

impl PendingBlock {
    fn add_primitive(&mut self, exponent: f64, coefficients: &[f64]) {
        if self.shells.is_empty() {
            self.shells = match &self.kind {
                ShellKind::Sp => vec![AngularMomentum::S, AngularMomentum::P],
                ShellKind::Single(angular) => vec![angular.clone(); coefficients.len()],
            }
            .into_iter()
            .map(|angular| ShellTemplate {
                angular,
                primitives: Vec::new(),
            })
            .collect();
        }

        assert_eq!(
            coefficients.len(),
            self.shells.len(),
            "Expected {} contraction coefficients for exponent {exponent} of element '{}' but found {}",
            self.shells.len(),
            self.symbol,
            coefficients.len()
        );

        for (shell, &coefficient) in self.shells.iter_mut().zip(coefficients) {
            // In general contractions most columns are zero for most exponents: such
            // primitives do not contribute and would only slow down the integrals.
            if coefficient != 0.0 {
                shell.primitives.push((exponent, coefficient));
            }
        }
    }

    fn flush_into(self, library: &mut BasisLibrary) {
        let element = element::from_symbol(self.symbol);
        library.entry(element).or_default().extend(self.shells);
    }
}

fn parse_nwchem_basis_text(text: &str) -> BasisLibrary {
    let mut library: HashMap<Element, Vec<ShellTemplate>> = HashMap::new();

    let mut current_block: Option<PendingBlock> = None;
    let mut spherical = false;
    let mut has_high_angular_momentum = false;

    for line in text.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("BASIS") {
            spherical = trimmed.split_whitespace().any(|s| s == "SPHERICAL");
            continue;
        }

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if trimmed == "END" {
            break;
        }

        let fields: Vec<_> = trimmed.split_whitespace().collect();

        // Shell header
        if fields.len() == 2 && fields.iter().all(|s| s.parse::<f64>().is_err()) {
            if let Some(block) = current_block.take() {
                block.flush_into(&mut library);
            }

            let kind = match fields[1] {
                "S" => ShellKind::Single(AngularMomentum::S),
                "P" => ShellKind::Single(AngularMomentum::P),
                "D" => ShellKind::Single(AngularMomentum::D),
                "F" => ShellKind::Single(AngularMomentum::F),
                "SP" => ShellKind::Sp,
                _ => panic!("Unknown orbital '{}'", fields[1]),
            };
            if matches!(fields[1], "D" | "F") {
                has_high_angular_momentum = true;
            }

            current_block = Some(PendingBlock {
                symbol: fields[0].to_string(),
                kind,
                shells: Vec::new(),
            });

            continue;
        }

        // Primitive: an exponent followed by one contraction coefficient per shell
        if fields.len() >= 2 {
            let exponent = fields[0].parse::<f64>().unwrap();
            let coefficients: Vec<f64> = fields[1..]
                .iter()
                .map(|c| c.parse::<f64>().unwrap())
                .collect();

            current_block
                .as_mut()
                .expect("Found primitive coefficients before any shell header")
                .add_primitive(exponent, &coefficients);
        }
    }

    // Final flush
    if let Some(block) = current_block.take() {
        block.flush_into(&mut library);
    }

    if spherical && has_high_angular_momentum {
        log::warn!(
            "The basis set is declared SPHERICAL, but d and f shells are always built as Cartesian functions (6 d, 10 f): energies will differ from reference calculations that use 5 spherical d and 7 spherical f functions."
        );
    }

    library
}

pub fn build_basis(atoms: &[Atom], basis_library: &BasisLibrary) -> BasisSet {
    let mut shells: Vec<(Shell, (u8, u8, u8))> = Vec::new();

    for atom in atoms {
        let templates = basis_library.get(&atom.element).unwrap_or_else(|| {
            panic!(
                "No basis functions found for element '{}'",
                atom.element.symbol
            )
        });

        for template in templates {
            let raw_primitives: Vec<PrimitiveGaussian> = template
                .primitives
                .iter()
                .map(|&(exponent, coeff)| PrimitiveGaussian::new(coeff, exponent, atom.position))
                .collect();
            let primitives =
                normalize_contraction(raw_primitives, atom.position, &template.angular);

            // One basis function per Cartesian component: s, px/py/pz, the 6 d functions or
            // the 10 f functions.
            for component in template.angular.cartesian_components() {
                shells.push((
                    Shell {
                        center: atom.position,
                        primitives: primitives.clone(),
                    },
                    component,
                ));
            }
        }
    }

    BasisSet::new(shells)
}

/// Contracted shells whose self-overlap is within this tolerance of 1 are used as written.
const CONTRACTION_NORMALIZATION_TOLERANCE: f64 = 1e-10;

/// Rescales the contraction coefficients of a shell so that its contracted functions are
/// normalized.
///
/// Basis set files print the coefficients with a limited number of digits (6 for the
/// Dunning cc-pVXZ sets on the Basis Set Exchange), so contractions are only normalized to
/// about 1e-6. Since the primitives share the same center, the self-overlap of a contracted
/// Cartesian function only depends on the total angular momentum, not on its components:
/// a single factor normalizes all the functions of the shell. Shells that are already
/// normalized within [`CONTRACTION_NORMALIZATION_TOLERANCE`] are left untouched.
fn normalize_contraction(
    primitives: Vec<PrimitiveGaussian>,
    center: Point,
    angular: &AngularMomentum,
) -> Vec<PrimitiveGaussian> {
    let probe = BasisFunction {
        shell: Arc::new(Shell { center, primitives }),
        angular_momentum: angular.cartesian_components()[0],
    };
    let self_overlap = integrals::overlap(&probe, &probe);
    let Shell {
        primitives: raw, ..
    } = Arc::unwrap_or_clone(probe.shell);

    if (self_overlap - 1.0).abs() <= CONTRACTION_NORMALIZATION_TOLERANCE {
        return raw;
    }

    let factor = 1.0 / self_overlap.sqrt();
    raw.iter()
        .map(|p| p.with_scaled_coefficient(factor))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rand::{RngExt, SeedableRng, rngs::ChaCha8Rng};
    use scf_core::point::Point;

    use crate::basis::BasisFunction;

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
        let seed = rand::rng().random();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let center = Point::new(
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
        );

        let library = parse_nwchem_basis_text(text);
        let basis = build_basis(
            &[Atom {
                element: element::HYDROGEN,
                position: center,
                charge: 1,
            }],
            &library,
        );
        let shell = Shell {
            center,
            primitives: vec![
                PrimitiveGaussian::new(0.1543289673, 3.425250914, center),
                PrimitiveGaussian::new(0.5353281423, 0.6239137298, center),
                PrimitiveGaussian::new(0.4446345422, 0.168855404, center),
            ],
        };
        let expected = BasisSet {
            functions: vec![BasisFunction {
                shell: Arc::new(shell),
                angular_momentum: (0, 0, 0),
            }],
        };

        assert_eq!(
            expected, basis,
            "Expected parsed basis set to be equal to {expected:?} but was {basis:?} (seed: {seed})."
        );
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

        let seed = rand::rng().random();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let center = Point::new(
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
        );

        let library = parse_nwchem_basis_text(text);
        let basis = build_basis(
            &[Atom {
                element: element::CARBON,
                position: center,
                charge: 6,
            }],
            &library,
        );

        let expected = BasisSet {
            functions: vec![
                // 1s core
                BasisFunction {
                    shell: Arc::new(Shell {
                        center,
                        primitives: vec![
                            PrimitiveGaussian::new(0.1543289673, 71.61683735, center),
                            PrimitiveGaussian::new(0.5353281423, 13.04509632, center),
                            PrimitiveGaussian::new(0.4446345422, 3.530512160, center),
                        ],
                    }),
                    angular_momentum: (0, 0, 0),
                },
                // 2s valence
                BasisFunction {
                    shell: Arc::new(Shell {
                        center,
                        primitives: vec![
                            PrimitiveGaussian::new(-0.09996722919, 2.941249355, center),
                            PrimitiveGaussian::new(0.3995128261, 0.6834830964, center),
                            PrimitiveGaussian::new(0.7001154689, 0.2222899159, center),
                        ],
                    }),
                    angular_momentum: (0, 0, 0),
                },
                // 2px
                BasisFunction {
                    shell: Arc::new(Shell {
                        center,
                        primitives: vec![
                            PrimitiveGaussian::new(0.1559162750, 2.941249355, center),
                            PrimitiveGaussian::new(0.6076837186, 0.6834830964, center),
                            PrimitiveGaussian::new(0.3919573931, 0.2222899159, center),
                        ],
                    }),
                    angular_momentum: (1, 0, 0),
                },
                // 2py
                BasisFunction {
                    shell: Arc::new(Shell {
                        center,
                        primitives: vec![
                            PrimitiveGaussian::new(0.1559162750, 2.941249355, center),
                            PrimitiveGaussian::new(0.6076837186, 0.6834830964, center),
                            PrimitiveGaussian::new(0.3919573931, 0.2222899159, center),
                        ],
                    }),
                    angular_momentum: (0, 1, 0),
                },
                // 2pz
                BasisFunction {
                    shell: Arc::new(Shell {
                        center,
                        primitives: vec![
                            PrimitiveGaussian::new(0.1559162750, 2.941249355, center),
                            PrimitiveGaussian::new(0.6076837186, 0.6834830964, center),
                            PrimitiveGaussian::new(0.3919573931, 0.2222899159, center),
                        ],
                    }),
                    angular_momentum: (0, 0, 1),
                },
            ],
        };
        assert_eq!(
            expected, basis,
            "Expected parsed basis set to be equal to {expected:?} but was {basis:?} (seed: {seed})."
        );
    }

    #[test]
    fn six_31g_star_carbon_d_shell() {
        let text = "BASIS \"ao basis\" CARTESIAN PRINT
#BASIS SET: (10s,4p,1d) -> [3s,2p,1d]
C    S
      0.3047524880E+04       0.1834737132E-02
      0.4573695180E+03       0.1403732281E-01
      0.1039486850E+03       0.6884262226E-01
      0.2921015530E+02       0.2321844432E+00
      0.9286662960E+01       0.4679413484E+00
      0.3163926960E+01       0.3623119853E+00
C    SP
      0.7868272350E+01      -0.1193324198E+00       0.6899906659E-01
      0.1881288540E+01      -0.1608541517E+00       0.3164239610E+00
      0.5442492580E+00       0.1143456438E+01       0.7443082909E+00
C    SP
      0.1687144782E+00       0.1000000000E+01       0.1000000000E+01
C    D
      0.8000000000E+00       1.0000000
END
";
        let center = Point::new(0.5, -1.0, 2.0);
        let library = parse_nwchem_basis_text(text);
        let basis = build_basis(
            &[Atom {
                element: element::CARBON,
                position: center,
                charge: 6,
            }],
            &library,
        );

        // [3s,2p,1d] with Cartesian d: 3 + 2 * 3 + 6 = 15 functions.
        assert_eq!(basis.functions.len(), 15);

        let d_functions = &basis.functions[9..];
        let expected_components = [
            (2, 0, 0),
            (1, 1, 0),
            (1, 0, 1),
            (0, 2, 0),
            (0, 1, 1),
            (0, 0, 2),
        ];
        for (bf, expected) in d_functions.iter().zip(expected_components) {
            assert_eq!(bf.angular_momentum, expected);
            assert_eq!(
                *bf.shell,
                Shell {
                    center,
                    primitives: vec![PrimitiveGaussian::new(1.0, 0.8, center)],
                }
            );
        }
    }

    /// The cc-pVTZ basis for hydrogen and an f shell, as written by the Basis Set Exchange:
    /// Dunning sets use general contractions, with one coefficient column per contracted
    /// shell.
    const CC_PVTZ_SAMPLE: &str = "BASIS \"ao basis\" SPHERICAL PRINT
#BASIS SET: (5s,2p,1d) -> [3s,2p,1d]
H    S
      3.387000E+01           0.000000E+00           6.068000E-03           0.000000E+00
      5.095000E+00           0.000000E+00           4.530800E-02           0.000000E+00
      1.159000E+00           0.000000E+00           2.028220E-01           0.000000E+00
      3.258000E-01           1.000000E+00           5.039030E-01           0.000000E+00
      1.027000E-01           0.000000E+00           3.834210E-01           1.000000E+00
H    P
      1.407000E+00           1.000000E+00           0.000000E+00
      3.880000E-01           0.000000E+00           1.000000E+00
H    D
      1.057000E+00           1.0000000
O    F
      1.428000E+00           1.0000000
END
";

    #[test]
    fn general_contractions_are_split_into_shells() {
        let library = parse_nwchem_basis_text(CC_PVTZ_SAMPLE);
        let hydrogen = &library[&element::HYDROGEN];

        // [3s,2p,1d]: one shell per coefficient column, without the zero coefficients.
        assert_eq!(
            *hydrogen,
            vec![
                ShellTemplate {
                    angular: AngularMomentum::S,
                    primitives: vec![(3.258e-1, 1.0)],
                },
                ShellTemplate {
                    angular: AngularMomentum::S,
                    primitives: vec![
                        (3.387e1, 6.068e-3),
                        (5.095, 4.5308e-2),
                        (1.159, 2.02822e-1),
                        (3.258e-1, 5.03903e-1),
                        (1.027e-1, 3.83421e-1),
                    ],
                },
                ShellTemplate {
                    angular: AngularMomentum::S,
                    primitives: vec![(1.027e-1, 1.0)],
                },
                ShellTemplate {
                    angular: AngularMomentum::P,
                    primitives: vec![(1.407, 1.0)],
                },
                ShellTemplate {
                    angular: AngularMomentum::P,
                    primitives: vec![(3.88e-1, 1.0)],
                },
                ShellTemplate {
                    angular: AngularMomentum::D,
                    primitives: vec![(1.057, 1.0)],
                },
            ]
        );
    }

    #[test]
    fn f_shell() {
        let library = parse_nwchem_basis_text(CC_PVTZ_SAMPLE);
        assert_eq!(
            library[&element::OXYGEN],
            vec![ShellTemplate {
                angular: AngularMomentum::F,
                primitives: vec![(1.428, 1.0)],
            }]
        );

        let center = Point::new(-0.3, 0.2, 1.1);
        let basis = build_basis(
            &[
                Atom {
                    element: element::HYDROGEN,
                    position: center,
                    charge: 1,
                },
                Atom {
                    element: element::OXYGEN,
                    position: center,
                    charge: 8,
                },
            ],
            &library,
        );

        // Hydrogen: 3 s + 2 * 3 p + 6 d = 15 functions; oxygen: 10 Cartesian f functions.
        assert_eq!(basis.functions.len(), 25);
        let f_components: Vec<_> = basis.functions[15..]
            .iter()
            .map(|bf| bf.angular_momentum)
            .collect();
        assert_eq!(f_components, AngularMomentum::F.cartesian_components());
    }

    #[test]
    fn contractions_are_normalized() {
        let seed = rand::rng().random();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let center = Point::new(
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
            rng.random_range(-10.0..10.0),
        );

        for angular in [
            AngularMomentum::S,
            AngularMomentum::P,
            AngularMomentum::D,
            AngularMomentum::F,
        ] {
            let primitives: Vec<_> = (0..4)
                .map(|_| {
                    PrimitiveGaussian::new(
                        rng.random_range(0.1..2.0),
                        rng.random_range(0.05..50.0),
                        center,
                    )
                })
                .collect();
            let normalized = normalize_contraction(primitives, center, &angular);
            let shell = Arc::new(Shell {
                center,
                primitives: normalized,
            });
            for angular_momentum in angular.cartesian_components() {
                let bf = BasisFunction {
                    shell: Arc::clone(&shell),
                    angular_momentum,
                };
                let actual = integrals::overlap(&bf, &bf);
                assert!(
                    (actual - 1.0).abs() < 1e-12,
                    "Expected {angular_momentum:?} to be normalized but its self-overlap was {actual} (seed: {seed})."
                );
            }
        }
    }

    #[test]
    fn normalized_contractions_are_left_untouched() {
        let center = Point::new(0.1, 0.2, 0.3);
        let primitives = vec![PrimitiveGaussian::new(1.0, 1.057, center)];
        assert_eq!(
            normalize_contraction(primitives.clone(), center, &AngularMomentum::D),
            primitives
        );
    }

    #[test]
    #[should_panic(expected = "Expected 2 contraction coefficients")]
    fn inconsistent_number_of_coefficients() {
        parse_nwchem_basis_text(
            "BASIS \"ao basis\" CARTESIAN PRINT
H    S
      3.387000E+01           0.1           0.2
      5.095000E+00           0.3
END
",
        );
    }
}
