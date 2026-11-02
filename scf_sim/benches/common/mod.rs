//! Shared fixture: water (experimental geometry, as in `examples/water.xyz`) with the
//! 6-31G* basis set, which has d shells on oxygen (25 Cartesian functions). The basis set
//! is embedded because `.gbs` files are not tracked by git.

use scf_core::point::Point;
use scf_core::{Atom, units::ANGSTROM_TO_BOHR};
use scf_sim::bench_support::Molecule;

const WATER_6_31GSTAR: &str = "BASIS \"ao basis\" CARTESIAN PRINT
H    S
      0.1873113696E+02       0.3349460434E-01
      0.2825394365E+01       0.2347269535E+00
      0.6401216923E+00       0.8137573261E+00
H    S
      0.1612777588E+00       1.0000000
O    S
      0.5484671660E+04       0.1831074430E-02
      0.8252349460E+03       0.1395017220E-01
      0.1880469580E+03       0.6844507810E-01
      0.5296450000E+02       0.2327143360E+00
      0.1689757040E+02       0.4701928980E+00
      0.5799635340E+01       0.3585208530E+00
O    SP
      0.1553961625E+02      -0.1107775495E+00       0.7087426823E-01
      0.3599933586E+01      -0.1480262627E+00       0.3397528391E+00
      0.1013761750E+01       0.1130767015E+01       0.7271585773E+00
O    SP
      0.2700058226E+00       0.1000000000E+01       0.1000000000E+01
O    D
      0.8000000000E+00       1.0000000
END
";

pub fn water() -> Molecule {
    let atom = |element: element::Element, x: f64, y: f64| Atom {
        charge: element.number,
        element,
        position: Point::new(x * ANGSTROM_TO_BOHR, y * ANGSTROM_TO_BOHR, 0.0),
    };
    Molecule::new(
        vec![
            atom(element::OXYGEN, 0.0, 0.0),
            atom(element::HYDROGEN, 0.9572, 0.0),
            atom(element::HYDROGEN, -0.23998721, 0.92662721),
        ],
        WATER_6_31GSTAR,
    )
}
