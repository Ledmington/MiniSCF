//! Length units. Everything inside the codebase works in bohr (atomic units).

/// The Bohr radius in angstroms (CODATA 2018).
pub const BOHR_IN_ANGSTROM: f64 = 0.529_177_210_903;

/// Multiply a length in angstroms by this factor to express it in bohr.
pub const ANGSTROM_TO_BOHR: f64 = 1.0 / BOHR_IN_ANGSTROM;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angstrom_to_bohr() {
        assert!((ANGSTROM_TO_BOHR - 1.889_726_124_626).abs() < 1e-12);
        assert!((ANGSTROM_TO_BOHR * BOHR_IN_ANGSTROM - 1.0).abs() < 1e-15);
    }
}
