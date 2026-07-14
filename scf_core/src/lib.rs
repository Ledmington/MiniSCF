#![forbid(unsafe_code)]

pub mod point;

use crate::point::Point;

#[derive(Clone)]
pub struct Atom {
    // TODO: make private and use getters
    pub symbol: String,
    pub position: Point,
    pub charge: u8,
}

// TODO: move this into Atom
pub fn atomic_number(symbol: &str) -> Result<u8, String> {
    match symbol {
        "H" => Ok(1),
        "He" => Ok(2),
        "Li" => Ok(3),
        "Be" => Ok(4),
        "B" => Ok(5),
        "C" => Ok(6),
        "N" => Ok(7),
        "O" => Ok(8),
        "F" => Ok(9),
        "Ne" => Ok(10),
        "Na" => Ok(11),
        "Mg" => Ok(12),
        "Al" => Ok(13),
        "Si" => Ok(14),
        "P" => Ok(15),
        "S" => Ok(16),
        _ => Err(format!("unknown element: {}", symbol)),
    }
}
