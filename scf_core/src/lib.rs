#[derive(Clone, Copy, Debug)]
pub struct Point {
    // TODO: make private and use getters
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point {
    pub fn sub(&self, p: &Point) -> Point {
        Point {
            x: self.x - p.x,
            y: self.y - p.y,
            z: self.z - p.z,
        }
    }

    pub fn norm_squared(&self) -> f64 {
        self.x.powi(2) + self.y.powi(2) + self.z.powi(2)
    }
}

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
