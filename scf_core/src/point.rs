#![forbid(unsafe_code)]

#[derive(Clone, Copy, Debug)]
pub struct Point {
    // TODO: make private and use getters
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Point { x, y, z }
    }

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

    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    pub fn distance(&self, p: &Point) -> f64 {
        self.sub(p).norm()
    }

    pub fn coordinates(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}
