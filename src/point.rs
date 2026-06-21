#[derive(Clone, Copy)]
pub(crate) struct Point {
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) z: f64,
}

impl Point {
    pub(crate) fn sub(&self, p: &Point) -> Point {
        Point {
            x: self.x - p.x,
            y: self.y - p.y,
            z: self.z - p.z,
        }
    }

    pub(crate) fn norm_squared(&self) -> f64 {
        self.x.powi(2) + self.y.powi(2) + self.z.powi(2)
    }
}
