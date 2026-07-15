use rug::Float;
use std::ops::{Mul, Sub};

#[derive(Clone, Debug)]
pub struct Point {
    x: Float,
    y: Float,
    z: Float,
}
impl Point {
    pub fn new(x: Float, y: Float, z: Float) -> Self {
        Point { x, y, z }
    }
    pub fn x(&self) -> &Float {
        &self.x
    }
    pub fn y(&self) -> &Float {
        &self.y
    }
    pub fn z(&self) -> &Float {
        &self.z
    }
    pub fn sub(&self, p: &Point) -> Point {
        Point::new(self.x.sub(&p.x), self.y.sub(&p.y), self.z.sub(&p.z))
    }
    pub fn norm_squared(&self) -> Float {
        self.x.mul(self.x) + self.y.mul(self.y) + self.z.mul(self.z)
    }
    pub fn norm(&self) -> Float {
        self.norm_squared().sqrt()
    }
    pub fn distance(&self, p: &Point) -> Float {
        self.sub(p).norm()
    }
    pub fn coordinates(&self) -> [Float; 3] {
        [self.x(), self.y(), self.z()]
    }
}
