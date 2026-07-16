#![forbid(unsafe_code)]

pub mod point;

use element::Element;

use crate::point::Point;

#[derive(Clone)]
pub struct Atom {
    pub element: Element,
    pub position: Point,
    pub charge: u8,
}
