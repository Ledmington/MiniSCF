use crate::point::Point;

#[derive(Clone)]
pub(crate) struct Atom {
    pub(crate) symbol: String,
    pub(crate) position: Point,
    pub(crate) charge: u8,
}
