#![forbid(unsafe_code)]

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Element {
    pub symbol: &'static str,
    pub number: u8,
}

pub const HYDROGEN: Element = Element {
    symbol: "H",
    number: 1,
};
pub const CARBON: Element = Element {
    symbol: "C",
    number: 6,
};

pub fn from_symbol(symbol: String) -> Element {
    match symbol.to_ascii_lowercase().as_str() {
        "h" => HYDROGEN,
        "c" => CARBON,
        _ => panic!("Unknown element with symbol '{symbol}'"),
    }
}
