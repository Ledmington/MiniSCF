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
pub const HELIUM: Element = Element {
    symbol: "He",
    number: 2,
};
pub const LITHIUM: Element = Element {
    symbol: "Li",
    number: 3,
};
pub const BERYLLIUM: Element = Element {
    symbol: "Be",
    number: 4,
};
pub const BORON: Element = Element {
    symbol: "B",
    number: 5,
};
pub const CARBON: Element = Element {
    symbol: "C",
    number: 6,
};
pub const NITROGEN: Element = Element {
    symbol: "N",
    number: 7,
};
pub const OXYGEN: Element = Element {
    symbol: "O",
    number: 8,
};
pub const FLUORINE: Element = Element {
    symbol: "F",
    number: 9,
};
pub const NEON: Element = Element {
    symbol: "Ne",
    number: 10,
};

pub fn from_symbol(symbol: String) -> Element {
    match symbol.to_ascii_lowercase().as_str() {
        "h" => HYDROGEN,
        "he" => HELIUM,
        "li" => LITHIUM,
        "be" => BERYLLIUM,
        "b" => BORON,
        "c" => CARBON,
        "n" => NITROGEN,
        "o" => OXYGEN,
        "f" => FLUORINE,
        "ne" => NEON,
        _ => panic!("Unknown element with symbol '{symbol}'"),
    }
}
