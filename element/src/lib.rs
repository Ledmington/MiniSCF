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
pub const SODIUM: Element = Element {
    symbol: "Na",
    number: 11,
};
pub const MAGNESIUM: Element = Element {
    symbol: "Mg",
    number: 12,
};
pub const ALUMINIUM: Element = Element {
    symbol: "Al",
    number: 13,
};
pub const SILICON: Element = Element {
    symbol: "Si",
    number: 14,
};
pub const PHOSPHORUS: Element = Element {
    symbol: "P",
    number: 15,
};
pub const SULFUR: Element = Element {
    symbol: "S",
    number: 16,
};
pub const CHLORINE: Element = Element {
    symbol: "Cl",
    number: 17,
};
pub const ARGON: Element = Element {
    symbol: "Ar",
    number: 18,
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
        "na" => SODIUM,
        "mg" => MAGNESIUM,
        "al" => ALUMINIUM,
        "si" => SILICON,
        "p" => PHOSPHORUS,
        "s" => SULFUR,
        "cl" => CHLORINE,
        "ar" => ARGON,
        _ => panic!("Unknown element with symbol '{symbol}'"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Element; 18] = [
        HYDROGEN, HELIUM, LITHIUM, BERYLLIUM, BORON, CARBON, NITROGEN, OXYGEN, FLUORINE, NEON,
        SODIUM, MAGNESIUM, ALUMINIUM, SILICON, PHOSPHORUS, SULFUR, CHLORINE, ARGON,
    ];

    #[test]
    fn atomic_numbers_are_sequential() {
        for (i, element) in ALL.iter().enumerate() {
            assert_eq!(usize::from(element.number), i + 1, "{}", element.symbol);
        }
    }

    #[test]
    fn from_symbol_roundtrips() {
        for element in ALL {
            assert_eq!(from_symbol(element.symbol.to_string()), element);
            assert_eq!(from_symbol(element.symbol.to_ascii_uppercase()), element);
        }
    }

    #[test]
    #[should_panic(expected = "Unknown element")]
    fn from_symbol_rejects_unknown() {
        from_symbol("K".to_string());
    }
}
