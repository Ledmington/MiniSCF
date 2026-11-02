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
pub const POTASSIUM: Element = Element {
    symbol: "K",
    number: 19,
};
pub const CALCIUM: Element = Element {
    symbol: "Ca",
    number: 20,
};
pub const SCANDIUM: Element = Element {
    symbol: "Sc",
    number: 21,
};
pub const TITANIUM: Element = Element {
    symbol: "Ti",
    number: 22,
};
pub const VANADIUM: Element = Element {
    symbol: "V",
    number: 23,
};
pub const CHROMIUM: Element = Element {
    symbol: "Cr",
    number: 24,
};
pub const MANGANESE: Element = Element {
    symbol: "Mn",
    number: 25,
};
pub const IRON: Element = Element {
    symbol: "Fe",
    number: 26,
};
pub const COBALT: Element = Element {
    symbol: "Co",
    number: 27,
};
pub const NICKEL: Element = Element {
    symbol: "Ni",
    number: 28,
};
pub const COPPER: Element = Element {
    symbol: "Cu",
    number: 29,
};
pub const ZINC: Element = Element {
    symbol: "Zn",
    number: 30,
};
pub const GALLIUM: Element = Element {
    symbol: "Ga",
    number: 31,
};
pub const GERMANIUM: Element = Element {
    symbol: "Ge",
    number: 32,
};
pub const ARSENIC: Element = Element {
    symbol: "As",
    number: 33,
};
pub const SELENIUM: Element = Element {
    symbol: "Se",
    number: 34,
};
pub const BROMINE: Element = Element {
    symbol: "Br",
    number: 35,
};
pub const KRYPTON: Element = Element {
    symbol: "Kr",
    number: 36,
};
pub const RUBIDIUM: Element = Element {
    symbol: "Rb",
    number: 37,
};
pub const STRONTIUM: Element = Element {
    symbol: "Sr",
    number: 38,
};
pub const YTTRIUM: Element = Element {
    symbol: "Y",
    number: 39,
};
pub const ZIRCONIUM: Element = Element {
    symbol: "Zr",
    number: 40,
};
pub const NIOBIUM: Element = Element {
    symbol: "Nb",
    number: 41,
};
pub const MOLYBDENUM: Element = Element {
    symbol: "Mo",
    number: 42,
};
pub const TECHNETIUM: Element = Element {
    symbol: "Tc",
    number: 43,
};
pub const RUTHENIUM: Element = Element {
    symbol: "Ru",
    number: 44,
};
pub const RHODIUM: Element = Element {
    symbol: "Rh",
    number: 45,
};
pub const PALLADIUM: Element = Element {
    symbol: "Pd",
    number: 46,
};
pub const SILVER: Element = Element {
    symbol: "Ag",
    number: 47,
};
pub const CADMIUM: Element = Element {
    symbol: "Cd",
    number: 48,
};
pub const INDIUM: Element = Element {
    symbol: "In",
    number: 49,
};
pub const TIN: Element = Element {
    symbol: "Sn",
    number: 50,
};
pub const ANTIMONY: Element = Element {
    symbol: "Sb",
    number: 51,
};
pub const TELLURIUM: Element = Element {
    symbol: "Te",
    number: 52,
};
pub const IODINE: Element = Element {
    symbol: "I",
    number: 53,
};
pub const XENON: Element = Element {
    symbol: "Xe",
    number: 54,
};
pub const CAESIUM: Element = Element {
    symbol: "Cs",
    number: 55,
};
pub const BARIUM: Element = Element {
    symbol: "Ba",
    number: 56,
};
pub const LANTHANUM: Element = Element {
    symbol: "La",
    number: 57,
};
pub const CERIUM: Element = Element {
    symbol: "Ce",
    number: 58,
};
pub const PRASEODYMIUM: Element = Element {
    symbol: "Pr",
    number: 59,
};
pub const NEODYMIUM: Element = Element {
    symbol: "Nd",
    number: 60,
};
pub const PROMETHIUM: Element = Element {
    symbol: "Pm",
    number: 61,
};
pub const SAMARIUM: Element = Element {
    symbol: "Sm",
    number: 62,
};
pub const EUROPIUM: Element = Element {
    symbol: "Eu",
    number: 63,
};
pub const GADOLINIUM: Element = Element {
    symbol: "Gd",
    number: 64,
};
pub const TERBIUM: Element = Element {
    symbol: "Tb",
    number: 65,
};
pub const DYSPROSIUM: Element = Element {
    symbol: "Dy",
    number: 66,
};
pub const HOLMIUM: Element = Element {
    symbol: "Ho",
    number: 67,
};
pub const ERBIUM: Element = Element {
    symbol: "Er",
    number: 68,
};
pub const THULIUM: Element = Element {
    symbol: "Tm",
    number: 69,
};
pub const YTTERBIUM: Element = Element {
    symbol: "Yb",
    number: 70,
};
pub const LUTETIUM: Element = Element {
    symbol: "Lu",
    number: 71,
};
pub const HAFNIUM: Element = Element {
    symbol: "Hf",
    number: 72,
};
pub const TANTALUM: Element = Element {
    symbol: "Ta",
    number: 73,
};
pub const TUNGSTEN: Element = Element {
    symbol: "W",
    number: 74,
};
pub const RHENIUM: Element = Element {
    symbol: "Re",
    number: 75,
};
pub const OSMIUM: Element = Element {
    symbol: "Os",
    number: 76,
};
pub const IRIDIUM: Element = Element {
    symbol: "Ir",
    number: 77,
};
pub const PLATINUM: Element = Element {
    symbol: "Pt",
    number: 78,
};
pub const GOLD: Element = Element {
    symbol: "Au",
    number: 79,
};
pub const MERCURY: Element = Element {
    symbol: "Hg",
    number: 80,
};
pub const THALLIUM: Element = Element {
    symbol: "Tl",
    number: 81,
};
pub const LEAD: Element = Element {
    symbol: "Pb",
    number: 82,
};
pub const BISMUTH: Element = Element {
    symbol: "Bi",
    number: 83,
};
pub const POLONIUM: Element = Element {
    symbol: "Po",
    number: 84,
};
pub const ASTATINE: Element = Element {
    symbol: "At",
    number: 85,
};
pub const RADON: Element = Element {
    symbol: "Rn",
    number: 86,
};
pub const FRANCIUM: Element = Element {
    symbol: "Fr",
    number: 87,
};
pub const RADIUM: Element = Element {
    symbol: "Ra",
    number: 88,
};
pub const ACTINIUM: Element = Element {
    symbol: "Ac",
    number: 89,
};
pub const THORIUM: Element = Element {
    symbol: "Th",
    number: 90,
};
pub const PROTACTINIUM: Element = Element {
    symbol: "Pa",
    number: 91,
};
pub const URANIUM: Element = Element {
    symbol: "U",
    number: 92,
};
pub const NEPTUNIUM: Element = Element {
    symbol: "Np",
    number: 93,
};
pub const PLUTONIUM: Element = Element {
    symbol: "Pu",
    number: 94,
};
pub const AMERICIUM: Element = Element {
    symbol: "Am",
    number: 95,
};
pub const CURIUM: Element = Element {
    symbol: "Cm",
    number: 96,
};
pub const BERKELIUM: Element = Element {
    symbol: "Bk",
    number: 97,
};
pub const CALIFORNIUM: Element = Element {
    symbol: "Cf",
    number: 98,
};
pub const EINSTEINIUM: Element = Element {
    symbol: "Es",
    number: 99,
};
pub const FERMIUM: Element = Element {
    symbol: "Fm",
    number: 100,
};
pub const MENDELEVIUM: Element = Element {
    symbol: "Md",
    number: 101,
};
pub const NOBELIUM: Element = Element {
    symbol: "No",
    number: 102,
};
pub const LAWRENCIUM: Element = Element {
    symbol: "Lr",
    number: 103,
};
pub const RUTHERFORDIUM: Element = Element {
    symbol: "Rf",
    number: 104,
};
pub const DUBNIUM: Element = Element {
    symbol: "Db",
    number: 105,
};
pub const SEABORGIUM: Element = Element {
    symbol: "Sg",
    number: 106,
};
pub const BOHRIUM: Element = Element {
    symbol: "Bh",
    number: 107,
};
pub const HASSIUM: Element = Element {
    symbol: "Hs",
    number: 108,
};
pub const MEITNERIUM: Element = Element {
    symbol: "Mt",
    number: 109,
};
pub const DARMSTADTIUM: Element = Element {
    symbol: "Ds",
    number: 110,
};
pub const ROENTGENIUM: Element = Element {
    symbol: "Rg",
    number: 111,
};
pub const COPERNICIUM: Element = Element {
    symbol: "Cn",
    number: 112,
};
pub const NIHONIUM: Element = Element {
    symbol: "Nh",
    number: 113,
};
pub const FLEROVIUM: Element = Element {
    symbol: "Fl",
    number: 114,
};
pub const MOSCOVIUM: Element = Element {
    symbol: "Mc",
    number: 115,
};
pub const LIVERMORIUM: Element = Element {
    symbol: "Lv",
    number: 116,
};
pub const TENNESSINE: Element = Element {
    symbol: "Ts",
    number: 117,
};
pub const OGANESSON: Element = Element {
    symbol: "Og",
    number: 118,
};

/// Every element of the periodic table, indexed by atomic number minus one.
pub const ELEMENTS: [Element; 118] = [
    HYDROGEN,
    HELIUM,
    LITHIUM,
    BERYLLIUM,
    BORON,
    CARBON,
    NITROGEN,
    OXYGEN,
    FLUORINE,
    NEON,
    SODIUM,
    MAGNESIUM,
    ALUMINIUM,
    SILICON,
    PHOSPHORUS,
    SULFUR,
    CHLORINE,
    ARGON,
    POTASSIUM,
    CALCIUM,
    SCANDIUM,
    TITANIUM,
    VANADIUM,
    CHROMIUM,
    MANGANESE,
    IRON,
    COBALT,
    NICKEL,
    COPPER,
    ZINC,
    GALLIUM,
    GERMANIUM,
    ARSENIC,
    SELENIUM,
    BROMINE,
    KRYPTON,
    RUBIDIUM,
    STRONTIUM,
    YTTRIUM,
    ZIRCONIUM,
    NIOBIUM,
    MOLYBDENUM,
    TECHNETIUM,
    RUTHENIUM,
    RHODIUM,
    PALLADIUM,
    SILVER,
    CADMIUM,
    INDIUM,
    TIN,
    ANTIMONY,
    TELLURIUM,
    IODINE,
    XENON,
    CAESIUM,
    BARIUM,
    LANTHANUM,
    CERIUM,
    PRASEODYMIUM,
    NEODYMIUM,
    PROMETHIUM,
    SAMARIUM,
    EUROPIUM,
    GADOLINIUM,
    TERBIUM,
    DYSPROSIUM,
    HOLMIUM,
    ERBIUM,
    THULIUM,
    YTTERBIUM,
    LUTETIUM,
    HAFNIUM,
    TANTALUM,
    TUNGSTEN,
    RHENIUM,
    OSMIUM,
    IRIDIUM,
    PLATINUM,
    GOLD,
    MERCURY,
    THALLIUM,
    LEAD,
    BISMUTH,
    POLONIUM,
    ASTATINE,
    RADON,
    FRANCIUM,
    RADIUM,
    ACTINIUM,
    THORIUM,
    PROTACTINIUM,
    URANIUM,
    NEPTUNIUM,
    PLUTONIUM,
    AMERICIUM,
    CURIUM,
    BERKELIUM,
    CALIFORNIUM,
    EINSTEINIUM,
    FERMIUM,
    MENDELEVIUM,
    NOBELIUM,
    LAWRENCIUM,
    RUTHERFORDIUM,
    DUBNIUM,
    SEABORGIUM,
    BOHRIUM,
    HASSIUM,
    MEITNERIUM,
    DARMSTADTIUM,
    ROENTGENIUM,
    COPERNICIUM,
    NIHONIUM,
    FLEROVIUM,
    MOSCOVIUM,
    LIVERMORIUM,
    TENNESSINE,
    OGANESSON,
];

/// Looks up an element by its symbol, ignoring case.
pub fn from_symbol(symbol: String) -> Element {
    ELEMENTS
        .iter()
        .find(|element| element.symbol.eq_ignore_ascii_case(&symbol))
        .cloned()
        .unwrap_or_else(|| panic!("Unknown element with symbol '{symbol}'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_numbers_are_sequential() {
        for (i, element) in ELEMENTS.iter().enumerate() {
            assert_eq!(usize::from(element.number), i + 1, "{}", element.symbol);
        }
    }

    #[test]
    fn symbols_are_unique_and_capitalized() {
        for (i, element) in ELEMENTS.iter().enumerate() {
            let mut chars = element.symbol.chars();
            assert!(
                chars.next().unwrap().is_ascii_uppercase(),
                "{}",
                element.symbol
            );
            assert!(chars.all(|c| c.is_ascii_lowercase()), "{}", element.symbol);
            assert!(element.symbol.len() <= 2, "{}", element.symbol);
            for other in &ELEMENTS[i + 1..] {
                assert!(
                    !element.symbol.eq_ignore_ascii_case(other.symbol),
                    "{} and {} clash",
                    element.symbol,
                    other.symbol
                );
            }
        }
    }

    #[test]
    fn landmark_elements_have_the_right_number() {
        for (element, number) in [
            (POTASSIUM, 19),
            (IRON, 26),
            (KRYPTON, 36),
            (SILVER, 47),
            (XENON, 54),
            (LUTETIUM, 71),
            (GOLD, 79),
            (RADON, 86),
            (URANIUM, 92),
            (OGANESSON, 118),
        ] {
            assert_eq!(element.number, number, "{}", element.symbol);
        }
    }

    #[test]
    fn from_symbol_roundtrips() {
        for element in ELEMENTS {
            assert_eq!(from_symbol(element.symbol.to_string()), element);
            assert_eq!(from_symbol(element.symbol.to_ascii_uppercase()), element);
            assert_eq!(from_symbol(element.symbol.to_ascii_lowercase()), element);
        }
    }

    #[test]
    #[should_panic(expected = "Unknown element")]
    fn from_symbol_rejects_unknown() {
        from_symbol("Xx".to_string());
    }
}
