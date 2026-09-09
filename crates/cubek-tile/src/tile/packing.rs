//! How an operand's values sit in memory, and what a leaf must read to serve one.

use cubecl::ir::types::Fp8Format;
use cubecl::quant::scheme::QuantValue;

/// How an operand's values sit in memory.
///
/// A leaf asks [`Tile::packing`](crate::Tile::packing) and reads through the matching view;
/// nothing outside the view constructors turns a factor back into a storage element.
///
/// Self-describing: a packed operand names the field its values occupy, so the read unpacks from
/// this alone. That is what lets packing be *stated* on an operand
/// ([`TileSpec::packed`](crate::TileSpec::packed)) with no scales beside it, rather than being
/// recovered from a quantization scheme.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Packing {
    /// Served as stored: the storage element is the served element, the physical line the served
    /// line.
    Plain,
    /// One `i8` per value, widened at the read.
    Native,
    /// Several values per stored `u32`, each occupying a `field`-wide slot, unpacked at the read.
    /// A line serves whole words.
    Packed {
        /// The slot one value occupies: its width in bits and how those bits read back.
        field: Field,
    },
    /// [`Packed`](Packing::Packed), served `width` values at a time out of one word rather than
    /// the whole word: a reader stepping one value a step, as a scales operand read under a
    /// runtime block index is. The line is one word, the slot picked at the read.
    Subword {
        field: Field,
        /// Values one line serves, dividing the word's count.
        width: usize,
    },
}

/// The slot one packed value occupies.
///
/// A quantized value's field, or an 8-bit float code stored as a byte: a scale in `ue8m0` or
/// `ue4m3` is one of those, four to a word.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Field {
    Quant(QuantValue),
    Fp8(Fp8Format),
}

impl From<QuantValue> for Field {
    fn from(value: QuantValue) -> Self {
        Field::Quant(value)
    }
}

impl From<Fp8Format> for Field {
    fn from(format: Fp8Format) -> Self {
        Field::Fp8(format)
    }
}

impl Field {
    /// The slot's width in bits.
    pub fn size_bits(self) -> usize {
        match self {
            Field::Quant(value) => value.size_bits(),
            Field::Fp8(_) => 8,
        }
    }

    /// Values one `u32` holds.
    pub fn per_word(self) -> usize {
        u32::BITS as usize / self.size_bits()
    }
}

/// How a stored field reads back.
///
/// Named rather than asked, because two callers need it for different reasons: the view matches it
/// to pick a read, and a launch matches it to refuse a field before it compiles a kernel around
/// one. Deriving it twice is how the two drift.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FieldDecode {
    /// An integer slot: the top bit is its sign, so the value sign-extends out of its bits.
    SignExtended,
    /// A 4-bit float code, read back by reinterpreting the byte two of them share.
    Reinterpreted,
    /// An 8-bit float code, read back through the format's own decoder.
    Byte(Fp8Format),
}

/// How the packed view reads `field` back.
pub fn field_decode(field: Field) -> FieldDecode {
    match field {
        Field::Quant(
            QuantValue::Q8F
            | QuantValue::Q8S
            | QuantValue::Q4F
            | QuantValue::Q4S
            | QuantValue::Q2F
            | QuantValue::Q2S,
        ) => FieldDecode::SignExtended,
        Field::Quant(QuantValue::E2M1) => FieldDecode::Reinterpreted,
        Field::Quant(QuantValue::E4M3) => FieldDecode::Byte(Fp8Format::E4M3),
        Field::Quant(QuantValue::E5M2) => FieldDecode::Byte(Fp8Format::E5M2),
        Field::Fp8(format) => FieldDecode::Byte(format),
    }
}

impl Packing {
    /// Values per stored element: one, unless a `u32` holds several fields.
    pub fn factor(&self) -> usize {
        match self {
            Packing::Plain | Packing::Native => 1,
            Packing::Packed { field } | Packing::Subword { field, .. } => field.per_word(),
        }
    }

    /// The physical line a `served`-wide logical line occupies: one word for a sub-word read.
    pub fn physical(&self, served: usize) -> usize {
        match self {
            Packing::Subword { .. } => 1,
            _ => served / self.factor(),
        }
    }

    /// The width a binding `bound` wide serves: the bound line times the values a stored element
    /// holds, or the stated width for a sub-word read, whose binding is one word.
    pub fn served(&self, bound: usize) -> usize {
        match self {
            Packing::Subword { field, width } => {
                assert!(
                    bound == 1 && field.per_word().is_multiple_of(*width) && *width > 0,
                    "Packing::Subword: a sub-word read binds one word and serves a width that \
                     divides the {} values it holds; got a {bound}-word line serving {width}",
                    field.per_word()
                );
                *width
            }
            _ => bound * self.factor(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A packed line is narrower than the line it serves; the other two are the line itself.
    #[test]
    fn a_packing_narrows_the_line_it_stores() {
        assert_eq!(Packing::Plain.physical(16), 16);
        assert_eq!(Packing::Native.physical(16), 16);
        assert_eq!(
            Packing::Packed {
                field: QuantValue::Q4S.into()
            }
            .physical(16),
            2
        );
    }

    /// The factor is the field's own: eight 4-bit values in a word, four 8-bit ones.
    #[test]
    fn a_field_states_how_many_fit_in_a_word() {
        assert_eq!(
            Packing::Packed {
                field: QuantValue::Q4S.into()
            }
            .factor(),
            8
        );
        assert_eq!(
            Packing::Packed {
                field: QuantValue::Q8S.into()
            }
            .factor(),
            4
        );
        assert_eq!(
            Packing::Packed {
                field: QuantValue::Q2S.into()
            }
            .factor(),
            16
        );
        assert_eq!(
            Packing::Packed {
                field: Fp8Format::UE8M0.into()
            }
            .factor(),
            4
        );
    }

    /// A sub-word read is one word wide however few values it serves.
    #[test]
    fn a_subword_read_binds_one_word() {
        let packing = Packing::Subword {
            field: Fp8Format::UE8M0.into(),
            width: 1,
        };
        assert_eq!(packing.physical(1), 1);
        assert_eq!(packing.served(1), 1);
        assert_eq!(packing.factor(), 4);
    }

    /// The 8-bit float codes decode through their format; the 4-bit one through its pair.
    #[test]
    fn a_byte_field_decodes_through_its_format() {
        assert_eq!(
            field_decode(QuantValue::E4M3.into()),
            FieldDecode::Byte(Fp8Format::E4M3)
        );
        assert_eq!(
            field_decode(Fp8Format::UE8M0.into()),
            FieldDecode::Byte(Fp8Format::UE8M0)
        );
        assert_eq!(
            field_decode(QuantValue::E2M1.into()),
            FieldDecode::Reinterpreted
        );
    }
}
