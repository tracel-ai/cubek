//! How an operand's values sit in memory, and what a leaf must read to serve one.

use cubecl::ir::types::Fp8Format;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::CubeDebug;
use cubecl::quant::scheme::QuantValue;
use cubecl::quant::scheme::ScaleDtype;

/// How an operand's values sit in memory.
///
/// A leaf asks [`Tile::packing`](crate::Tile::packing) and reads through the matching view;
/// nothing outside the view constructors turns a factor back into a storage element.
///
/// Self-describing: a packed operand names the field its values occupy, so the read unpacks from
/// this alone. That lets packing be *stated* on an operand
/// ([`TileSpec::packed`](crate::TileSpec::packed)) with no scales beside it.
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
}
impl CubeDebug for Packing {}

/// The slot one packed value occupies.
///
/// A quantized value's field, an 8-bit float code stored as a byte, or a whole float. Every
/// stored width is one of these, so one binding serves them all: a `ue8m0` scale sits four to a
/// word, an `f16` one two, an `f32` one alone, and the read is the same read.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Field {
    Quant(QuantValue),
    Fp8(Fp8Format),
    Float(FloatKind),
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

/// The one conversion from a float kind: an 8-bit float code is a [`Fp8`](Field::Fp8) field
/// read through its own decoder, a wider float a [`Float`](Field::Float) field read by
/// reinterpreting its slot.
impl From<FloatKind> for Field {
    fn from(kind: FloatKind) -> Self {
        Field::of_float(kind)
    }
}

impl Field {
    /// The slot's width in bits.
    pub fn size_bits(self) -> usize {
        match self {
            Field::Quant(value) => value.size_bits(),
            Field::Fp8(_) => 8,
            Field::Float(kind) => Field::float_bits(kind),
        }
    }

    /// Values one `u32` holds.
    pub fn per_word(self) -> usize {
        u32::BITS as usize / self.size_bits()
    }
}

impl Field {
    /// The field a float of `kind` occupies: an 8-bit code is a [`Fp8`](Field::Fp8) field read
    /// through its own decoder, a wider float a [`Float`](Field::Float) field read back by
    /// reinterpreting its slot. One call, so a caller holding an element type never has to know
    /// which of the two it is looking at.
    pub fn of_float(kind: FloatKind) -> Field {
        match kind {
            FloatKind::E4M3 => Field::Fp8(Fp8Format::E4M3),
            FloatKind::E5M2 => Field::Fp8(Fp8Format::E5M2),
            FloatKind::UE8M0 => Field::Fp8(Fp8Format::UE8M0),
            other => Field::Float(other),
        }
    }

    /// The field a scale of `dtype` occupies in the word it is stored in. Every scale is one of
    /// these, which is what lets one binding serve them all, and [`per_word`](Field::per_word) is
    /// then how many of them a read brings.
    pub fn of_scale(dtype: ScaleDtype) -> Field {
        match ElemType::from_scale_dtype(dtype) {
            ElemType::Float(kind) => Field::of_float(kind),
            other => panic!("Field: a scale is stored as a float, and {dtype:?} is {other:?}"),
        }
    }

    /// The bits a whole float occupies. Two widths, because an 8-bit code is [`Field::Fp8`] and
    /// reads back through its own decoder, not by reinterpreting a slot.
    pub fn float_bits(kind: FloatKind) -> usize {
        match kind {
            FloatKind::F32 => 32,
            FloatKind::F16 | FloatKind::BF16 => 16,
            other => panic!(
                "Field::Float: a stored float is 16 or 32 bits wide, got {other:?}; an 8-bit \
                 code is Field::Fp8"
            ),
        }
    }

    /// How this field reads back.
    ///
    /// Named rather than asked, because two callers need it for different reasons: the view
    /// matches it to pick a read, and a launch matches it to refuse a field before it compiles a
    /// kernel around one. Deriving it twice is how the two drift.
    pub fn decode(self) -> FieldDecode {
        match self {
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
            Field::Float(kind) => FieldDecode::Bits(kind),
        }
    }
}

/// How a stored field reads back ([`Field::decode`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FieldDecode {
    /// An integer slot: the top bit is its sign, so the value sign-extends out of its bits.
    SignExtended,
    /// A 4-bit float code, read back by reinterpreting the byte two of them share.
    Reinterpreted,
    /// An 8-bit float code, read back through the format's own decoder.
    Byte(Fp8Format),
    /// A whole float, read back by reinterpreting the bits of its slot.
    Bits(FloatKind),
}

impl Packing {
    /// Values per stored element: one, unless a `u32` holds several fields.
    pub fn factor(&self) -> usize {
        match self {
            Packing::Plain | Packing::Native => 1,
            Packing::Packed { field } => field.per_word(),
        }
    }

    /// The physical line a `served`-wide logical line occupies.
    pub fn physical(&self, served: usize) -> usize {
        served / self.factor()
    }

    /// The width a binding `bound` wide serves: the bound line times the values a stored element
    /// holds. A packed line serves whole words — every field of every word it reads.
    pub fn served(&self, bound: usize) -> usize {
        bound * self.factor()
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

    /// An 8-bit code is a byte field however it is named; anything wider is its own bits.
    #[test]
    fn a_float_element_names_its_own_field() {
        assert_eq!(
            Field::of_float(FloatKind::UE8M0),
            Field::Fp8(Fp8Format::UE8M0)
        );
        assert_eq!(
            Field::of_float(FloatKind::E4M3),
            Field::Fp8(Fp8Format::E4M3)
        );
        assert_eq!(
            Field::of_float(FloatKind::F16),
            Field::Float(FloatKind::F16)
        );
    }

    /// A stored float is one, two or four to a word, and the widest fills it.
    #[test]
    fn a_stored_float_is_a_field_of_its_own_width() {
        assert_eq!(Field::Float(FloatKind::F32).per_word(), 1);
        assert_eq!(Field::Float(FloatKind::F16).per_word(), 2);
        assert_eq!(Field::Float(FloatKind::BF16).size_bits(), 16);
        assert_eq!(
            Field::from(FloatKind::F32).decode(),
            FieldDecode::Bits(FloatKind::F32)
        );
    }

    /// The 8-bit float codes decode through their format; the 4-bit one through its pair.
    #[test]
    fn a_byte_field_decodes_through_its_format() {
        assert_eq!(
            Field::from(QuantValue::E4M3).decode(),
            FieldDecode::Byte(Fp8Format::E4M3)
        );
        assert_eq!(
            Field::from(Fp8Format::UE8M0).decode(),
            FieldDecode::Byte(Fp8Format::UE8M0)
        );
        assert_eq!(
            Field::from(QuantValue::E2M1).decode(),
            FieldDecode::Reinterpreted
        );
    }
}
