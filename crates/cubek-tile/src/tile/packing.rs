//! How an operand's values sit in memory, and what a leaf must read to serve one.

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
    Packed {
        /// The slot one value occupies: its width in bits and how those bits read back.
        field: QuantValue,
    },
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
    /// A float code: the bits are an index into the format's values, read back by reinterpreting
    /// the byte that [`QuantValue::native_packing`] of them share.
    Reinterpreted,
    /// A field no packed view here serves.
    Unserved,
}

/// How the packed view reads `field` back.
///
/// The 8-bit minifloats are [`Unserved`](FieldDecode::Unserved): they reinterpret like `e2m1` does,
/// but a byte holds one rather than a pair, so the read is a third shape and nothing asks for it.
/// Bind such a tensor at its own element and let the contraction cast it.
pub fn field_decode(field: QuantValue) -> FieldDecode {
    match field {
        QuantValue::Q8F
        | QuantValue::Q8S
        | QuantValue::Q4F
        | QuantValue::Q4S
        | QuantValue::Q2F
        | QuantValue::Q2S => FieldDecode::SignExtended,
        QuantValue::E2M1 => FieldDecode::Reinterpreted,
        QuantValue::E4M3 | QuantValue::E5M2 => FieldDecode::Unserved,
    }
}

impl Packing {
    /// Values per stored element: one, unless a `u32` holds several fields.
    pub fn factor(&self) -> usize {
        match self {
            Packing::Plain | Packing::Native => 1,
            Packing::Packed { field } => u32::BITS as usize / field.size_bits(),
        }
    }

    /// The physical line a `served`-wide logical line occupies.
    pub fn physical(&self, served: usize) -> usize {
        served / self.factor()
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
                field: QuantValue::Q4S
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
                field: QuantValue::Q4S
            }
            .factor(),
            8
        );
        assert_eq!(
            Packing::Packed {
                field: QuantValue::Q8S
            }
            .factor(),
            4
        );
        assert_eq!(
            Packing::Packed {
                field: QuantValue::Q2S
            }
            .factor(),
            16
        );
    }
}
