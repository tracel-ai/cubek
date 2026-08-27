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
