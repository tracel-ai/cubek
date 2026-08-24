//! How an operand's values sit in memory, and what a leaf must read to serve one.

/// How an operand's values sit in memory.
///
/// A leaf asks [`Tile::packing`](crate::Tile::packing) and reads through the matching view;
/// nothing outside the view constructors turns a factor back into a storage element.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Packing {
    /// Served as stored: the storage element is the served element, the physical line the served
    /// line.
    Plain,
    /// One `i8` per value, scaled at the read.
    Native,
    /// `factor` values per stored `u32`, unpacked and scaled at the read.
    Packed {
        /// Values per stored word.
        factor: usize,
    },
}

impl Packing {
    /// The physical line a `served`-wide logical line occupies.
    pub fn physical(&self, served: usize) -> usize {
        match self {
            Packing::Plain | Packing::Native => served,
            Packing::Packed { factor } => served / factor,
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
        assert_eq!(Packing::Packed { factor: 8 }.physical(16), 2);
    }
}
