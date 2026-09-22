//! One axis's size: a comptime constant or a runtime scalar.

/// One axis's size.
/// `Static` is a comptime constant (a tile edge);
/// `Dynamic` is a runtime scalar resolved in-kernel from the tensor shape.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Extent {
    Static(usize),
    Dynamic,
}

impl Extent {
    /// The comptime size; panics on `Dynamic` (a runtime extent has no comptime value;
    /// resolve it from the tensor shape).
    pub fn get(self) -> usize {
        match self {
            Extent::Static(n) => n,
            Extent::Dynamic => {
                panic!("Extent::get: this axis is Dynamic; its size is only known at runtime")
            }
        }
    }

    pub fn is_dynamic(self) -> bool {
        matches!(self, Extent::Dynamic)
    }
}
