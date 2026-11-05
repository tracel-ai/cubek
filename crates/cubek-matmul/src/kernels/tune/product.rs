use crate::kernels::select::problem::OperandKind;
use cubecl::tile::launch::Refusal;

/// The kind of a product, which a candidate listed for the other kind declines before compiling
/// anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Product {
    Float,
    /// A packed input, on either side.
    Packed,
}

impl Product {
    /// The kind of a product whose inputs are `kinds`.
    pub(crate) fn of(kinds: impl IntoIterator<Item = OperandKind>) -> Self {
        let packed = |kind| matches!(kind, OperandKind::Quantized(_));
        match kinds.into_iter().any(packed) {
            true => Self::Packed,
            false => Self::Float,
        }
    }
}

/// The kind of product a candidate is listed for: one kind, or either where the same setting
/// serves both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ListedFor {
    Float,
    Packed,
    Either,
}

/// What a float setting declines: a packed input, which it was not listed for.
pub const FLOAT_SETTING_PACKED_INPUT: Refusal = Refusal::Declined {
    why: "a setting listed for float inputs reads them, and an input is packed",
};

/// What a packed setting declines: two float inputs, which it was not listed for.
pub const PACKED_SETTING_FLOAT_INPUTS: Refusal = Refusal::Declined {
    why: "a setting listed for a packed input reads one, and both inputs are float",
};

impl ListedFor {
    /// Declines a product of kind `actual` where it is not a kind the candidate is listed for.
    ///
    /// # Errors
    ///
    /// [`FLOAT_SETTING_PACKED_INPUT`] or [`PACKED_SETTING_FLOAT_INPUTS`], for the kind it is not
    /// listed for.
    pub(crate) fn admits(self, actual: Product) -> Result<(), Refusal> {
        match (self, actual) {
            (Self::Either, _) | (Self::Float, Product::Float) | (Self::Packed, Product::Packed) => {
                Ok(())
            }
            (Self::Float, Product::Packed) => Err(FLOAT_SETTING_PACKED_INPUT),
            (Self::Packed, Product::Float) => Err(PACKED_SETTING_FLOAT_INPUTS),
        }
    }
}
