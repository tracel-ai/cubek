use crate::kernels::select::problem::OperandKind;

/// The kind of product a candidate is listed for: one table holds both, and a candidate declines
/// the other kind before compiling anything.
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

impl ListedFor {
    /// Whether a product of kind `actual` is one the candidate is listed for.
    pub(crate) fn admits(self, actual: Product) -> bool {
        match (self, actual) {
            (Self::Either, _) | (Self::Float, Product::Float) | (Self::Packed, Product::Packed) => {
                true
            }
            (Self::Float, Product::Packed) | (Self::Packed, Product::Float) => false,
        }
    }
}

/// Declines a product of kind `actual` where it is not the kind a candidate is `listed` for.
///
/// # Errors
///
/// Where `actual` is the other kind.
pub(crate) fn listed_for(listed: Product, actual: Product) -> Result<(), String> {
    match (listed, actual) {
        (Product::Float, Product::Float) | (Product::Packed, Product::Packed) => Ok(()),
        (Product::Float, Product::Packed) => {
            Err("listed for float inputs; an input is packed".into())
        }
        (Product::Packed, Product::Float) => {
            Err("listed for a packed input; both are float".into())
        }
    }
}
