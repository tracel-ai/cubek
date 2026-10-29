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

/// A setting and the kind of product it is listed for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Listing<S> {
    pub settings: S,
    pub kind: ListedFor,
}

impl<S: PartialEq> Listing<S> {
    /// The settings of a float product's list and of a packed product's, each with the kind it is
    /// listed for. A setting in both lists is listed for either, so no candidate name repeats.
    pub(crate) fn merged(float: Vec<S>, packed: Vec<S>) -> Vec<Self> {
        let (shared, only_packed): (Vec<_>, Vec<_>) = packed
            .into_iter()
            .partition(|settings| float.contains(settings));
        let float = float.into_iter().map(|settings| {
            let kind = match shared.contains(&settings) {
                true => ListedFor::Either,
                false => ListedFor::Float,
            };
            Self { settings, kind }
        });
        let packed = only_packed.into_iter().map(|settings| Self {
            settings,
            kind: ListedFor::Packed,
        });
        float.chain(packed).collect()
    }

    /// The settings of a float product's list, each listed for either kind since a float setting
    /// declines a stack it cannot read, then those of a packed product's the float list lacks,
    /// listed for a packed one.
    pub(crate) fn float_then_packed(float: Vec<S>, packed: Vec<S>) -> Vec<Self> {
        let extras: Vec<_> = packed
            .into_iter()
            .filter(|settings| !float.contains(settings))
            .collect();
        let float = float.into_iter().map(|settings| Self {
            settings,
            kind: ListedFor::Either,
        });
        let packed = extras.into_iter().map(|settings| Self {
            settings,
            kind: ListedFor::Packed,
        });
        float.chain(packed).collect()
    }

    /// The listings of `list` whose kind admits a product of kind `product`.
    pub(crate) fn admitted(list: &[Self], product: Product) -> impl Iterator<Item = &Self> {
        list.iter()
            .filter(move |listing| listing.kind.admits(product).is_ok())
    }
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

/// The first of `attempts` that is `Ok`, the rest never run; where none is, the last refusal.
///
/// # Panics
///
/// On no attempt: a floor is listed for every kind of stack.
pub(crate) fn first_served(
    attempts: impl IntoIterator<Item = Result<(), Refusal>>,
) -> Result<(), Refusal> {
    let mut refusal = None;
    for attempt in attempts {
        match attempt {
            Ok(()) => return Ok(()),
            Err(why) => refusal = Some(why),
        }
    }
    Err(refusal.expect("a floor is listed for every kind of stack"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A setting in both lists is listed for either kind of stack, so no candidate name repeats.
    #[test]
    fn a_setting_in_both_lists_is_listed_for_either_stack() {
        let listing = Listing::merged(vec![1, 2], vec![2, 3]);
        let kinds: Vec<_> = listing.iter().map(|l| (l.settings, l.kind)).collect();
        assert_eq!(
            kinds,
            [
                (1, ListedFor::Float),
                (2, ListedFor::Either),
                (3, ListedFor::Packed)
            ]
        );
    }

    /// Every float setting races on either kind of stack, and a packed one only where the float
    /// list lacks it.
    #[test]
    fn float_settings_are_listed_for_either_and_new_packed_ones_for_packed() {
        let listing = Listing::float_then_packed(vec![1, 2], vec![2, 3]);
        let kinds: Vec<_> = listing.iter().map(|l| (l.settings, l.kind)).collect();
        assert_eq!(
            kinds,
            [
                (1, ListedFor::Either),
                (2, ListedFor::Either),
                (3, ListedFor::Packed)
            ]
        );
    }
}
