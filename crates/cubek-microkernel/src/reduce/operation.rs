use cubecl::tile::Monoid;

/// A reduction along one axis: a map on each value as it is read, a monoid that folds the
/// values, and a last step on the total.
///
/// | operation | map on read | monoid | carrier | last step |
/// |---|---|---|---|---|
/// | `Sum`, `Prod`, `Max`, `Min` | – | itself | the values | – |
/// | `Mean` | – | `Sum` | the values | × 1/n |
/// | `MaxAbs` | `abs` | `Max` | the values | – |
/// | `Any` | `x != 0` | `Max` | the values | – |
/// | `All` | `x != 0` | `Min` | the values | – |
/// | `ArgMax`, `ArgMin` | – | `Max`, `Min` | [`Indexed`](cubecl::tile::Indexed) | its index |
/// | `TopK`, `ArgTopK` | – | `Max` | [`TopK`](cubecl::tile::TopK) | its values, its indices |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOperation {
    Sum,
    Prod,
    Mean,
    Max,
    Min,
    MaxAbs,
    Any,
    All,
    ArgMax,
    ArgMin,
    /// The `k` largest values, largest first.
    TopK {
        k: usize,
    },
    /// The indices of the `k` largest values, largest first.
    ArgTopK {
        k: usize,
    },
}

/// What a reduction does to each value as it reads it, before the monoid folds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReadMap {
    AsRead,
    Abs,
    /// `1` where the value is not zero, `0` where it is.
    NonZero,
}

/// What a reduction folds its values into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceCarrier {
    /// The values themselves.
    Values,
    /// A value with the index it was read at.
    Indexed,
    /// The `k` best values with their indices.
    TopK { k: usize },
}

/// What a reduction writes of what it folded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceWrites {
    Values,
    Indices,
}

impl ReduceOperation {
    /// The monoid that folds the values.
    pub const fn monoid(self) -> Monoid {
        match self {
            Self::Sum | Self::Mean => Monoid::Sum,
            Self::Prod => Monoid::Prod,
            Self::Max | Self::MaxAbs | Self::Any | Self::ArgMax => Monoid::Max,
            Self::TopK { .. } | Self::ArgTopK { .. } => Monoid::Max,
            Self::Min | Self::All | Self::ArgMin => Monoid::Min,
        }
    }

    /// What each value becomes as it is read.
    pub const fn read_map(self) -> ReadMap {
        match self {
            Self::MaxAbs => ReadMap::Abs,
            Self::Any | Self::All => ReadMap::NonZero,
            _ => ReadMap::AsRead,
        }
    }

    /// What the values fold into.
    pub const fn carrier(self) -> ReduceCarrier {
        match self {
            Self::ArgMax | Self::ArgMin => ReduceCarrier::Indexed,
            Self::TopK { k } | Self::ArgTopK { k } => ReduceCarrier::TopK { k },
            _ => ReduceCarrier::Values,
        }
    }

    /// What is written of the folded carrier.
    pub const fn writes(self) -> ReduceWrites {
        match self {
            Self::ArgMax | Self::ArgMin | Self::ArgTopK { .. } => ReduceWrites::Indices,
            _ => ReduceWrites::Values,
        }
    }

    /// Whether the total is divided by the count of values reduced.
    pub const fn divides_by_count(self) -> bool {
        matches!(self, Self::Mean)
    }

    /// Cells one reduced row writes: `k` for a top-k, one otherwise.
    pub const fn written_per_row(self) -> usize {
        match self {
            Self::TopK { k } | Self::ArgTopK { k } => k,
            _ => 1,
        }
    }
}
