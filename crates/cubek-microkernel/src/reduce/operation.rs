use cubecl::tile::{Monoid, TOP_K_UNROLL_BUDGET};

/// The top-k slots a thread keeps across the lanes of a line, `k` values and as many indices a
/// lane, before the line narrows: past it they spill out of registers. Scalar, a top-20 of a
/// 151936-wide row ran 2.4 times faster than at width 8 on a GP100.
pub const TOP_K_LANE_SLOTS: usize = 32;

/// A reduction along one axis: a map on each value as it is read, a monoid that combines the
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
/// | `TopK` | – | `Max` | [`TopValues`](cubecl::tile::TopValues) | its values |
/// | `ArgTopK` | – | `Max` | [`TopK`](cubecl::tile::TopK) | its indices |
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

/// What a reduction does to each value as it reads it, before the monoid combines it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReadMap {
    AsRead,
    Abs,
    /// `1` where the value is not zero, `0` where it is.
    NonZero,
}

/// What a reduction accumulates its values into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceCarrier {
    /// The values themselves.
    Values,
    /// A value with the index it was read at.
    Indexed,
    /// The `k` best values with their indices.
    TopK { k: usize },
    /// The `k` best values, without their indices.
    TopValues { k: usize },
}

impl ReduceCarrier {
    /// Whether it keeps the index each value was read at.
    pub const fn carries_indices(self) -> bool {
        !matches!(self, Self::Values | Self::TopValues { .. })
    }

    /// Bytes one carrier of `width` lanes holds, its values `value_bytes` each and, where it
    /// carries indices, a `u32` beside each.
    pub const fn bytes(self, width: usize, value_bytes: usize) -> usize {
        let lane = match self.carries_indices() {
            true => value_bytes + size_of::<u32>(),
            false => value_bytes,
        };
        let slots = match self {
            Self::TopK { k } | Self::TopValues { k } => k,
            Self::Values | Self::Indexed => 1,
        };
        slots * width * lane
    }
}

/// What a reduction writes of what it accumulated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceWrites {
    Values,
    Indices,
    /// The values to the output, and their indices to a second output beside it.
    Both,
}

impl ReduceOperation {
    /// The monoid that combines the values.
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

    /// What the values accumulate into.
    pub const fn carrier(self) -> ReduceCarrier {
        match self {
            Self::ArgMax | Self::ArgMin => ReduceCarrier::Indexed,
            Self::TopK { k } => ReduceCarrier::TopValues { k },
            Self::ArgTopK { k } => ReduceCarrier::TopK { k },
            _ => ReduceCarrier::Values,
        }
    }

    /// What is written of the accumulated carrier.
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

    /// The widest line its accumulator keeps apart, where the carrier bounds it. A top-k keeps `k`
    /// values and their indices for every lane of a line: past [`TOP_K_LANE_SLOTS`] slots a
    /// thread they spill out of registers, so the line narrows, and once its `k`-by-`k` walks
    /// stay loops ([`TOP_K_UNROLL_BUDGET`]) its lists are indexed at run time, out of local memory,
    /// so the line is one value.
    pub const fn widest_line(self) -> Option<usize> {
        match self {
            Self::TopK { k } | Self::ArgTopK { k } => match k * k > TOP_K_UNROLL_BUDGET {
                true => Some(1),
                false => Some(TOP_K_LANE_SLOTS / k),
            },
            _ => None,
        }
    }

    /// Cells one reduced row writes: `k` for a top-k, one otherwise.
    pub const fn written_per_row(self) -> usize {
        match self {
            Self::TopK { k } | Self::ArgTopK { k } => k,
            _ => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each operation is the monoid, map and carrier its table row states.
    #[test]
    fn each_operation_is_its_table_row() {
        use ReduceOperation as Op;
        let rows = [
            (Op::Sum, Monoid::Sum, ReadMap::AsRead, ReduceCarrier::Values),
            (
                Op::Prod,
                Monoid::Prod,
                ReadMap::AsRead,
                ReduceCarrier::Values,
            ),
            (
                Op::Mean,
                Monoid::Sum,
                ReadMap::AsRead,
                ReduceCarrier::Values,
            ),
            (Op::Max, Monoid::Max, ReadMap::AsRead, ReduceCarrier::Values),
            (Op::Min, Monoid::Min, ReadMap::AsRead, ReduceCarrier::Values),
            (Op::MaxAbs, Monoid::Max, ReadMap::Abs, ReduceCarrier::Values),
            (
                Op::Any,
                Monoid::Max,
                ReadMap::NonZero,
                ReduceCarrier::Values,
            ),
            (
                Op::All,
                Monoid::Min,
                ReadMap::NonZero,
                ReduceCarrier::Values,
            ),
            (
                Op::ArgMax,
                Monoid::Max,
                ReadMap::AsRead,
                ReduceCarrier::Indexed,
            ),
            (
                Op::ArgMin,
                Monoid::Min,
                ReadMap::AsRead,
                ReduceCarrier::Indexed,
            ),
            (
                Op::TopK { k: 5 },
                Monoid::Max,
                ReadMap::AsRead,
                ReduceCarrier::TopValues { k: 5 },
            ),
            (
                Op::ArgTopK { k: 5 },
                Monoid::Max,
                ReadMap::AsRead,
                ReduceCarrier::TopK { k: 5 },
            ),
        ];
        for (operation, monoid, map, carrier) in rows {
            assert_eq!(operation.monoid(), monoid, "{operation:?}");
            assert_eq!(operation.read_map(), map, "{operation:?}");
            assert_eq!(operation.carrier(), carrier, "{operation:?}");
        }
    }

    /// Only the arg operations write indices, only the mean divides, and a top-k writes `k` cells.
    #[test]
    fn the_arg_operations_write_indices_and_a_top_k_writes_k() {
        use ReduceOperation as Op;
        for operation in [Op::ArgMax, Op::ArgMin, Op::ArgTopK { k: 2 }] {
            assert_eq!(operation.writes(), ReduceWrites::Indices, "{operation:?}");
        }
        for operation in [Op::Sum, Op::Max, Op::TopK { k: 2 }, Op::Any] {
            assert_eq!(operation.writes(), ReduceWrites::Values, "{operation:?}");
        }
        assert!(Op::Mean.divides_by_count() && !Op::Sum.divides_by_count());
        assert_eq!(Op::TopK { k: 7 }.written_per_row(), 7);
        assert_eq!(Op::ArgTopK { k: 7 }.written_per_row(), 7);
        assert_eq!(Op::ArgMax.written_per_row(), 1);
    }
}
