use cubecl::tile::{Monoid, TOP_K_UNROLL_BUDGET};

/// The top-k slots a thread keeps across a line's lanes that autotune tries: a setting of every
/// reduce microkernel.
pub const TOP_K_LANE_SLOTS: [usize; 1] = [32];

/// A reduction along one dimension: a map on each value as it is read, a monoid that combines the
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

    /// The monoid whose identity a read past the problem's edge returns, so the read leaves the
    /// total unchanged: the combining monoid's, where its map leaves that identity as it is, else
    /// the one whose identity the map takes there (`MaxAbs` and `Any` read zero, `All` one).
    pub const fn edge_monoid(self) -> Monoid {
        match self {
            Self::Sum | Self::Mean | Self::MaxAbs | Self::Any => Monoid::Sum,
            Self::Prod | Self::All => Monoid::Prod,
            Self::Max | Self::ArgMax | Self::TopK { .. } | Self::ArgTopK { .. } => Monoid::Max,
            Self::Min | Self::ArgMin => Monoid::Min,
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
    ///
    /// A top-1 is the max: a list of one keeps [`Indexed`](cubecl::tile::Indexed)'s order, so it
    /// accumulates as the max does, without the list's walk on every value read.
    pub const fn carrier(self) -> ReduceCarrier {
        match self {
            Self::ArgMax | Self::ArgMin | Self::ArgTopK { k: 1 } => ReduceCarrier::Indexed,
            Self::TopK { k: 1 } => ReduceCarrier::Values,
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

    /// The widest line its accumulator keeps apart, where the carrier bounds it, for a thread
    /// keeping `lane_slots` top-k slots across a line's lanes. A top-k keeps `k` values and their
    /// indices for every lane of a line: past `lane_slots` they spill out of registers, so the line
    /// narrows, and once its `k`-by-`k` walks stay loops ([`TOP_K_UNROLL_BUDGET`]) its lists are
    /// indexed at run time, out of local memory, so the line is one value.
    ///
    /// # Panics
    ///
    /// Where a top-k keeps `k = 0` values.
    pub const fn widest_line_at_lane_slots(self, lane_slots: usize) -> Option<usize> {
        match self {
            Self::TopK { k } | Self::ArgTopK { k } => match k * k > TOP_K_UNROLL_BUDGET {
                true => Some(1),
                false => Some(lane_slots / k),
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

/// The host's answer for each operation, the Reference Implementation every reduce kernel and
/// microkernel is checked against.
#[cfg(feature = "cpu-reference")]
impl ReduceOperation {
    /// The cells it writes for `row`, computed on the host: one value, or `k` best first for a
    /// top-k, and an index written as a number. A NaN ranks ahead of every number and the lower
    /// index wins a tie, so a max or a min keeps a NaN. A row shorter than `k` writes one cell a
    /// value.
    pub fn reference_cells(self, row: &[f32]) -> Vec<f32> {
        match self {
            Self::Sum => vec![row.iter().sum()],
            Self::Prod => vec![row.iter().product()],
            Self::Mean => vec![row.iter().sum::<f32>() / row.len() as f32],
            Self::Max => vec![ranked(row, Rank::Largest)[0].0],
            Self::Min => vec![ranked(row, Rank::Smallest)[0].0],
            Self::MaxAbs => vec![ranked(&magnitudes(row), Rank::Largest)[0].0],
            Self::Any => vec![flag(row.iter().any(|value| *value != 0.0))],
            Self::All => vec![flag(row.iter().all(|value| *value != 0.0))],
            Self::ArgMax => vec![ranked(row, Rank::Largest)[0].1 as f32],
            Self::ArgMin => vec![ranked(row, Rank::Smallest)[0].1 as f32],
            Self::TopK { k } => ranked(row, Rank::Largest)
                .iter()
                .take(k)
                .map(|e| e.0)
                .collect(),
            Self::ArgTopK { k } => ranked(row, Rank::Largest)
                .iter()
                .take(k)
                .map(|e| e.1 as f32)
                .collect(),
        }
    }
}

/// Which end of a row ranks first.
#[cfg(feature = "cpu-reference")]
#[derive(Clone, Copy)]
enum Rank {
    Largest,
    Smallest,
}

/// Every value of `row` with its index, best first under `rank`: a NaN ahead of every number,
/// then the larger (smaller) value, then the lower index.
#[cfg(feature = "cpu-reference")]
fn ranked(row: &[f32], rank: Rank) -> Vec<(f32, usize)> {
    let mut order: Vec<(f32, usize)> = row.iter().copied().zip(0..).collect();
    order.sort_by(|a, b| {
        let nan_first = b.0.is_nan().cmp(&a.0.is_nan());
        let by_value = match rank {
            Rank::Largest => b.0.partial_cmp(&a.0),
            Rank::Smallest => a.0.partial_cmp(&b.0),
        };
        nan_first
            .then(by_value.unwrap_or(core::cmp::Ordering::Equal))
            .then(a.1.cmp(&b.1))
    });
    order
}

/// The magnitude of every value of `row`, in order.
#[cfg(feature = "cpu-reference")]
fn magnitudes(row: &[f32]) -> Vec<f32> {
    row.iter().map(|value| value.abs()).collect()
}

/// One where `holds`, zero where not: how a reduction writes a truth.
#[cfg(feature = "cpu-reference")]
fn flag(holds: bool) -> f32 {
    f32::from(u8::from(holds))
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
            (
                Op::TopK { k: 1 },
                Monoid::Max,
                ReadMap::AsRead,
                ReduceCarrier::Values,
            ),
            (
                Op::ArgTopK { k: 1 },
                Monoid::Max,
                ReadMap::AsRead,
                ReduceCarrier::Indexed,
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

    /// A read past the edge leaves the total unchanged: each operation's edge monoid is the one
    /// whose identity its map leaves as it is.
    #[test]
    fn each_edge_reads_the_identity_its_map_keeps() {
        use ReduceOperation as Op;
        let rows = [
            (Op::Sum, Monoid::Sum),
            (Op::Mean, Monoid::Sum),
            (Op::MaxAbs, Monoid::Sum),
            (Op::Any, Monoid::Sum),
            (Op::Prod, Monoid::Prod),
            (Op::All, Monoid::Prod),
            (Op::Max, Monoid::Max),
            (Op::ArgMax, Monoid::Max),
            (Op::TopK { k: 3 }, Monoid::Max),
            (Op::ArgTopK { k: 3 }, Monoid::Max),
            (Op::Min, Monoid::Min),
            (Op::ArgMin, Monoid::Min),
        ];
        for (operation, edge) in rows {
            assert_eq!(operation.edge_monoid(), edge, "{operation:?}");
        }
    }

    /// A carrier holds its slots' values, each lane of each, and a `u32` beside each value where
    /// it carries indices.
    #[test]
    fn a_carrier_holds_its_values_and_indices() {
        const WIDTH: usize = 2;
        const F32_BYTES: usize = 4;
        const K: usize = 3;
        let index = size_of::<u32>();
        assert_eq!(
            ReduceCarrier::Values.bytes(WIDTH, F32_BYTES),
            WIDTH * F32_BYTES
        );
        assert_eq!(
            ReduceCarrier::Indexed.bytes(WIDTH, F32_BYTES),
            WIDTH * (F32_BYTES + index)
        );
        assert_eq!(
            ReduceCarrier::TopValues { k: K }.bytes(WIDTH, F32_BYTES),
            K * WIDTH * F32_BYTES
        );
        assert_eq!(
            ReduceCarrier::TopK { k: K }.bytes(WIDTH, F32_BYTES),
            K * WIDTH * (F32_BYTES + index)
        );
    }

    /// Only a mean divides by the count.
    #[test]
    fn only_a_mean_divides_by_the_count() {
        assert!(ReduceOperation::Mean.divides_by_count());
        assert!(!ReduceOperation::Sum.divides_by_count());
    }

    /// No carrier but a top-k's bounds the line.
    #[test]
    fn a_line_is_unbounded_but_for_a_top_k() {
        const LANE_SLOTS: usize = 32;
        assert_eq!(
            ReduceOperation::Sum.widest_line_at_lane_slots(LANE_SLOTS),
            None
        );
    }

    /// A top-k's line spreads its lane slots over its `k` values.
    #[test]
    fn a_top_k_line_spreads_its_slots_over_k() {
        const LANE_SLOTS: usize = 32;
        const K: usize = 4;
        assert_eq!(
            ReduceOperation::TopK { k: K }.widest_line_at_lane_slots(LANE_SLOTS),
            Some(LANE_SLOTS / K)
        );
    }

    /// Past the unroll budget, a top-k's lists live in local memory, so its line is one value.
    #[test]
    fn a_top_k_past_the_unroll_budget_reads_one_value_a_line() {
        const LANE_SLOTS: usize = 32;
        let k = TOP_K_UNROLL_BUDGET;
        assert_eq!(
            ReduceOperation::ArgTopK { k }.widest_line_at_lane_slots(LANE_SLOTS),
            Some(1)
        );
    }

    /// A top-k keeping no value has no line: its documented panic.
    #[test]
    #[should_panic]
    fn a_top_k_keeping_nothing_panics() {
        const LANE_SLOTS: usize = 32;
        ReduceOperation::TopK { k: 0 }.widest_line_at_lane_slots(LANE_SLOTS);
    }
}
