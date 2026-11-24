use cubecl::tile::Monoid;

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

    /// The lines it holds: one of values a slot and, where it carries indices, one of them beside
    /// each.
    pub const fn lines(self) -> usize {
        match self.carries_indices() {
            true => 2 * self.slots(),
            false => self.slots(),
        }
    }

    /// Bytes one carrier of `width` lanes holds, its values `value_bytes` each and, where it
    /// carries indices, a `u32` beside each.
    pub const fn bytes(self, width: usize, value_bytes: usize) -> usize {
        let lane = match self.carries_indices() {
            true => value_bytes + size_of::<u32>(),
            false => value_bytes,
        };
        self.slots() * width * lane
    }
}

impl ReduceCarrier {
    /// The values it keeps of a row.
    const fn slots(self) -> usize {
        match self {
            Self::TopK { k } | Self::TopValues { k } => k,
            Self::Values | Self::Indexed => 1,
        }
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
    /// keeping `lane_slots` top-k slots across a line's lanes. A top-k keeps `k` values, and their
    /// indices where it writes them, for every lane of a line: past `lane_slots` they spill out of registers, so the line
    /// narrows, down to one value for a list longer than `lane_slots`.
    ///
    /// # Panics
    ///
    /// Where a top-k keeps `k = 0` values.
    pub const fn widest_line_at_lane_slots(self, lane_slots: usize) -> Option<usize> {
        match self {
            Self::TopK { k } | Self::ArgTopK { k } => Some(lines_at_least_one(lane_slots / k)),
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
    /// index wins a tie, so a max or a min keeps a NaN. Past a row shorter than `k`, a top-k
    /// writes what an empty slot holds: `-inf` for a value and `u32::MAX` for an index.
    ///
    /// # Panics
    ///
    /// Where `row` is empty and the operation keeps one of its values: a max, a min, their
    /// magnitudes and their indices.
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
            Self::TopK { k } => best_k(row, k, |(value, _)| value, f32::NEG_INFINITY),
            Self::ArgTopK { k } => best_k(row, k, |(_, index)| index as f32, u32::MAX as f32),
        }
    }
}

/// The `k` largest values of `row`, best first, each written as `cell` writes it, and `past_row`
/// for each of the `k` the row is too short to hold.
#[cfg(feature = "cpu-reference")]
fn best_k(row: &[f32], k: usize, cell: fn((f32, usize)) -> f32, past_row: f32) -> Vec<f32> {
    let mut cells: Vec<f32> = ranked(row, Rank::Largest)
        .into_iter()
        .take(k)
        .map(cell)
        .collect();
    cells.resize(k, past_row);
    cells
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

/// `lines`, or one where it is none: a line holds at least one value.
const fn lines_at_least_one(lines: usize) -> usize {
    match lines {
        0 => 1,
        lines => lines,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A carrier of values and indices holds a line of each a slot.
    #[test]
    fn a_carrier_of_indices_holds_two_lines_a_slot() {
        const K: usize = 3;
        assert_eq!(ReduceCarrier::TopK { k: K }.lines(), 2 * K);
    }

    /// A carrier of values alone holds one.
    #[test]
    fn a_carrier_of_values_holds_one_line_a_slot() {
        const K: usize = 3;
        assert_eq!(ReduceCarrier::TopValues { k: K }.lines(), K);
    }

    /// A test of each property of `$operation` its table row states: its monoid, its read map
    /// and its carrier.
    macro_rules! table_row {
        ($name:ident: $operation:expr => $monoid:expr, $map:expr, $carrier:expr) => {
            mod $name {
                use super::*;

                #[test]
                fn monoid() {
                    assert_eq!($operation.monoid(), $monoid);
                }

                #[test]
                fn read_map() {
                    assert_eq!($operation.read_map(), $map);
                }

                #[test]
                fn carrier() {
                    assert_eq!($operation.carrier(), $carrier);
                }
            }
        };
    }

    table_row!(sum: ReduceOperation::Sum => Monoid::Sum, ReadMap::AsRead, ReduceCarrier::Values);
    table_row!(prod: ReduceOperation::Prod => Monoid::Prod, ReadMap::AsRead, ReduceCarrier::Values);
    table_row!(mean: ReduceOperation::Mean => Monoid::Sum, ReadMap::AsRead, ReduceCarrier::Values);
    table_row!(max: ReduceOperation::Max => Monoid::Max, ReadMap::AsRead, ReduceCarrier::Values);
    table_row!(min: ReduceOperation::Min => Monoid::Min, ReadMap::AsRead, ReduceCarrier::Values);
    table_row!(max_abs: ReduceOperation::MaxAbs => Monoid::Max, ReadMap::Abs, ReduceCarrier::Values);
    table_row!(any: ReduceOperation::Any => Monoid::Max, ReadMap::NonZero, ReduceCarrier::Values);
    table_row!(all: ReduceOperation::All => Monoid::Min, ReadMap::NonZero, ReduceCarrier::Values);
    table_row!(arg_max: ReduceOperation::ArgMax => Monoid::Max, ReadMap::AsRead, ReduceCarrier::Indexed);
    table_row!(arg_min: ReduceOperation::ArgMin => Monoid::Min, ReadMap::AsRead, ReduceCarrier::Indexed);
    table_row!(top_5: ReduceOperation::TopK { k: 5 } => Monoid::Max, ReadMap::AsRead, ReduceCarrier::TopValues { k: 5 });
    table_row!(arg_top_5: ReduceOperation::ArgTopK { k: 5 } => Monoid::Max, ReadMap::AsRead, ReduceCarrier::TopK { k: 5 });
    table_row!(top_1: ReduceOperation::TopK { k: 1 } => Monoid::Max, ReadMap::AsRead, ReduceCarrier::Values);
    table_row!(arg_top_1: ReduceOperation::ArgTopK { k: 1 } => Monoid::Max, ReadMap::AsRead, ReduceCarrier::Indexed);

    /// A test that `$operation` writes what `$writes` names.
    macro_rules! writes {
        ($name:ident: $operation:expr => $writes:expr) => {
            #[test]
            fn $name() {
                assert_eq!($operation.writes(), $writes);
            }
        };
    }

    writes!(an_arg_max_writes_indices: ReduceOperation::ArgMax => ReduceWrites::Indices);
    writes!(an_arg_min_writes_indices: ReduceOperation::ArgMin => ReduceWrites::Indices);
    writes!(an_arg_top_k_writes_indices: ReduceOperation::ArgTopK { k: 2 } => ReduceWrites::Indices);
    writes!(a_sum_writes_values: ReduceOperation::Sum => ReduceWrites::Values);
    writes!(a_max_writes_values: ReduceOperation::Max => ReduceWrites::Values);
    writes!(a_top_k_writes_values: ReduceOperation::TopK { k: 2 } => ReduceWrites::Values);
    writes!(an_any_writes_values: ReduceOperation::Any => ReduceWrites::Values);

    /// A top-k writes `k` cells a row.
    #[test]
    fn a_top_k_writes_k_cells() {
        const K: usize = 7;
        assert_eq!(ReduceOperation::TopK { k: K }.written_per_row(), K);
    }

    /// An arg top-k writes `k` cells a row.
    #[test]
    fn an_arg_top_k_writes_k_cells() {
        const K: usize = 7;
        assert_eq!(ReduceOperation::ArgTopK { k: K }.written_per_row(), K);
    }

    /// Every other operation writes one cell a row.
    #[test]
    fn an_arg_max_writes_one_cell() {
        assert_eq!(ReduceOperation::ArgMax.written_per_row(), 1);
    }

    /// A test that a read past the edge of `$operation`'s row reads the identity of `$edge`, the
    /// monoid whose identity its map leaves as it is, so the total stays unchanged.
    macro_rules! edge_reads {
        ($name:ident: $operation:expr => $edge:expr) => {
            #[test]
            fn $name() {
                assert_eq!($operation.edge_monoid(), $edge);
            }
        };
    }

    edge_reads!(a_sum_edge_reads_zero: ReduceOperation::Sum => Monoid::Sum);
    edge_reads!(a_mean_edge_reads_zero: ReduceOperation::Mean => Monoid::Sum);
    edge_reads!(a_max_abs_edge_reads_zero: ReduceOperation::MaxAbs => Monoid::Sum);
    edge_reads!(an_any_edge_reads_zero: ReduceOperation::Any => Monoid::Sum);
    edge_reads!(a_prod_edge_reads_one: ReduceOperation::Prod => Monoid::Prod);
    edge_reads!(an_all_edge_reads_one: ReduceOperation::All => Monoid::Prod);
    edge_reads!(a_max_edge_reads_its_identity: ReduceOperation::Max => Monoid::Max);
    edge_reads!(an_arg_max_edge_reads_its_identity: ReduceOperation::ArgMax => Monoid::Max);
    edge_reads!(a_top_k_edge_reads_its_identity: ReduceOperation::TopK { k: 3 } => Monoid::Max);
    edge_reads!(an_arg_top_k_edge_reads_its_identity: ReduceOperation::ArgTopK { k: 3 } => Monoid::Max);
    edge_reads!(a_min_edge_reads_its_identity: ReduceOperation::Min => Monoid::Min);
    edge_reads!(an_arg_min_edge_reads_its_identity: ReduceOperation::ArgMin => Monoid::Min);

    /// Lines `WIDTH` wide of f32 values.
    const WIDTH: usize = 2;
    const F32_BYTES: usize = 4;

    /// A values carrier holds a line of values.
    #[test]
    fn a_values_carrier_holds_a_line() {
        assert_eq!(
            ReduceCarrier::Values.bytes(WIDTH, F32_BYTES),
            WIDTH * F32_BYTES
        );
    }

    /// An indexed carrier holds a `u32` beside each value.
    #[test]
    fn an_indexed_carrier_holds_an_index_beside_each_value() {
        assert_eq!(
            ReduceCarrier::Indexed.bytes(WIDTH, F32_BYTES),
            WIDTH * (F32_BYTES + size_of::<u32>())
        );
    }

    /// A top-values carrier holds `k` lines of values.
    #[test]
    fn a_top_values_carrier_holds_k_lines() {
        const K: usize = 3;
        assert_eq!(
            ReduceCarrier::TopValues { k: K }.bytes(WIDTH, F32_BYTES),
            K * WIDTH * F32_BYTES
        );
    }

    /// A top-k carrier holds `k` lines of values with their indices.
    #[test]
    fn a_top_k_carrier_holds_k_indexed_lines() {
        const K: usize = 3;
        assert_eq!(
            ReduceCarrier::TopK { k: K }.bytes(WIDTH, F32_BYTES),
            K * WIDTH * (F32_BYTES + size_of::<u32>())
        );
    }

    /// A mean divides by the count.
    #[test]
    fn a_mean_divides_by_the_count() {
        assert!(ReduceOperation::Mean.divides_by_count());
    }

    /// A sum does not divide by the count.
    #[test]
    fn a_sum_does_not_divide_by_the_count() {
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

    /// A top-k whose list is longer than its lane slots reads one value a line.
    #[test]
    fn a_top_k_longer_than_its_lane_slots_reads_one_value_a_line() {
        const LANE_SLOTS: usize = 32;
        const K: usize = 64;
        assert_eq!(
            ReduceOperation::ArgTopK { k: K }.widest_line_at_lane_slots(LANE_SLOTS),
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

    /// A row with a value that is not zero holds any.
    #[test]
    #[cfg(feature = "cpu-reference")]
    fn any_of_a_row_with_one_value_set_is_one() {
        assert_eq!(
            ReduceOperation::Any.reference_cells(&[0.0, 1.0, 0.0, 0.0]),
            vec![1.0]
        );
    }

    /// A row of zeros holds none.
    #[test]
    #[cfg(feature = "cpu-reference")]
    fn any_of_a_row_of_zeros_is_zero() {
        assert_eq!(
            ReduceOperation::Any.reference_cells(&[0.0, 0.0, 0.0, 0.0]),
            vec![0.0]
        );
    }

    /// A row with one zero does not hold all.
    #[test]
    #[cfg(feature = "cpu-reference")]
    fn all_of_a_row_with_one_value_set_is_zero() {
        assert_eq!(
            ReduceOperation::All.reference_cells(&[0.0, 1.0, 0.0, 0.0]),
            vec![0.0]
        );
    }

    /// A row with no zero holds all.
    #[test]
    #[cfg(feature = "cpu-reference")]
    fn all_of_a_row_of_ones_is_one() {
        assert_eq!(
            ReduceOperation::All.reference_cells(&[1.0, 1.0, 1.0, 1.0]),
            vec![1.0]
        );
    }

    /// A max keeps a NaN ahead of every number.
    #[test]
    #[cfg(feature = "cpu-reference")]
    fn a_max_keeps_a_nan() {
        let cells = ReduceOperation::Max.reference_cells(&[1.0, f32::NAN, 3.0]);
        assert!(cells[0].is_nan());
    }

    /// Of two equal values, the lower index wins.
    #[test]
    #[cfg(feature = "cpu-reference")]
    fn an_arg_max_tie_keeps_the_lower_index() {
        assert_eq!(
            ReduceOperation::ArgMax.reference_cells(&[1.0, 3.0, 3.0]),
            vec![1.0]
        );
    }

    /// A top-k keeps its `k` largest values, largest first.
    #[test]
    #[cfg(feature = "cpu-reference")]
    fn a_top_k_keeps_the_largest_first() {
        assert_eq!(
            ReduceOperation::TopK { k: 2 }.reference_cells(&[1.0, 4.0, 2.0, 3.0]),
            vec![4.0, 3.0]
        );
    }

    /// Past a row shorter than `k`, a top-k writes `-inf`.
    #[test]
    #[cfg(feature = "cpu-reference")]
    fn a_top_k_past_a_short_row_writes_the_lowest_value() {
        assert_eq!(
            ReduceOperation::TopK { k: 3 }.reference_cells(&[2.0]),
            vec![2.0, f32::NEG_INFINITY, f32::NEG_INFINITY]
        );
    }

    /// Past a row shorter than `k`, an arg top-k writes `u32::MAX`.
    #[test]
    #[cfg(feature = "cpu-reference")]
    fn an_arg_top_k_past_a_short_row_writes_the_largest_index() {
        assert_eq!(
            ReduceOperation::ArgTopK { k: 2 }.reference_cells(&[2.0]),
            vec![0.0, u32::MAX as f32]
        );
    }

    /// A max of an empty row has no value to keep: its documented panic.
    #[test]
    #[should_panic]
    #[cfg(feature = "cpu-reference")]
    fn a_max_of_an_empty_row_panics() {
        ReduceOperation::Max.reference_cells(&[]);
    }
}
