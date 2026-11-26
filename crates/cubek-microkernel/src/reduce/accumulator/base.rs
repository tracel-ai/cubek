use cubecl::tile::{Carrier, Monoid};
use cubecl::{self as cubecl, cube, prelude::*};

/// What a reduction accumulates a unit's lines into, `N` lanes of `Acc`: how a line read at a position
/// joins it, and the lines it keeps, which the microkernels write out, store and stage. Which monoid
/// combines it is its [`Carrier`]'s. It reads and writes no memory: the microkernels do.
#[cube]
pub trait Accumulable<Acc: Numeric, N: Size>:
    Carrier + CubeType<ExpandType: Assign + Copy>
{
    /// `line` accumulated into `acc`, the line read `line_index` lines into the row, `row_lanes`
    /// of its lanes holding one row's values (all of them along the row, one across rows), under
    /// `monoid`. In place, since it runs once a line read: a carrier returned anew is copied back
    /// slot by slot, which for a top-k's `K` slots costs more than the accumulation.
    ///
    /// # Assumptions
    ///
    /// - `line_index` is above that of every earlier call on `acc`, as a unit's reads along a row
    ///   are.
    ///
    /// # Guarantees
    ///
    /// - A tie keeps what `acc` holds: of equal values, the one read first.
    fn accumulate(
        acc: &mut Self,
        line: Vector<Acc, N>,
        line_index: usize,
        #[comptime] row_lanes: usize,
        #[comptime] monoid: Monoid,
    );

    /// Line `slot` of what it keeps: its one line for the values themselves and an
    /// [`Indexed`](cubecl::tile::Indexed), line `slot` of a top-k's `k`, best first.
    ///
    /// # Assumptions
    ///
    /// - `slot` is below the lines the carrier keeps.
    fn value_line(acc: &Self, slot: usize) -> Vector<Acc, N>;

    /// The indices beside line `slot` ([`value_line`](Accumulable::value_line)); zero where
    /// the carrier keeps none ([`ReduceCarrier::carries_indices`](crate::reduce::ReduceCarrier::carries_indices)).
    ///
    /// # Assumptions
    ///
    /// - `slot` is below the lines the carrier keeps.
    fn index_line(acc: &Self, slot: usize) -> Vector<u32, N>;

    /// `acc` with line `slot` set to `value`, and to `index` beside it where the carrier carries
    /// indices: what a carrier stored line by line is loaded back from.
    ///
    /// # Assumptions
    ///
    /// - `slot` is below the lines the carrier keeps.
    fn set_line(acc: &mut Self, slot: usize, value: Vector<Acc, N>, index: Vector<u32, N>);

    /// `acc`, whose lanes hold one row, its lanes combined under `monoid` into the row's own
    /// carrier, the same in every lane: what a row along the reduced dimension writes.
    fn one_row(acc: Self, #[comptime] monoid: Monoid) -> Self;
}
