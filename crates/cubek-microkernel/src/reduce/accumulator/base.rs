use cubecl::tile::{Carrier, Monoid};
use cubecl::{self as cubecl, cube, prelude::*};

/// What a reduction accumulates a unit's lines into, `N` lanes of `Acc`: how a line read at a position
/// joins it, how the planes of a cube meet, and the lines it keeps, which the microkernels write
/// out and store. Which monoid combines it is its [`Carrier`]'s. It reads and writes no Tile: the
/// microkernels do.
#[cube]
pub trait ReduceAccumulator<Acc: Numeric, N: Size>:
    Carrier + CubeType<ExpandType: Assign + Copy>
{
    /// `line` accumulated into `acc`, the line read `line_index` lines into the row, `row_lanes`
    /// of its lanes holding one row's values (all of them along the row, one across rows), under
    /// `monoid`. In place, since it runs once a line read: a carrier returned anew is copied back
    /// slot by slot, which for a top-k's `K` slots costs more than the accumulation. Each call's
    /// `line_index` is above every earlier call's on `acc`, as a unit's reads along a row are, so
    /// a tie keeps what `acc` holds.
    fn accumulate(
        acc: &mut Self,
        line: Vector<Acc, N>,
        line_index: usize,
        #[comptime] row_lanes: usize,
        #[comptime] monoid: Monoid,
    );

    /// The partials of a cube's `planes` planes, each already met across its own lanes, met in
    /// shared memory, one total for each of the `columns` lines of rows its units lie across:
    /// the unit of each plane and column where `stages` is set writes its partial to that
    /// column's slot of its plane, the cube meets, each column's slots are combined, and the
    /// cube meets again so the slots may be written anew. The first plane's units are left with
    /// their `column`'s total; an implementation may leave the other units without it, where
    /// combining costs more than one plane should spend.
    fn across_planes(
        acc: Self,
        column: usize,
        stages: bool,
        #[comptime] planes: usize,
        #[comptime] columns: usize,
        #[comptime] monoid: Monoid,
    ) -> Self;

    /// Line `slot` of what it keeps: its one line for the values themselves and an
    /// [`Indexed`](cubecl::tile::Indexed), line `slot` of a top-k's `k`, best first.
    fn value_line(acc: &Self, slot: usize) -> Vector<Acc, N>;

    /// The indices beside line `slot` ([`value_line`](ReduceAccumulator::value_line)); zero where
    /// the carrier keeps none ([`ReduceCarrier::carries_indices`](crate::reduce::ReduceCarrier::carries_indices)).
    fn index_line(acc: &Self, slot: usize) -> Vector<u32, N>;

    /// `acc` with line `slot` set to `value`, and to `index` beside it where the carrier carries
    /// indices: what a carrier stored line by line is loaded back from.
    fn set_line(acc: &mut Self, slot: usize, value: Vector<Acc, N>, index: Vector<u32, N>);

    /// `acc`, whose lanes hold one row, its lanes combined under `monoid` into the row's own
    /// carrier, held in every lane: what a row along the reduced dimension writes.
    fn one_row(acc: Self, #[comptime] monoid: Monoid) -> Self;
}
