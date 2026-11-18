use crate::reduce::{ReduceLayout, ReduceOperation, ReduceWrites};
use cubecl::tile::{Carrier, Monoid, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// What a reduction accumulates a unit's lines into, `N` lanes of `Acc`: how a line read at a position
/// joins it, and how what it holds is written out. Which monoid combines it, and how the units
/// holding partials meet, is its [`Carrier`]'s.
#[cube]
pub trait ReduceAccumulator<Acc: Numeric, N: Size>:
    Carrier + CubeType<ExpandType: Assign + Copy>
{
    /// `line` accumulated into `acc`, the line read `line_index` lines into the row, under
    /// `monoid`. In place, since it runs once a line read: a carrier returned anew is copied back
    /// slot by slot, which for a top-k's `K` slots costs more than the accumulation.
    fn accumulate(
        acc: &mut Self,
        line: Vector<Acc, N>,
        line_index: usize,
        #[comptime] layout: ReduceLayout,
        #[comptime] monoid: Monoid,
    );

    /// The partials of a cube's `planes` planes, each already met across its own lanes, met in
    /// shared memory: every plane's first lane writes its partial to a slot of its plane, the
    /// cube meets, every unit combines every slot, and the cube meets again so the slots may be
    /// written anew. Every unit of the cube is left with the total.
    fn across_planes(acc: Self, #[comptime] planes: usize, #[comptime] monoid: Monoid) -> Self;

    /// `acc` stored as partial `part` of a row its cubes split, for the cube that merges them
    /// ([`load`](ReduceAccumulator::load)): its lines, one for the values and for an [`Indexed`], `k` for a
    /// top-k, written that many lines times `part` into `values`, in `Acc`, and its indices at
    /// the same lines of `indices`, in `u32`: a carrier that carries indices is given them, and no
    /// other is.
    ///
    /// # Panics
    ///
    /// Where the carrier carries indices and `indices` is none.
    fn store(acc: Self, values: &mut Tile, indices: &ComptimeOption<Tile>, part: usize);

    /// Partial `part`, as [`store`](ReduceAccumulator::store) left it.
    ///
    /// # Panics
    ///
    /// Where the carrier carries indices and `indices` is none.
    fn load(values: &Tile, indices: &ComptimeOption<Tile>, part: usize) -> Self;

    /// What `acc` holds written as `writes` says, its lanes reduced to one first where they
    /// hold one row ([`ReduceLayout::Along`]): to `out`'s first cell along the row, or its first
    /// line across rows (its first `k` of either, for a top-k). With [`ReduceWrites::Both`] the
    /// indices go to `indices` at the same places, in `Idx`; `indices` is given only then.
    /// `count` is the number of values reduced, which a mean divides by.
    ///
    /// # Panics
    ///
    /// Where `writes` asks for indices the carrier does not hold, or for both halves without
    /// `indices`.
    fn write<Out: Numeric, Idx: Numeric>(
        acc: Self,
        out: &mut Tile,
        indices: &ComptimeOption<Tile>,
        count: usize,
        #[comptime] layout: ReduceLayout,
        #[comptime] operation: ReduceOperation,
        #[comptime] writes: ReduceWrites,
    );
}
