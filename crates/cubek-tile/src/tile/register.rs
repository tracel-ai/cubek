//! The register-resident accumulator ([`RegisterData`]), the software leaf's counterpart to
//! a cmma fragment.

use cubecl::prelude::*;

use crate::{instruction::plane, *};

// The block's line width, a scope-registered size rather than a generic, as `MmaData` carries
// `NA`/`NL`/`NR`: `alloc` binds it with `register_size`, every op reads `Vector<T, RA>`, and it
// never reaches `PlaneTile` / `TileKind` / `Tile`.
//
// Allocate at the vector element: a scalar `Array::<T>` re-viewed as lines has nothing behind the
// reinterpret, and the CPU backend refuses a vectorized operand built that way.
define_size!(pub(crate) RA);

/// An `mr × nr` block of `RA`-wide accumulators living in registers, the software instruction's
/// encoding of a [`PlaneTile`].
///
/// The block exists so the software leaf can accumulate the way the hardware ones do: created by
/// [`block_accumulator`](Tile::block_accumulator) and passed in, it outlives a leaf call and meets
/// memory only on drain, so a deep contraction into `f16` never round-trips partials through it.
///
/// Its lines are the rhs's. Lined along the accumulator, a line is `RA` neighbouring cells; lined
/// along the contraction (a weight stored along `K`), it is `RA` partials of *one* cell
/// ([`fold`](Self::fold)), collapsed on drain so that sum, too, stays in `T` across the walk.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct RegisterData<T: Numeric> {
    /// `mr * nr` lines, each `Vector<T, RA>` (width registered in [`alloc`](Self::alloc)).
    pub(crate) data: Array<Vector<T, RA>>,
    /// Physical line width, the numeric twin of `RA`; comptime, for the line arithmetic.
    #[cube(comptime)]
    pub(crate) vector_size: usize,
    /// Contracted values a line holds of one cell: `1` where its lanes are neighbouring cells,
    /// the line width where they are that cell's partials. What [`vector_size`](Self::vector_size)
    /// means, not how wide it is.
    #[cube(comptime)]
    pub(crate) fold: usize,
    /// Rows in the block.
    #[cube(comptime)]
    pub(crate) mr: usize,
    /// Lines per row: the `n` extent divided by [`vector_size`](Self::vector_size), or `n` itself
    /// where every cell has a line of its own partials.
    #[cube(comptime)]
    pub(crate) nr: usize,
    /// The sink's matrix, the one this block was sized against. Carried rather than re-derived:
    /// a block that drains through a different grouping than it was allocated for writes its
    /// lines at coordinates the sink reads as something else.
    #[cube(comptime)]
    pub(crate) axes: MatrixAxes,
    /// Execution configuration for this register leaf.
    #[cube(comptime)]
    pub(crate) config: RegisterBlock,
    /// How this block's partials merge: the `⊕` it accumulates under, [`Sum`](Monoid::Sum) for a
    /// matmul. Stated where the block is built ([`Tile::block_accumulator`]), since comptime state
    /// cannot be set afterwards; read on drain with [`lanes`](Self::lanes), which says they exist.
    #[cube(comptime)]
    pub(crate) monoid: Monoid,
}

/// Bind the block width `RA` for the rest of the kernel's scope.
#[cube]
fn register_block_size(#[comptime] vector_size: usize) {
    intrinsic!(|scope| {
        scope.register_size::<RA>(vector_size);
    });
}

#[cube]
impl<T: Numeric> RegisterData<T> {
    /// An uninitialized `m × n` block at `vector_size`, its lines `fold` partials of one cell or,
    /// at a `fold` of one, `vector_size` neighbouring cells, in which case `n` must divide into
    /// whole lines: the leaf reads and writes nothing narrower.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn alloc(
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] axes: MatrixAxes,
        #[comptime] vector_size: usize,
        #[comptime] fold: usize,
        #[comptime] config: RegisterBlock,
        #[comptime] monoid: Monoid,
    ) -> RegisterData<T> {
        comptime!(assert!(
            fold == 1 || fold == vector_size,
            "RegisterData::alloc: a line holds the {fold} partials of one cell, so it is that wide, \
             not {vector_size}"
        ));
        comptime!(assert!(
            vector_size > 0 && (fold > 1 || n.is_multiple_of(vector_size)),
            "RegisterData::alloc: n ({n}) must be a whole number of {vector_size}-wide lines"
        ));
        register_block_size(vector_size);
        let nr = comptime!(if fold > 1 { n } else { n / vector_size });
        RegisterData::<T> {
            data: Array::<Vector<T, RA>>::new(comptime!(m * nr)),
            vector_size,
            fold,
            mr: m,
            nr,
            axes,
            config,
            monoid,
        }
    }

    pub(crate) fn zero(&mut self) {
        self.init(T::from_int(0));
    }

    pub(crate) fn init(&mut self, val: T) {
        let count = comptime!(self.mr * self.nr);
        #[allow(clippy::needless_range_loop)]
        #[unroll]
        for i in 0..count {
            self.data[i] = Vector::<T, RA>::cast_from(val);
        }
    }

    /// Add `src`'s block into this one, casting each line up to `T`.
    ///
    /// The promotion a narrow leaf drains through. A block accumulating in the operands' own
    /// element reaches a device's packed instruction (`hfma2` on `f16`, twice the arithmetic of
    /// `f32`), but `f16` counts integers exactly to 2048, so a long reduction in it stalls.
    ///
    /// Draining the narrow block into a wide one **inside** the walk keeps both: the packed
    /// instruction on the leaf's own steps and the wide sum across them, with the error bounded by
    /// the leaf's depth rather than by the whole contraction.
    ///
    /// Both blocks are the same `mr × nr` grid at the same fold, since the narrow one is opened
    /// against the same sink; the add is line-wise and needs no view of either.
    pub(crate) fn add_cast_from<S: Numeric>(&mut self, src: &RegisterData<S>) {
        comptime!(assert!(
            self.mr == src.mr && self.nr == src.nr && self.fold == src.fold,
            "RegisterData::add_cast_from: the blocks are {}x{} fold {} and {}x{} fold {}; a \
             promotion adds line for line and both are opened against one sink",
            self.mr,
            self.nr,
            self.fold,
            src.mr,
            src.nr,
            src.fold
        ));
        let count = comptime!(self.mr * self.nr);
        #[allow(clippy::needless_range_loop)]
        #[unroll]
        for i in 0..count {
            self.data[i] += Vector::<T, RA>::cast_from(src.data[i]);
        }
    }

    /// Multiply every partial this block holds by `factor`: one multiply per slot, which is what
    /// a scale covering the whole block costs when it is applied to the sum rather than to each
    /// term of it.
    pub(crate) fn scale(&mut self, factor: T) {
        let count = comptime!(self.mr * self.nr);
        let line = Vector::<T, RA>::cast_from(factor);
        #[allow(clippy::needless_range_loop)]
        #[unroll]
        for i in 0..count {
            self.data[i] *= line;
        }
    }
}

#[cube]
impl<T: Numeric> RegisterData<T> {
    /// Write the block into `mem`'s window, casting down to its element: the same manual,
    /// row-major store the mma fragment does, over lines instead of lane positions.
    ///
    /// Under a folded [`LaneShare`] each lane holds only part of every cell, so fold first, then
    /// let one lane write, as [`AccumulateView::commit`] does; skipping it is every lane writing
    /// its fraction over the last. The share is `mem`'s (descended every level), not the block's.
    ///
    /// A write that folds ([`Write::Accumulate`]) rather than replaces adds one more election:
    /// lanes that repeat each other's work would each add the same contribution.
    ///
    /// The write goes through the sink's masked matrix view, as [`AccumulateView::commit`] does: a
    /// block is sized to the leaf, so it may overhang the real extent, and the lines past the edge
    /// belong to the next row. The mask is a comptime flag, so a block that fits stores straight.
    ///
    /// A line of one cell's partials ([`fold`](Self::fold)) is collapsed after the lanes are
    /// combined and lands in a scalar cell; a line of neighbouring cells lands as it is.
    pub(crate) fn store_cast_window<Out: Numeric>(
        &self,
        mem: &mut MemData<Out>,
        #[comptime] space: Space,
    ) {
        if comptime!(self.fold > 1) {
            let size!(A) = 1usize;
            self.drain::<Out, A>(mem, space);
        } else {
            self.drain::<Out, RA>(mem, space);
        }
    }

    /// [`store_cast_window`](Self::store_cast_window) into a sink addressed in `A`-wide cells:
    /// the block's own width, or scalar where a line folds into one cell.
    fn drain<Out: Numeric, A: Size>(&self, mem: &mut MemData<Out>, #[comptime] space: Space) {
        let mem_write = comptime!(mem.access.write);
        let lanes = comptime!(mem.lanes);
        let fold = comptime!(self.fold);
        let monoid = comptime!(self.monoid);
        // Bounded by the window extent.
        let mut sink = mem.matrix_mut::<A>(0usize, comptime!(self.axes), space);

        // Split comptime rather than branching per line: a value-producing `match` plus a lane
        // guard emits a binding the CPU backend cannot resolve ("Value should have been declared
        // before"), and a `Whole` share (every CPU, whose planes are one lane) needs neither.
        match comptime!(Drain::of(lanes, mem_write)) {
            Drain::EachLane =>
            {
                #[unroll]
                for i in 0..comptime!(self.mr) {
                    #[unroll]
                    for n in 0..comptime!(self.nr) {
                        let cell =
                            cell::<T, Out, A>(self.data[comptime!(i * self.nr + n)], fold, monoid);
                        sink.write(((i as u32).runtime(), (n as u32).runtime()), cell);
                    }
                }
            }
            Drain::LaneZero =>
            {
                #[unroll]
                for i in 0..comptime!(self.mr) {
                    #[unroll]
                    for n in 0..comptime!(self.nr) {
                        let cell =
                            cell::<T, Out, A>(self.data[comptime!(i * self.nr + n)], fold, monoid);
                        if UNIT_POS_X == 0 {
                            sink.write(((i as u32).runtime(), (n as u32).runtime()), cell);
                        }
                    }
                }
            }
            Drain::PlaneFold =>
            {
                #[unroll]
                for i in 0..comptime!(self.mr) {
                    #[unroll]
                    for n in 0..comptime!(self.nr) {
                        let combined = plane::broadcast::<Vector<T, RA>>(
                            self.data[comptime!(i * self.nr + n)],
                            monoid,
                        );
                        let cell = cell::<T, Out, A>(combined, fold, monoid);
                        if UNIT_POS_X == 0 {
                            sink.write(((i as u32).runtime(), (n as u32).runtime()), cell);
                        }
                    }
                }
            }
            Drain::GroupFold { fold_mask } =>
            {
                #[unroll]
                for i in 0..comptime!(self.mr) {
                    #[unroll]
                    for n in 0..comptime!(self.nr) {
                        let combined = plane::group::<T, RA>(
                            self.data[comptime!(i * self.nr + n)],
                            comptime!(fold_mask),
                            monoid,
                        );
                        let cell = cell::<T, Out, A>(combined, fold, monoid);
                        let lane_in_group = UNIT_POS_X & comptime!(fold_mask as u32);
                        if lane_in_group == 0 {
                            sink.write(((i as u32).runtime(), (n as u32).runtime()), cell);
                        }
                    }
                }
            }
        }
    }
}

/// What a drained line lands as: the line itself, cast, where its lanes are neighbouring cells;
/// their fold under `monoid`, cast, where they are `fold` partials of one cell.
#[cube]
fn cell<T: Numeric, Out: Numeric, A: Size>(
    line: Vector<T, RA>,
    #[comptime] fold: usize,
    #[comptime] monoid: Monoid,
) -> Vector<Out, A> {
    if comptime!(fold > 1) {
        Vector::<Out, A>::cast_from(Monoid::fold_lanes::<T, RA>(line, fold, monoid))
    } else {
        Vector::<Out, A>::cast_from(line)
    }
}
