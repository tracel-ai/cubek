//! The register-resident accumulator ([`RegisterData`]), the software leaf's counterpart to
//! a cmma fragment.

use cubecl::prelude::*;

use crate::{instruction::plane, *};

// The block's line width, as a scope-registered size rather than a generic. `RA` names the
// vector element `data` is allocated at; `alloc` binds it to the promoting tile's width with
// `register_size`, and every op reads the block as `Vector<T, RA>`. This is exactly how
// `MmaData` carries `NA`/`NL`/`NR`: the width stays a storage detail of the leaf and never
// reaches `PlaneTile` / `TileKind` / `Tile` as a generic. (An earlier version allocated
// `Array::<T>` scalar and re-viewed it as lines; that reinterpret has nothing behind it and the
// CPU backend refuses a vectorized operand: allocate at the vector element instead.)
define_size!(pub(crate) RA);

/// An `mr × nr` block of `RA`-wide accumulators living in registers, the software instruction's
/// encoding of a [`PlaneTile`].
///
/// The block exists so the software leaf can accumulate the way the hardware ones do. It used
/// to allocate its own inside the instruction, which meant the accumulator could not outlive a
/// single call: a `K` walk that visits the leaf repeatedly round-tripped its partials through
/// the output's element between visits, so a deep contraction into `f16` lost precision it did
/// not have to. Created by [`accumulate`](Tile::accumulate) and passed in, it survives the whole walk
/// and only meets memory at [`drain_cast_into`](Tile::drain_cast_into).
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct RegisterData<T: Numeric> {
    /// `mr * nr` lines, each `Vector<T, RA>` (width registered in [`alloc`](Self::alloc)).
    pub(crate) data: Array<Vector<T, RA>>,
    /// Physical line width, the numeric twin of `RA`; comptime, for the line arithmetic.
    #[cube(comptime)]
    pub(crate) vector_size: usize,
    /// Rows in the block.
    #[cube(comptime)]
    pub(crate) mr: usize,
    /// Lines per row: the `n` extent divided by [`vector_size`](Self::vector_size).
    #[cube(comptime)]
    pub(crate) nr: usize,
    /// Whether each lane holds whole cells or a partial of them. Inherited from the memory this
    /// was promoted from, and only read on drain: the contraction is per-lane either way, but a
    /// partial is not the answer until the plane's lanes are summed.
    #[cube(comptime)]
    pub(crate) lane_share: LaneShare,
    /// Execution configuration for this register leaf.
    #[cube(comptime)]
    pub(crate) config: RegisterBlock,
    /// How this block's partials merge: the `⊕` it accumulates under. Stated where the block is
    /// built ([`Tile::accumulate`]), because comptime state cannot be set afterwards, and read on
    /// drain next to [`lane_share`](Self::lane_share): that one says partials exist, this one says
    /// what combining them means. A matmul's is [`Sum`](LeafOp::Sum).
    #[cube(comptime)]
    pub(crate) fold: LeafOp,
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
    /// An uninitialized `m × n` block at `vector_size`. `n` must divide into whole lines; the
    /// leaf reads and writes nothing narrower.
    pub(crate) fn alloc(
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] vector_size: usize,
        #[comptime] lane_share: LaneShare,
        #[comptime] config: RegisterBlock,
        #[comptime] fold: LeafOp,
    ) -> RegisterData<T> {
        comptime!(assert!(
            vector_size > 0 && n.is_multiple_of(vector_size),
            "RegisterData::alloc: n ({n}) must be a whole number of {vector_size}-wide lines"
        ));
        register_block_size(vector_size);
        let nr = comptime!(n / vector_size);
        RegisterData::<T> {
            data: Array::<Vector<T, RA>>::new(comptime!(m * nr)),
            vector_size,
            mr: m,
            nr,
            lane_share,
            config,
            fold,
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

    /// Write the block into `mem`'s window, casting down to its element: the same manual,
    /// row-major store the mma fragment does, over lines instead of lane positions.
    ///
    /// Under a folded [`LaneShare`] each lane holds only part of every cell, so the block is not
    /// the answer until those lanes are combined: fold first, then let one of them write. This is
    /// what [`AccumulateView::commit`] does for the memory-backed leaf, and skipping it is
    /// every lane writing its own fraction over the last.
    pub(crate) fn store_cast_window<Out: Numeric>(&self, mem: &mut MemData<Out>) {
        let vw = comptime!(self.vector_size);
        // `row_stride` counts scalars, but the window is indexed in *lines* (its offset is a
        // line offset, and the buffer's real element is `Vector<Out, vector_size>`). Write whole
        // lines at line indices: stepping by scalars here spreads each row `vw` times too far.
        let line_stride = mem.row_stride() / comptime!(vw as u32);
        let window = mem.window_slice_mut();
        // The sink's storage is vectorized at the same width the block was promoted at, so re-view
        // it at `RA` too: this is a real vector store, not the reinterpret the block once did.
        let out_lines = window.as_vectorized_mut().with_vector_size_mut::<RA>();

        // Split comptime rather than branching per line: a value-producing `match` plus a
        // lane guard emits a binding the CPU backend cannot resolve ("Value should have been
        // declared before"), and a `Whole` share (every CPU, whose planes are one lane) has
        // no reason to emit either.
        match comptime!(self.lane_share) {
            LaneShare::Whole =>
            {
                #[unroll]
                for i in 0..comptime!(self.mr) {
                    #[unroll]
                    for n in 0..comptime!(self.nr) {
                        let offset = (i as u32) * line_stride + comptime!(n as u32);
                        out_lines[offset as usize] =
                            Vector::<Out, RA>::cast_from(self.data[comptime!(i * self.nr + n)]);
                    }
                }
            }
            LaneShare::Plane =>
            {
                #[unroll]
                for i in 0..comptime!(self.mr) {
                    #[unroll]
                    for n in 0..comptime!(self.nr) {
                        let combined = plane::broadcast::<Vector<T, RA>>(
                            self.data[comptime!(i * self.nr + n)],
                            comptime!(self.fold),
                        );
                        if UNIT_POS_X == 0 {
                            let offset = (i as u32) * line_stride + comptime!(n as u32);
                            out_lines[offset as usize] = Vector::<Out, RA>::cast_from(combined);
                        }
                    }
                }
            }
            LaneShare::Group { fold_mask } =>
            {
                #[unroll]
                for i in 0..comptime!(self.mr) {
                    #[unroll]
                    for n in 0..comptime!(self.nr) {
                        let combined = plane::group::<T, RA>(
                            self.data[comptime!(i * self.nr + n)],
                            comptime!(fold_mask),
                            comptime!(self.fold),
                        );
                        let lane_in_group = UNIT_POS_X & comptime!(fold_mask as u32);
                        if lane_in_group == 0 {
                            let offset = (i as u32) * line_stride + comptime!(n as u32);
                            out_lines[offset as usize] = Vector::<Out, RA>::cast_from(combined);
                        }
                    }
                }
            }
        }
    }
}
