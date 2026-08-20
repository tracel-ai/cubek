//! The register-resident accumulator ([`RegisterData`]), the software leaf's counterpart to
//! a cmma fragment.

use cubecl::prelude::*;

use crate::{
    instruction::{
        mma::{
            packed::{PackedChunk, packed_chunk_walk},
            register::Block,
        },
        sum,
    },
    *,
};

// The block's line width, as a scope-registered size rather than a generic. `RA` names the
// vector element `data` is allocated at; `alloc` binds it to the promoting tile's width with
// `register_size`, and every op reads the block as `Vector<T, RA>`. This is exactly how
// `MmaData` carries `NA`/`NL`/`NR` — the width stays a storage detail of the leaf and never
// reaches `PlaneTile` / `TileKind` / `Tile` as a generic. (An earlier version allocated
// `Array::<T>` scalar and re-viewed it as lines; that reinterpret has nothing behind it and the
// CPU backend refuses a vectorized operand — allocate at the vector element instead.)
define_size!(pub(crate) RA);

/// An `mr × nr` block of `RA`-wide accumulators living in registers, the [`Leaf::Memory`]
/// encoding of a [`PlaneTile`].
///
/// The block exists so the software leaf can accumulate the way the hardware ones do. It used
/// to allocate its own inside the microkernel, which meant the accumulator could not outlive a
/// single call: a `K` walk that visits the leaf repeatedly round-tripped its partials through
/// the output's element between visits, so a deep contraction into `f16` lost precision it did
/// not have to. Created by [`promote`](Tile::promote) and passed in, it survives the whole walk
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
    /// Lines per row — the `n` extent divided by [`vector_size`](Self::vector_size).
    #[cube(comptime)]
    pub(crate) nr: usize,
    /// Whether each lane holds whole cells or a partial of them. Inherited from the memory this
    /// was promoted from, and only read on drain: the contraction is per-lane either way, but a
    /// partial is not the answer until the plane's lanes are summed.
    #[cube(comptime)]
    pub(crate) lane_share: LaneShare,
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
        }
    }

    pub(crate) fn zero(&mut self) {
        let count = comptime!(self.mr * self.nr);
        // Indexed rather than iterated: `#[cube]` traces this loop, and the expansion has no
        // `iter_mut` to trace through.
        #[allow(clippy::needless_range_loop)]
        #[unroll]
        for i in 0..count {
            self.data[i] = Vector::<T, RA>::cast_from(0u32);
        }
    }

    /// Write the block into `mem`'s window, casting down to its element — the same manual,
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
        // lines at line indices — stepping by scalars here spreads each row `vw` times too far.
        let line_stride = mem.row_stride() / comptime!(vw as u32);
        let window = mem.window_slice_mut();
        // The sink's storage is vectorized at the same width the block was promoted at, so re-view
        // it at `RA` too — this is a real vector store, not the reinterpret the block once did.
        let out_lines = window.as_vectorized_mut().with_vector_size_mut::<RA>();

        // Split comptime rather than branching per line: a value-producing `match` plus a
        // lane guard emits a binding the CPU backend cannot resolve ("Value should have been
        // declared before"), and a `Whole` share — every CPU, whose planes are one lane — has
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
                        let combined = plane_sum(self.data[comptime!(i * self.nr + n)]);
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
                        let combined = sum::group::<T, RA>(
                            self.data[comptime!(i * self.nr + n)],
                            comptime!(fold_mask),
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

    /// `self += lhs · rhs` over the block, one rank-1 update per scalar `K` step.
    ///
    /// The same contraction the memory-backed microkernel runs, minus the round trip: there the
    /// block is seeded from the sink and committed back on every visit, so a `K` walk that
    /// returns here repeatedly loses precision to the sink's element between visits. This one
    /// *is* the accumulator, so the partials stay in `T` until [`store_cast_window`] drains them.
    ///
    /// A packed-u32 lhs takes the chunked walk ([`mma_packed_lhs`](Self::mma_packed_lhs)); the
    /// other quantized forms stay unwired here and are served by the memory-backed leaf.
    pub(crate) fn mma<EL: Numeric, ER: Numeric>(
        &mut self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] out: Space,
    ) {
        let pack_l = lhs.quant_pack();
        let pack_r = rhs.quant_pack();
        let vw = rhs.vector_size();
        let lw = lhs.vector_size();
        comptime!(assert!(
            pack_r == 0,
            "RegisterData::mma: a quantized rhs against a promoted accumulator is not wired \
             yet — the memory-backed leaf serves those (it dequantizes per read)"
        ));
        if comptime!(pack_l > 1) {
            self.mma_packed_lhs(lhs, rhs, out);
        } else {
            comptime!(assert!(
                pack_l == 0,
                "RegisterData::mma: a native-i8 lhs against a promoted accumulator is not wired \
                 yet — the memory-backed leaf serves those (it dequantizes per read)"
            ));
            comptime!(assert!(
                vw == self.vector_size,
                "RegisterData::mma: the block's lines must match the rhs's"
            ));

            let size!(L) = lw;
            let kc = comptime!(rhs.space.extent_at(rhs.space.rank() - 2));
            let block = self.block(lhs, rhs);
            let (mr, nr) = comptime!((block.mr, block.nr));
            let unroll = comptime!(block.unroll);

            let lhs_mat = lhs.matrix_transparent::<EL, L, L>(0usize);
            // The rhs and the block share the width `RA` (asserted above, `vw == self.vector_size`).
            let rhs_mat = rhs.matrix_transparent::<ER, RA, RA>(0usize);

            for p in 0..kc {
                let mut b = Array::<Vector<T, RA>>::new(nr);
                #[unroll(unroll)]
                for n in 0..nr {
                    b[n] = Vector::<T, RA>::cast_from(rhs_mat.read((p as u32, n as u32)));
                }
                #[unroll(unroll)]
                for i in 0..mr {
                    let lhs_line = lhs_mat.read((i as u32, (p / lw) as u32));
                    let a = Vector::<T, RA>::cast_from(lhs_line.extract_dynamic(p % lw));
                    #[unroll(unroll)]
                    for n in 0..nr {
                        self.data[i * nr + n] = fma(a, b[n], self.data[i * nr + n]);
                    }
                }
            }
        }
    }

    /// The chunked packed-lhs walk into the resident block, the register twin of the
    /// memory-backed leaf's: same walk ([`packed_chunk_walk`]), a different accumulator behind
    /// it. Accumulating here instead of the memory leaf is what removes the per-chunk
    /// fold-and-write: the partials stay in these registers across the whole `K` walk and drain
    /// once, through [`store_cast_window`](Self::store_cast_window)'s lane-share fold.
    fn mma_packed_lhs<EL: Numeric, ER: Numeric>(
        &mut self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] out: Space,
    ) {
        let chunk = PackedChunk::new(lhs, rhs, out);
        comptime!(assert!(
            self.vector_size == chunk.acc_width(),
            "RegisterData::mma: this walk accumulates {}-wide cells, but the block holds {}-wide \
             ones",
            chunk.acc_width(),
            self.vector_size
        ));
        let size!(WPL) = comptime!(chunk.words());
        let size!(X) = comptime!(chunk.vw);
        let block = self.block(lhs, rhs);

        packed_chunk_walk::<T, EL, ER, WPL, RA, X>(&mut self.data, lhs, rhs, 0usize, chunk, block);
    }

    /// This block as a microkernel states it, masking with whichever operand it reads.
    fn block<EL: Numeric, ER: Numeric>(
        &self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
    ) -> comptime_type!(Block) {
        let lhs_masked = lhs.masks();
        let rhs_masked = rhs.masks();
        comptime!(Block::new(self.mr, self.nr, lhs_masked || rhs_masked))
    }
}
