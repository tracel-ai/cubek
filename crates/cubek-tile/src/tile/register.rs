//! The register-resident accumulator ([`RegisterData`]), the software leaf's counterpart to
//! a cmma fragment.

use cubecl::prelude::*;

use crate::*;

/// An `mr × nr` block of accumulators living in registers, the [`Leaf::Register`] encoding of
/// a [`PlaneTile`].
///
/// Vectorization stays *inside*: `data` is scalar-typed by Rust-side erasure, exactly as
/// [`Store`](crate::Store)'s buffer is, and the real element is `Vector<T, vector_size>`. Every
/// operation re-groups it with `size!` at the width held here, so the lines never reach a
/// generic — a `Vector<T, W>` field would put `W` on [`PlaneTile`] and from there on
/// [`TileKind`] and [`Tile`], for a width that is a storage detail of one leaf.
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
    /// `mr * nr` lines of `vector_size`, scalar-typed by erasure (see the type docs).
    pub(crate) data: Array<T>,
    /// Physical line size of `data`; comptime so `size!` can read it.
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
        let nr = comptime!(n / vector_size);
        RegisterData::<T> {
            data: Array::<T>::new(comptime!(m * n)),
            vector_size,
            mr: m,
            nr,
            lane_share,
        }
    }

    /// The block re-grouped into its real `Vector<T, W>` element.
    pub(crate) fn lines<W: Size>(&self) -> &[Vector<T, W>] {
        self.data.as_vectorized().with_vector_size::<W>()
    }

    /// The mutable twin of [`lines`](Self::lines).
    pub(crate) fn lines_mut<W: Size>(&mut self) -> &mut [Vector<T, W>] {
        self.data.as_vectorized_mut().with_vector_size_mut::<W>()
    }

    pub(crate) fn zero(&mut self) {
        let size!(W) = comptime!(self.vector_size);
        let count = comptime!(self.mr * self.nr);
        let lines = self.lines_mut::<W>();
        // Indexed rather than iterated: `#[cube]` traces this loop, and the expansion has no
        // `iter_mut` to trace through.
        #[allow(clippy::needless_range_loop)]
        #[unroll]
        for i in 0..count {
            lines[i] = Vector::<T, W>::cast_from(0u32);
        }
    }

    /// Write the block into `mem`'s window, casting down to its element — the same manual,
    /// row-major store the mma fragment does, over lines instead of lane positions.
    ///
    /// Under [`LaneShare::Partial`] each lane holds only part of every cell, so the block is not
    /// the answer until the plane is summed: combine first, then let one lane write. This is
    /// what [`AccumulateView::commit`] does for the memory-backed leaf, and skipping it is
    /// every lane writing its own fraction over the last.
    pub(crate) fn store_cast_window<Out: Numeric>(&self, mem: &mut MemData<Out>) {
        let size!(W) = comptime!(self.vector_size);
        let vw = comptime!(self.vector_size);
        // `row_stride` counts scalars, but the window is indexed in *lines* (its offset is a
        // line offset, and the buffer's real element is `Vector<Out, vector_size>`). Write whole
        // lines at line indices — stepping by scalars here spreads each row `vw` times too far.
        let line_stride = mem.row_stride() / comptime!(vw as u32);
        let window = mem.window_slice_mut();
        let out_lines = window.as_vectorized_mut().with_vector_size_mut::<W>();
        let lines = self.lines::<W>();

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
                            Vector::<Out, W>::cast_from(lines[comptime!(i * self.nr + n)]);
                    }
                }
            }
            LaneShare::Partial =>
            {
                #[unroll]
                for i in 0..comptime!(self.mr) {
                    #[unroll]
                    for n in 0..comptime!(self.nr) {
                        let combined = plane_sum(lines[comptime!(i * self.nr + n)]);
                        if UNIT_POS_X == 0 {
                            let offset = (i as u32) * line_stride + comptime!(n as u32);
                            out_lines[offset as usize] = Vector::<Out, W>::cast_from(combined);
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
    pub(crate) fn mma<EL: Numeric, ER: Numeric>(&mut self, lhs: &Tile<EL>, rhs: &Tile<ER>) {
        let pack_l = lhs.quant_pack();
        let pack_r = rhs.quant_pack();
        let vw = rhs.vector_size();
        let lw = lhs.vector_size();
        comptime!(assert!(
            pack_l == 0 && pack_r == 0,
            "RegisterData::mma: a quantized operand against a promoted accumulator is not wired \
             yet — the memory-backed leaf serves those (it dequantizes per read)"
        ));
        comptime!(assert!(
            vw == self.vector_size,
            "RegisterData::mma: the block's lines must match the rhs's"
        ));

        let size!(W) = comptime!(self.vector_size);
        let size!(L) = lw;
        let kc = comptime!(rhs.space.extent_at(rhs.space.rank() - 2));
        let (mr, nr) = comptime!((self.mr, self.nr));

        let lhs_mat = lhs.matrix_transparent::<EL, L, L>(0usize);
        let rhs_mat = rhs.matrix_transparent::<ER, W, W>(0usize);
        // Matches the memory-backed leaf's cap: past it, hundreds of inlined cells
        // overflow the optimizer's recursive block pass.
        let unroll = comptime!(mr * nr <= 64);
        let c = self.lines_mut::<W>();

        for p in 0..kc {
            let mut b = Array::<Vector<T, W>>::new(nr);
            #[unroll(unroll)]
            for n in 0..nr {
                b[n] = Vector::<T, W>::cast_from(rhs_mat.read((p as u32, n as u32)));
            }
            #[unroll(unroll)]
            for i in 0..mr {
                let lhs_line = lhs_mat.read((i as u32, (p / lw) as u32));
                let a = Vector::<T, W>::cast_from(lhs_line.extract(p % lw));
                #[unroll(unroll)]
                for n in 0..nr {
                    c[i * nr + n] = fma(a, b[n], c[i * nr + n]);
                }
            }
        }
    }
}
