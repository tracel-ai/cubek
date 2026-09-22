//! The split ending: the cross-split weights a split walk's teams merge
//! through, beside the [`publish`](crate::StreamFold::publish) that feeds them.

use cubecl::prelude::*;

use crate::*;

/// Split counts [`Tile::merge_splits`] unrolls its walk over the states for.
const MERGE_UNROLL: usize = 64;

#[cube]
impl<EA: Float> Tile<EA> {
    /// Merge a split fold's per-team running states into cross-split weights:
    /// per row, `self[r, t] = exp(m[r, t] - m*) / Σ_t l[r, t] · exp(m[r, t] - m*)`
    /// where `m* = max_t m[r, t]`.
    ///
    /// The normalizer is folded into the weights, so `Σ_t self[r, t] · acc[r, t, ·]`
    /// is the whole merge: a drain that contracts the weights as an operand has
    /// no value in hand left to scale.
    ///
    /// `self`, `m` and `l` span the score rows and `split`, laid out alike. The
    /// space says where `split` sits: outermost gives a team one contiguous run
    /// of rows, innermost lets the drain contract it as a matmul's `k`.
    ///
    /// `self` must be shared memory: the merge passes each cell's state to the rest of its row
    /// through the weights themselves, and `sync_cube` orders only the workgroup address space.
    /// `m` and `l` are read, never written, so they may be global.
    ///
    /// A fully-masked row gets weights of exactly zero, and a split that folded nothing published
    /// `(min, 0)` so it weighs zero on its own. A unit per cell where the cube has that many,
    /// syncing between its three passes; otherwise one unit per row, cyclic over the cube.
    ///
    /// Every unit of the cube calls it, and the caller syncs on both sides. A single split
    /// degenerates to the plain epilogue.
    pub fn merge_splits(&mut self, m: &Tile<EA>, l: &Tile<EA>, #[comptime] split: Axis) {
        // The cell path passes values *between* units: each cell parks its `m`, then its
        // `weight * l`, in the weights for its row's other cells to scan, ordered only by
        // `sync_cube`, which orders workgroup memory; it turns on `CUBE_DIM`, so the call needs it.
        let shared = self.is_shared();
        comptime!(assert!(
            shared,
            "merge_splits: the weights are what the cube merges through, so they must be \
             shared memory; a global buffer is not ordered by the sync_cube between the \
             passes. The states may still be global."
        ));
        let space = comptime!(self.space.clone());
        comptime!(assert!(
            space.contains(split),
            "merge_splits: {split:?} is not an axis of the weights"
        ));
        let splits = comptime!(space.extent(split));
        let rows = comptime!(space.tile_size() / splits);
        // The states can arrive partitioned differently from the weights (a
        // global buffer beside a shared-memory tile); only the layout has to agree.
        let m_space = comptime!(m.space.clone());
        let l_space = comptime!(l.space.clone());
        comptime!(assert!(
            m_space.laid_out_like(&space) && l_space.laid_out_like(&space),
            "merge_splits: the states must be laid out like the weights they merge into"
        ));

        // A step of `t` crosses whatever sits inside the split axis; a step of
        // the rows outside it crosses a whole slice of splits.
        let stride = comptime!(
            ((space.position(split) + 1)..space.rank())
                .map(|p| space.extent_at(p))
                .product::<usize>()
        );
        let slice = comptime!(splits * stride);

        let size!(W) = self.vector_size();
        let size!(WM) = m.vector_size();
        let size!(WL) = l.vector_size();
        let mf = m.flat::<WM>();
        let lf = l.flat::<WL>();
        let mut wf = self.flat_mut::<W>();

        let workers = CUBE_DIM as usize;
        let cells = comptime!(rows * splits);
        if cells <= workers {
            // A unit per cell, which is every merge a split walk closes: the states load at once,
            // and the row's maximum and normalizer are scans of the tile this merge writes. A unit
            // per row read each split in turn (26 µs for 32 splits on GP100, as long as the walk).
            let cell = UNIT_POS as usize;
            let owns = cell < cells;
            let t = (cell / stride) % splits;
            let base = cell - t * stride;
            let mut m_cell = EA::min_value();
            let mut l_cell = EA::from_int(0);
            if owns {
                m_cell = mf.read(cell).extract(0usize);
                l_cell = lf.read(cell).extract(0usize);
                wf.write(cell, Vector::cast_from(m_cell));
            }
            sync_cube();
            let mut weight = EA::from_int(0);
            if owns {
                let mut mstar = EA::min_value();
                #[unroll(splits <= MERGE_UNROLL)]
                for s in 0..splits {
                    mstar = max(mstar, wf.read(base + s * stride).extract(0usize));
                }
                weight = (m_cell - mstar).exp();
            }
            sync_cube();
            if owns {
                wf.write(cell, Vector::cast_from(weight * l_cell));
            }
            sync_cube();
            if owns {
                let mut lstar = EA::from_int(0);
                #[unroll(splits <= MERGE_UNROLL)]
                for s in 0..splits {
                    lstar += wf.read(base + s * stride).extract(0usize);
                }
                weight *= masked_recip::<EA>(lstar);
            }
            sync_cube();
            if owns {
                wf.write(cell, Vector::cast_from(weight));
            }
        } else {
            let mut r = UNIT_POS as usize;
            while r < rows {
                // Row `r`'s first cell; its splits step by `stride` from there.
                let base = (r / stride) * slice + r % stride;
                let mut mstar = EA::min_value();
                #[unroll(splits <= MERGE_UNROLL)]
                for t in 0..splits {
                    mstar = max(mstar, mf.read(base + t * stride).extract(0usize));
                }
                // The normalizer needs every split, so park the unnormalized
                // weights and scale them where they sit.
                let mut lstar = EA::from_int(0);
                #[unroll(splits <= MERGE_UNROLL)]
                for t in 0..splits {
                    let w = (mf.read(base + t * stride).extract(0usize) - mstar).exp();
                    lstar += lf.read(base + t * stride).extract(0usize) * w;
                    wf.write(base + t * stride, Vector::cast_from(w));
                }
                let recip = masked_recip::<EA>(lstar);
                #[unroll(splits <= MERGE_UNROLL)]
                for t in 0..splits {
                    let w = wf.read(base + t * stride).extract(0usize) * recip;
                    wf.write(base + t * stride, Vector::cast_from(w));
                }
                r += workers;
            }
        }
    }
}
