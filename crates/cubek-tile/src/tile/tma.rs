//! The TMA backing store ([`TmaData`], a tensor-map source): not element-addressable, its
//! only sink is a bulk copy into shared memory, blocking ([`Tile::tma_load`]) or pipelined
//! under a caller-hoisted barrier ([`Tile::stage_from`]).

use cubecl::{
    prelude::barrier::Barrier,
    prelude::*,
    std::tensor::{ViewMut, layout::CoordsDyn},
};

use crate::*;

/// A TMA tensor-map source: the launch-built `ViewMut` (backed by a `TensorMapTiled` `ViewArg`),
/// the current global box origin `pos`, the logical `bound`, and the comptime box shape. Not
/// element-addressable — its only sink is a [`stage_from`](Tile::stage_from) into shared memory, which
/// lowers to a hardware bulk copy. `at` advances `pos`; the descriptor and bound ride along unchanged.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct TmaData<T: Numeric> {
    view: ViewMut<'static, T, CoordsDyn>,
    pos: CoordsDyn,
    pub(crate) bound: CoordsDyn,
    #[cube(comptime)]
    box_rows: u32,
    #[cube(comptime)]
    box_cols: u32,
    #[cube(comptime)]
    transposed: bool,
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// Wrap a TMA tensor-map [`ViewMut`] (built on the client side) as a `TmaGmem` tile. The global
    /// `(batch, row, col)` bound is read off the view; `pos` starts at the origin and advances on
    /// [`at`](Tile::at). Element addressing is unavailable — its only sink is a
    /// [`stage_from`](Tile::stage_from) into shared memory. The store is scalar (`vector_size == 1`); the box
    /// shape is carried comptime for the `tensor_map_load`. Dormant: no launch path builds this yet.
    pub fn from_tensor_map(
        view: ViewMut<'static, T, CoordsDyn>,
        #[comptime] space: Space,
        #[comptime] box_rows: u32,
        #[comptime] box_cols: u32,
        #[comptime] transposed: bool,
    ) -> Tile<T> {
        let bound = view.shape();
        let mut pos = CoordsDyn::new();
        #[unroll]
        for _ in 0..comptime!(space.rank()) {
            pos.push(0u32);
        }
        Tile::<T> {
            tile_kind: TileKind::new_TmaGmem(TmaData::<T> {
                view,
                pos,
                bound,
                box_rows,
                box_cols,
                transposed,
            }),
            space: comptime!(space),
        }
    }

    /// The transaction byte count a TMA fill into this shared-memory tile lands (its buffer
    /// length, runtime). Zero for the non-smem kinds, which are never TMA fill targets.
    pub(crate) fn tma_transaction_bytes(&self) -> u32 {
        match &self.tile_kind {
            TileKind::Smem(d) => d.buffer.len() as u32 * comptime!(T::type_size() as u32),
            TileKind::Gmem(_) => 0,
            TileKind::Cmma(_) => 0,
            TileKind::CmmaPartition(_) => 0,
            TileKind::TmaGmem(_) => 0,
        }
    }

    /// TMA transport leaf, pipelined: issue the elected `tensor_map_load` of `src` (a tensor-map source)
    /// into this shared-memory tile onto `barrier`, without arriving or waiting — the caller hoists those
    /// so the copy overlaps compute. The core shared by [`stage_from`](Tile::stage_from) (barrier hoisted) and the
    /// blocking [`tma_load`](Tile::tma_load) (barrier owned locally).
    pub(crate) fn tma_stage(&mut self, src: &Tile<T>, barrier: &Shared<Barrier>) {
        match (&src.tile_kind, &mut self.tile_kind) {
            (TileKind::TmaGmem(s), TileKind::Smem(d)) => {
                // The bulk copy is issued by one elected unit only; the transaction count
                // declared by the caller is that unit's alone, so more issuers would over-count
                // and corrupt the stage.
                if UNIT_POS == 0 {
                    s.view
                        .tensor_map_load(barrier, d.buffer.downcast_mut(), s.pos.clone());
                }
            }
            // `stage_from`/`tma_load` route here only for a tma source targeting shared memory.
            _ => panic!("Tile::stage_from: TMA source must target shared memory"),
        }
    }

    /// TMA transport leaf, blocking: hardware bulk-copy `src` (a tensor-map source) into this
    /// shared-memory tile and wait for it. Owns its `mbarrier` locally — arms it, issues the copy via
    /// [`tma_stage`](Tile::tma_stage), then `arrive_and_expect_tx` + `wait` before returning. The
    /// double-buffered path hoists that barrier out with [`stage_from`](Tile::stage_from) instead.
    pub(crate) fn tma_load(&mut self, src: &Tile<T>) {
        let barrier = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
        sync_async_proxy_shared();
        // One elected issuer only, matching the declared transaction count; more issuers over-count.
        let expected = select(UNIT_POS == 0, self.tma_transaction_bytes(), 0);
        self.tma_stage(src, &barrier);
        let token = barrier.arrive_and_expect_tx(1, expected);
        barrier.wait(token);
    }
}

#[cube]
impl<T: Numeric> TmaData<T> {
    /// Window down to `region`: advance the global origin by each axis's tile coordinate times its
    /// sub-tile edge. The descriptor and bound carry through unchanged — only `pos` moves — so the
    /// next `tensor_map_load` copies the windowed box.
    pub(crate) fn at(&self, region: &Region, #[comptime] space: Space) -> TmaData<T> {
        let mut pos = CoordsDyn::new();

        #[unroll]
        for p in 0..space.rank() {
            let axis = space.axis_at(p);
            let edge = space.partitioner().edge(axis);
            let index = region.coord(axis);
            pos.push(self.pos[p] + (index * edge) as u32);
        }

        TmaData::<T> {
            view: self.view.clone(),
            pos,
            bound: self.bound.clone(),
            box_rows: comptime!(self.box_rows),
            box_cols: comptime!(self.box_cols),
            transposed: comptime!(self.transposed),
        }
    }
}
