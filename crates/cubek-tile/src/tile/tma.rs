//! The TMA backing store ([`TmaData`], a tensor-map source): not element-addressable, its
//! only sink is a bulk copy into shared memory, blocking ([`TmaData::load_into`]) or pipelined
//! under a caller-hoisted barrier ([`TmaData::stage_into`]).

use cubecl::{
    prelude::barrier::Barrier,
    prelude::*,
    std::tensor::{ViewMut, layout::CoordsDyn},
};

use crate::*;

/// A TMA tensor-map source: the launch-built `ViewMut`, the current global box origin
/// `pos`, the logical `bound`, and the comptime box shape. `at` advances `pos`; the
/// descriptor and bound ride along unchanged.
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
impl<T: Numeric> TmaData<T> {
    /// Wrap a TMA tensor-map [`ViewMut`] (built on the client side) as a `TmaGmem` tile. `pos`
    /// starts at the origin and advances on [`at`](Tile::at); the box shape is carried comptime
    /// for the `tensor_map_load`. Dormant: no launch path builds this yet.
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
}

#[cube]
impl<T: Numeric> TmaData<T> {
    /// TMA transport leaf, pipelined: issue the elected `tensor_map_load` into `dst`
    /// onto `barrier`, without arriving or waiting; the caller hoists those so the copy
    /// overlaps compute.
    pub(crate) fn stage_into(&self, dst: &mut MemData<T>, barrier: &Shared<Barrier>) {
        // One elected issuer only: the declared transaction count is that unit's alone, so
        // more issuers would over-count and corrupt the stage.
        if UNIT_POS == 0 {
            self.view
                .tensor_map_load(barrier, dst.buffer.downcast_mut(), self.pos.clone());
        }
    }

    /// TMA transport leaf, blocking: bulk-copy into `dst` (shared memory) and wait. Owns its
    /// mbarrier locally; the pipelined path hoists it out via [`stage_into`](TmaData::stage_into).
    pub(crate) fn load_into(&self, dst: &mut MemData<T>) {
        let barrier = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
        sync_async_proxy_shared();
        let expected = select(UNIT_POS == 0, dst.size_bytes(), 0);
        self.stage_into(dst, &barrier);
        let token = barrier.arrive_and_expect_tx(1, expected);
        barrier.wait(token);
    }

    /// Window down to `region`: advance the global origin by each axis's tile coordinate
    /// times its sub-tile edge, so the next `tensor_map_load` copies the windowed box.
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
