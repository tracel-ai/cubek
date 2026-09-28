//! The TMA backing store ([`TmaData`], a tensor-map source): not element-addressable, its
//! only sink is a bulk copy into shared memory, blocking ([`TmaData::load_into`]) or pipelined
//! under a caller-owned barrier ([`TmaData::stage_into`]).

use cubecl::{
    prelude::barrier::Barrier,
    prelude::*,
    std::tensor::{ViewMut, layout::CoordsDyn},
};

use crate::*;

/// A TMA tensor-map source: the launch-built `ViewMut`, the current global box origin
/// `pos`, and the logical `bound`. `at` advances `pos`; the descriptor (which owns the
/// box shape) and bound ride along unchanged.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct TmaData<T: Numeric> {
    view: ViewMut<'static, T, CoordsDyn>,
    pos: CoordsDyn,
    pub(crate) bound: CoordsDyn,
    /// The launch's cube size, `0` when unknown ([`Access::units`](crate::Access)).
    #[cube(comptime)]
    pub(crate) units: usize,
}

#[cube]
impl<T: Numeric> TmaData<T> {
    /// Wrap a TMA tensor-map [`ViewMut`] (built on the client side, [`TmaTileArg`]) as the payload
    /// of a `TmaGmem` tile. `pos` starts at the origin and advances on [`at`](Tile::at).
    pub(crate) fn from_tensor_map(
        view: ViewMut<'static, T, CoordsDyn>,
        #[comptime] rank: usize,
        #[comptime] units: usize,
    ) -> TmaData<T> {
        let bound = view.shape();
        let mut pos = CoordsDyn::new();
        #[unroll]
        for _ in 0..rank {
            pos.push(0u32);
        }
        TmaData::<T> {
            view,
            pos,
            bound,
            units,
        }
    }
}

#[cube]
impl<T: Numeric> TmaData<T> {
    /// TMA transport leaf, pipelined: issue the `tensor_map_load` into `dst` onto `barrier`,
    /// without arriving or waiting; the caller issues those itself so the copy overlaps compute.
    ///
    /// The caller elects, because the same unit must declare the transaction count: the bytes are
    /// that unit's alone, and a second issuer would over-count and corrupt the stage.
    pub(crate) fn stage_into(&self, dst: &mut Memory<T>, barrier: &Shared<Barrier>) {
        // The bulk copy lands its box row after row, in order and unpadded; the tensor map's own
        // swizzle modes are a different permutation from a swizzled stage's.
        comptime!(dst.layout.rows.assert_in_order(
            "TmaData::stage_into",
            "a TMA box lands its rows dense and in order"
        ));
        self.view.tensor_map_load(
            barrier,
            dst.store.buffer_mut().downcast_mut(),
            self.pos.clone(),
        );
    }

    /// TMA transport leaf, blocking: bulk-copy into `dst` (shared memory) and wait. Owns its
    /// mbarrier locally; the pipelined path leaves it to the caller via
    /// [`stage_into`](TmaData::stage_into).
    pub(crate) fn load_into(&self, dst: &mut Memory<T>) {
        comptime!(assert!(
            dst.address == AddressSpace::Shared,
            "TmaData::load_into: a tensor map bulk-copies into shared memory only"
        ));
        let barrier = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
        sync_async_proxy_shared();
        // Unit 0 issues the copy and declares its bytes; every unit arrives and waits.
        let expected = select(UNIT_POS == 0, dst.size_bytes(), 0);
        if UNIT_POS == 0 {
            self.stage_into(dst, &barrier);
        }
        let token = barrier.arrive_and_expect_tx(1, expected);
        barrier.wait(token);
    }

    /// Window down to `region`: advance the global origin by each axis's tile coordinate
    /// times its sub-tile edge, so the next `tensor_map_load` copies the windowed box.
    pub(crate) fn at(&self, step: &Step, #[comptime] space: Space) -> TmaData<T> {
        let mut pos = CoordsDyn::new();

        #[unroll]
        for p in 0..space.rank() {
            let axis = space.axis_at(p);
            match comptime!(step.level.tile(axis)) {
                Some(tile) => {
                    let index = step.coord(axis);
                    pos.push(self.pos[p] + (index * tile) as u32);
                }
                None => pos.push(self.pos[p]),
            }
        }

        TmaData::<T> {
            view: self.view.clone(),
            pos,
            bound: self.bound.clone(),
            units: self.units,
        }
    }
}
