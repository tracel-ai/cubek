//! The TMA backing store ([`TmaData`]): a tensor-map source bulk-copied into shared memory.

use cubecl::{
    prelude::barrier::Barrier,
    prelude::*,
    std::tensor::{ViewMut, layout::CoordsDyn},
};

use crate::*;

/// A TMA tensor-map source: the launch-built view, the current box origin and the logical bound.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct TmaData<T: Numeric> {
    view: ViewMut<'static, T, CoordsDyn>,
    pos: CoordsDyn,
    pub(crate) bound: CoordsDyn,
    /// The launch's cube size, `0` when unknown ([`Access::units`](crate::Access)).
    #[cube(comptime)]
    pub(crate) units: usize,
}

#[cube]
impl<T: Numeric> TmaData<T> {
    /// Wrap a TMA tensor-map [`ViewMut`] as a `TmaGmem` tile payload, positioned at the origin.
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
    /// Issue the `tensor_map_load` into `dst` on `barrier` without arriving or waiting.
    /// Only the electing unit may call it, since it alone declares the transaction count.
    pub(crate) fn stage_into(&self, dst: &mut Memory<T>, barrier: &Shared<Barrier>) {
        // A box lands its rows dense, swizzled by the descriptor where the stage keeps them
        // swizzled; the launch built the descriptor off the same answer
        // (`StageStorage::tma_swizzle`).
        comptime!(
            dst.layout
                .rows
                .tma_swizzle()
                .unwrap_or_else(|why| panic!("TmaData::stage_into: a TMA box cannot land {why}"))
        );
        self.view.tensor_map_load(
            barrier,
            dst.store.buffer_mut().downcast_mut(),
            self.pos.clone(),
        );
    }

    /// Bulk-copy into shared-memory `dst` and wait, on a local mbarrier.
    pub(crate) fn load_into(&self, dst: &mut Memory<T>) {
        comptime!(assert!(
            dst.address == AddressSpace::Shared,
            "TmaData::load_into: a tensor map bulk-copies into shared memory only"
        ));
        let barrier = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
        sync_async_proxy_shared();
        let expected = select(UNIT_POS == 0, dst.size_bytes(), 0);
        if UNIT_POS == 0 {
            self.stage_into(dst, &barrier);
        }
        let token = barrier.arrive_and_expect_tx(1, expected);
        barrier.wait(token);
    }

    /// Window down to `step`: advance the global box origin by each axis's tile offset.
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
