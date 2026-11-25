use super::microkernel::unit_partial_store_microkernel;
use crate::reduce::{ReduceAccumulator, ReduceWorkItem};
use cubecl::tile::Tile;
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// One unit storing its carrier as one of its rows' partials, as
/// [`UnitPartialsReduceMicrokernel`](crate::reduce::read::UnitPartialsReduceMicrokernel) loads it
/// back: the carrier's lines side by side, the indices beside them where it keeps any.
///
/// It is part of the cube's partial reduce, which hands it the carrier its cube met, and has no
/// launch of its own.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitPartialStoreMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
}

impl UnitPartialStoreMicrokernel {
    /// A unit storing `item`'s carrier.
    pub fn new(item: ReduceWorkItem) -> Self {
        Self { item }
    }

    pub fn item(&self) -> ReduceWorkItem {
        self.item
    }
}

#[cube]
impl UnitPartialStoreMicrokernel {
    /// `acc` stored as partial `part` of `partials` and `partial_indices`
    /// (`unit_partial_store_microkernel`).
    ///
    /// # Panics
    ///
    /// Where `partial_indices` is there and its carrier keeps no index, or the other way around.
    pub fn store<Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
        &self,
        acc: C,
        partials: &mut Tile,
        partial_indices: &ComptimeOption<Tile>,
        part: usize,
    ) {
        unit_partial_store_microkernel::<Acc, N, C>(
            acc,
            partials,
            partial_indices,
            part,
            comptime!(UnitPartialStoreMicrokernel { item: self.item }),
        )
    }
}
