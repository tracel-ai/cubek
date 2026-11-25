use super::microkernel::unit_partials_reduce_microkernel;
use crate::reduce::{ReduceAccumulator, ReduceWorkItem};
use cubecl::tile::Tile;
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// One unit reducing the partials several cubes stored of its rows
/// ([`UnitPartialStoreMicrokernel`](crate::reduce::write::UnitPartialStoreMicrokernel)): each
/// partial loaded back into its carrier, the carriers combined and the rows written.
///
/// It is part of the hypercube microkernel, whose last cube to finish runs it on one unit, and has
/// no launch of its own.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitPartialsReduceMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    /// As [`UnitCarrierWriteMicrokernel`](crate::reduce::write::UnitCarrierWriteMicrokernel)'s.
    #[cube(comptime)]
    row_lanes: usize,
}

impl UnitPartialsReduceMicrokernel {
    /// A unit reducing `item`'s partials, `row_lanes` lanes of each line holding one row.
    pub fn new(item: ReduceWorkItem, row_lanes: usize) -> Self {
        Self { item, row_lanes }
    }

    pub fn item(&self) -> ReduceWorkItem {
        self.item
    }

    pub fn row_lanes(&self) -> usize {
        self.row_lanes
    }
}

#[cube]
impl UnitPartialsReduceMicrokernel {
    /// The first `parts` partials of `partials` and `partial_indices` reduced into `out` and
    /// `indices`, `values` the values each of the rows holds (`unit_partials_reduce_microkernel`).
    ///
    /// # Panics
    ///
    /// As [`UnitCarrierWriteMicrokernel::write`](crate::reduce::write::UnitCarrierWriteMicrokernel::write),
    /// or where `partial_indices` is there and its carrier keeps no index, or the other way
    /// around.
    pub fn reduce<
        Out: Numeric,
        Idx: Numeric,
        Acc: Numeric,
        N: Size,
        C: ReduceAccumulator<Acc, N>,
    >(
        &self,
        partials: &Tile,
        partial_indices: &ComptimeOption<Tile>,
        parts: usize,
        out: &mut Tile,
        indices: &ComptimeOption<Tile>,
        values: usize,
    ) {
        unit_partials_reduce_microkernel::<Out, Idx, Acc, N, C>(
            partials,
            partial_indices,
            parts,
            out,
            indices,
            values,
            comptime!(UnitPartialsReduceMicrokernel {
                item: self.item,
                row_lanes: self.row_lanes
            }),
        )
    }
}
