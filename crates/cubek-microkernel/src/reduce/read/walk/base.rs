use super::microkernel::unit_walk_reduce_microkernel;
use crate::reduce::{ReduceAccumulator, ReduceWorkItem};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// One unit's share of rows its plane or its cube reads with its units across them: the cells
/// a region's walk gives it, `per_step` positions of the reduced dimension a step, accumulated
/// into its carrier, each as its position along the reduced dimension.
///
/// It is part of the plane and cube microkernels, which meet the units' carriers after it, and has
/// no launch of its own.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitWalkReduceMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    /// The positions of the reduced dimension one step of the walk covers.
    #[cube(comptime)]
    per_step: usize,
}

impl UnitWalkReduceMicrokernel {
    /// A unit reducing `item` from a walk covering `per_step` positions a step.
    pub fn new(item: ReduceWorkItem, per_step: usize) -> Self {
        Self { item, per_step }
    }

    pub fn item(&self) -> ReduceWorkItem {
        self.item
    }

    pub fn per_step(&self) -> usize {
        self.per_step
    }
}

#[cube]
impl UnitWalkReduceMicrokernel {
    /// The cells of `input` `region`'s walk gives this unit, `values` positions long, accumulated
    /// from the identity (`unit_walk_reduce_microkernel`).
    pub fn reduce<In: Numeric, Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
        &self,
        region: &Region,
        input: &Tile,
        values: usize,
    ) -> C {
        unit_walk_reduce_microkernel::<In, Acc, N, C>(
            region,
            input,
            values,
            comptime!(UnitWalkReduceMicrokernel {
                item: self.item,
                per_step: self.per_step
            }),
        )
    }
}
