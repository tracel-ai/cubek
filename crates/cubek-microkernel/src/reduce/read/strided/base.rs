use super::microkernel::unit_strided_reduce_microkernel;
use crate::reduce::{ReduceAccumulator, ReduceWorkItem};
use cubecl::tile::Tile;
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// One unit's share of a row its plane or its cube reads together: the lines from its first on,
/// every `stride`-th, accumulated into its carrier, each as its line of the row.
///
/// It is part of the plane and cube microkernels, which meet the units' carriers after it, and has
/// no launch of its own.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitStridedReduceMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    /// The units that read the row together, one line each in turn.
    #[cube(comptime)]
    stride: usize,
}

impl UnitStridedReduceMicrokernel {
    /// A unit reducing `item` with `stride` units reading the row in turns.
    pub fn new(item: ReduceWorkItem, stride: usize) -> Self {
        Self { item, stride }
    }

    pub fn item(&self) -> ReduceWorkItem {
        self.item
    }

    pub fn stride(&self) -> usize {
        self.stride
    }
}

#[cube]
impl UnitStridedReduceMicrokernel {
    /// The lines of `input` from `first` on, every `stride`-th up to `count`, accumulated from
    /// the identity, line `l` as line `offset + l` of its row (`unit_strided_reduce_microkernel`).
    pub fn reduce<In: Numeric, Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
        &self,
        input: &Tile,
        first: usize,
        count: usize,
        offset: usize,
    ) -> C {
        unit_strided_reduce_microkernel::<In, Acc, N, C>(
            input,
            first,
            count,
            offset,
            comptime!(UnitStridedReduceMicrokernel {
                item: self.item,
                stride: self.stride
            }),
        )
    }
}
