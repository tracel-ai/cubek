use super::microkernel::unit_lines_write_microkernel;
use crate::reduce::{ReduceAccumulator, ReduceWorkItem};
use cubecl::tile::Tile;
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// One unit writing the lines its carrier keeps, its lanes met: the values divided by the row's
/// length for a mean, the indices beside them where its item writes both.
///
/// [`UnitCarrierWriteMicrokernel`](crate::reduce::write::UnitCarrierWriteMicrokernel) runs it once
/// the lanes that hold one row are met.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitLinesWriteMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
}

impl UnitLinesWriteMicrokernel {
    /// A unit writing `item`'s carrier.
    pub fn new(item: ReduceWorkItem) -> Self {
        Self { item }
    }

    pub fn item(&self) -> ReduceWorkItem {
        self.item
    }
}

#[cube]
impl UnitLinesWriteMicrokernel {
    /// `acc` written to `out` and `indices` in lines `W` wide, `values` the values each of its rows
    /// held (`unit_lines_write_microkernel`).
    ///
    /// # Panics
    ///
    /// Where `indices` is there and its item does not write them beside the values, or the other
    /// way around; or where it writes indices its carrier does not keep.
    pub fn write<
        Out: Numeric,
        Idx: Numeric,
        Acc: Numeric,
        N: Size,
        W: Size,
        C: ReduceAccumulator<Acc, N>,
    >(
        &self,
        acc: C,
        out: &mut Tile,
        indices: &ComptimeOption<Tile>,
        values: usize,
    ) {
        unit_lines_write_microkernel::<Out, Idx, Acc, N, W, C>(
            acc,
            out,
            indices,
            values,
            comptime!(UnitLinesWriteMicrokernel { item: self.item }),
        )
    }
}
