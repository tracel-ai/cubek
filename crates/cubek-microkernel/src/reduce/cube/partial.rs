use crate::reduce::Accumulable;
use crate::reduce::cube::{CubeStridedReduceMicrokernel, cube_strided_reduce_microkernel};
use crate::reduce::write::{UnitPartialStoreMicrokernel, unit_partial_store_microkernel};
use cubecl::tile::Tile;
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// A part of a row reduced by one cube into one of the row's partials: the cube reduces its window
/// as [`CubeStridedReduceMicrokernel`] does, and its first unit stores the carrier
/// ([`UnitPartialStoreMicrokernel`](crate::reduce::write::UnitPartialStoreMicrokernel)).
///
/// The hypercube's cubes run it, a part of a row each.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubePartialReduceMicrokernel {
    /// The microkernel the cube reduces its part with.
    #[cube(comptime)]
    cube: CubeStridedReduceMicrokernel,
}

impl CubePartialReduceMicrokernel {
    pub fn new(cube: CubeStridedReduceMicrokernel) -> Self {
        Self { cube }
    }
}

#[cube]
impl CubePartialReduceMicrokernel {
    /// The lines of `input`, the line at `l` read as line `first_line + l` of its row, reduced into
    /// partial `part` of `partials` and `partial_indices` (`cube_partial_reduce_microkernel`).
    ///
    /// # Assumptions
    ///
    /// - Those of [`ReduceWorkItem`](crate::reduce::ReduceWorkItem), of its item.
    ///
    /// # Panics
    ///
    /// When the kernel compiles, where the item breaks
    /// [`ReduceWorkItem`](crate::reduce::ReduceWorkItem)'s Assumption on `partial_indices`: a tile
    /// handed where its carrier keeps no indices, or none where it keeps them.
    pub fn reduce<In: Numeric, Acc: Numeric, N: Size, C: Accumulable<Acc, N>>(
        &self,
        input: &Tile,
        partials: &mut Tile,
        partial_indices: &ComptimeOption<Tile>,
        part: usize,
        first_line: usize,
    ) {
        cube_partial_reduce_microkernel::<In, Acc, N, C>(
            input,
            partials,
            partial_indices,
            part,
            first_line,
            comptime!(CubePartialReduceMicrokernel { cube: self.cube }),
        )
    }
}

/// One cube's part of a row: the lines of `input`, the line at `l` read as line `first_line + l` of
/// the row, reduced by the cube, whose first unit stores the carrier as partial `part` of `partials`
/// and `partial_indices`. A part the row's end cuts short reads the edge monoid's identity past it,
/// which leaves the carrier as it is: every line of the window is read.
#[cube]
pub(crate) fn cube_partial_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    N: Size,
    C: Accumulable<Acc, N>,
>(
    input: &Tile,
    partials: &mut Tile,
    partial_indices: &ComptimeOption<Tile>,
    part: usize,
    first_line: usize,
    #[comptime] microkernel: CubePartialReduceMicrokernel,
) {
    let cube = microkernel.cube;
    let read = cube_strided_reduce_microkernel::<In, Acc, N, C>(input, first_line, cube);

    if UNIT_POS == 0 {
        unit_partial_store_microkernel::<Acc, N, C>(
            read.carrier,
            partials,
            partial_indices,
            part,
            comptime!(UnitPartialStoreMicrokernel::new(cube.item())),
        );
    }
}

#[cfg(test)]
mod tests {

    use crate::reduce::testing::CubePartialMicrokernelLauncher;

    /// The cube's rows 300 values long along the reduced dimension, a value a line, each a partial.
    mod partial_along_300_long_one_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(CubePartialMicrokernelLauncher::new(4), Along, 300, 1, Plain, [
            sum: ReduceOperation::Sum,
            prod: ReduceOperation::Prod,
            max: ReduceOperation::Max,
            min: ReduceOperation::Min,
            max_abs: ReduceOperation::MaxAbs,
            any: ReduceOperation::Any,
            all: ReduceOperation::All,
        ]);
    }

    /// The cube's rows 300 values long across the reduced dimension, each lane its own row, four a partial.
    mod partial_across_300_long_four_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(CubePartialMicrokernelLauncher::new(4), Across, 300, 4, Plain, [
            sum: ReduceOperation::Sum,
            max: ReduceOperation::Max,
            min: ReduceOperation::Min,
        ]);
    }
}
