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
    /// # Panics
    ///
    /// Where the cube's carrier and `partial_indices` disagree on whether it keeps indices.
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
    let count = input.flat::<In, N>().shape();
    let total = cube_strided_reduce_microkernel::<In, Acc, N, C>(input, first_line, count, cube);

    if UNIT_POS == 0 {
        unit_partial_store_microkernel::<Acc, N, C>(
            total,
            partials,
            partial_indices,
            part,
            comptime!(UnitPartialStoreMicrokernel::new(cube.item())),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::testing::{ComponentLaunch, LaunchedRows, MicrokernelLauncher, cube_lanes};
    use crate::reduce::{BATCH, ReduceCarrierFamily, ReduceWorkItem};
    use cubecl::ir::ElemType;
    use cubecl::tile::launch::Decline;
    use cubecl::tile::{Levels, Partitioning, TileTensor};
    use cubek_std::Hardware;

    /// A cube reduces its rows into their one partial.
    #[derive(Debug, Clone, Copy)]
    struct CubePartialMicrokernelLauncher {
        planes: usize,
    }

    impl MicrokernelLauncher for CubePartialMicrokernelLauncher {
        fn levels(&self, _hw: &Hardware, rows: &LaunchedRows) -> Levels {
            rows.whole_rows().cubes(&[BATCH])
        }

        fn plane_width(&self, hw: &Hardware, item: &ReduceWorkItem) -> Result<usize, Decline> {
            cube_lanes(hw, item, self.planes)
        }

        fn launch_plane_width(&self, _hw: &Hardware, rows: &LaunchedRows) -> usize {
            rows.lanes * self.planes
        }

        fn launch_into<F: ReduceCarrierFamily>(&self, launch: ComponentLaunch<'_>) {
            let [input, accumulator, _] = launch.elements();
            let elements = [input, accumulator];
            launch_component!(
                launch,
                cube_partial_reduce_kernel,
                sizes: [],
                [],
                CubePartialReduceMicrokernel::new(CubeStridedReduceMicrokernel::new(launch.rows.item, launch.rows.lanes, self.planes)),
                elements,
            );
        }
    }

    /// Every cube's row, reduced by the cube into a carrier of values alone and stored to `out` as its
    /// one partial.
    ///
    /// Its operands' leaves:
    /// - `input`: the rows, one value a line along the reduced dimension, or a row a lane across it.
    /// - `out`: the partials, one line a row.
    #[cube(launch_unchecked)]
    fn cube_partial_reduce_kernel<In: Numeric, Acc: Numeric, N: Size>(
        input: &TileTensor<'_>,
        out: &mut TileTensor<'_>,
        partitioning: Partitioning,
        #[comptime] microkernel: CubePartialReduceMicrokernel,
        #[define(In, Acc)] _elements: [ElemType; 2],
    ) {
        let x = input.tile(&partitioning);
        let o = out.tile_mut(&partitioning);

        for cube in &partitioning {
            let mut partials = o.at(&cube);
            let first = 0usize;
            microkernel.runtime().reduce::<In, Acc, N, Vector<Acc, N>>(
                &x.at(&cube),
                &mut partials,
                &ComptimeOption::new_None(),
                first,
                first,
            );
        }
    }

    /// The cube's rows 300 values long along the reduced dimension, a value a line, each a partial.
    mod partial_along_300_long_one_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(CubePartialMicrokernelLauncher { planes: 4 }, Along, 300, 1, Plain, [
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
        reduce_tests!(CubePartialMicrokernelLauncher { planes: 4 }, Across, 300, 4, Plain, [
            sum: ReduceOperation::Sum,
            max: ReduceOperation::Max,
            min: ReduceOperation::Min,
        ]);
    }
}
