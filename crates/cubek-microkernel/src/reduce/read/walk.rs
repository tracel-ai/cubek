use crate::reduce::read::accumulate_line;
use crate::reduce::{Accumulable, REDUCED, ReduceWorkItem};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// One unit's share of rows its plane or its cube reads with its units across them: the cells
/// a region's walk gives it, accumulated into its carrier, each as its position along the reduced dimension.
///
/// [`UnitStridedReduceMicrokernel`](crate::reduce::read::UnitStridedReduceMicrokernel) is its twin
/// where the units read one line of rows.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitWalkReduceMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    /// The positions of the reduced dimension one step of the walk covers.
    #[cube(comptime)]
    per_step: usize,
}

impl UnitWalkReduceMicrokernel {
    pub fn new(item: ReduceWorkItem, per_step: usize) -> Self {
        Self { item, per_step }
    }
}

#[cube]
impl UnitWalkReduceMicrokernel {
    /// The cells of `input` `region`'s walk gives this unit, `values` positions long, accumulated
    /// from the identity (`unit_walk_reduce_microkernel`).
    pub fn reduce<In: Numeric, Acc: Numeric, N: Size, C: Accumulable<Acc, N>>(
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

/// Every step but the last reads without a check, as
/// [`unit_strided_reduce_microkernel`](crate::reduce::read::unit_strided_reduce_microkernel)'s do.
#[cube]
pub(crate) fn unit_walk_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    N: Size,
    C: Accumulable<Acc, N>,
>(
    region: &Region,
    input: &Tile,
    values: usize,
    #[comptime] microkernel: UnitWalkReduceMicrokernel,
) -> C {
    let item = microkernel.item;
    let walk = region.walk();
    let load = input.vector_tile();
    let row_lanes = comptime!(load.extent_along(REDUCED));
    let full_steps = values / microkernel.per_step;
    let mut acc = C::identity(item.operation.monoid());
    let first_cell = 0usize;
    for step in 0..full_steps {
        for unit in walk.region(step).leaves() {
            let cell = input.at(&unit);
            accumulate_line::<In, Acc, N, C>(
                &mut acc,
                &cell.flat::<In, N>(),
                first_cell,
                unit.origin(REDUCED),
                row_lanes,
                item,
            );
        }
    }

    // The step the rows' end cuts short, where one is: each unit checks its own position.
    for step in full_steps..values.div_ceil(microkernel.per_step) {
        for unit in walk.region(step).leaves() {
            let cell = input.at(&unit);
            let position = unit.origin(REDUCED);
            if position < values {
                accumulate_line::<In, Acc, N, C>(
                    &mut acc,
                    &cell.flat::<In, N>(),
                    first_cell,
                    position,
                    row_lanes,
                    item,
                );
            }
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::ReduceCarrierFamily;
    use crate::reduce::plane::PlaneColumnsReduceMicrokernel;
    use crate::reduce::testing::{
        ComponentLaunch, LaunchedRows, MicrokernelLauncher, a_plane_a_cube, plane_lanes,
    };
    use crate::reduce::write::{
        UnitGroupColumnWriteMicrokernel, unit_group_column_write_microkernel,
    };
    use cubecl::ir::ElemType;
    use cubecl::tile::launch::Decline;
    use cubecl::tile::{Levels, Partitioning, TileTensor};
    use cubek_std::Hardware;

    /// A plane's units lie across several lines of rows, and each column's first unit's carrier is written.
    #[derive(Debug, Clone, Copy)]
    struct UnitWalkMicrokernelLauncher {
        columns: usize,
    }

    impl MicrokernelLauncher for UnitWalkMicrokernelLauncher {
        fn levels(&self, _hw: &Hardware, rows: &LaunchedRows) -> Levels {
            a_plane_a_cube(
                PlaneColumnsReduceMicrokernel::new(rows.item, rows.lanes, self.columns)
                    .levels(&rows.owner()),
            )
        }

        fn plane_width(&self, hw: &Hardware, item: &ReduceWorkItem) -> Result<usize, Decline> {
            plane_lanes(hw, item)
        }

        fn stride_on(&self, hw: &Hardware) -> usize {
            hw.plane_dim().unwrap_or(1) as usize / self.columns
        }

        fn launch_into<F: ReduceCarrierFamily>(&self, launch: ComponentLaunch<'_>) {
            let elements = launch.elements();
            launch_component!(
                launch,
                unit_walk_reduce_kernel,
                sizes: [],
                [launch.rows.reduced],
                UnitWalkReduceMicrokernel::new(launch.rows.item, launch.rows.lanes / self.columns),
                elements,
            );
        }
    }

    /// Every plane's rows, its units across lines of them, read by the walked read into a carrier of values alone, written by the column's write.
    ///
    /// Its operands' leaves:
    /// - `input`: the rows, a row a lane, across the reduced dimension.
    /// - `out`: one tensor of a cell a row.
    #[cube(launch_unchecked)]
    fn unit_walk_reduce_kernel<In: Numeric, Acc: Numeric, Out: Numeric, N: Size>(
        input: &TileTensor<'_>,
        out: &mut TileTensor<'_>,
        partitioning: Partitioning,
        #[comptime] values: usize,
        #[comptime] microkernel: UnitWalkReduceMicrokernel,
        #[define(In, Acc, Out)] _elements: [ElemType; 3],
    ) {
        let x = input.tile(&partitioning);
        let o = out.tile_mut(&partitioning);

        for plane in &partitioning {
            let carrier = microkernel
                .runtime()
                .reduce::<In, Acc, N, Vector<Acc, N>>(&plane, &x, values);
            unit_group_column_write_microkernel::<Out, Out, Acc, N, Vector<Acc, N>>(
                &plane,
                carrier,
                &o,
                &ComptimeOption::new_None(),
                values,
                comptime!(UnitGroupColumnWriteMicrokernel::new(microkernel.item)),
            );
        }
    }

    /// Walked across rows 300 values long, each lane its own row: the last step of the walk is cut
    /// short.
    mod walk_across_300_long_four_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(UnitWalkMicrokernelLauncher { columns: 2 }, Across, 300, 4, Plain, [
            sum: ReduceOperation::Sum,
            prod: ReduceOperation::Prod,
            max: ReduceOperation::Max,
            min: ReduceOperation::Min,
            max_abs: ReduceOperation::MaxAbs,
            any: ReduceOperation::Any,
            all: ReduceOperation::All,
        ]);
    }

    /// Walked across rows 320 values long, no step of the walk cut short whatever the plane's width.
    mod walk_across_320_long_four_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(UnitWalkMicrokernelLauncher { columns: 2 }, Across, 320, 4, Plain, [
            sum: ReduceOperation::Sum,
        ]);
    }
}
