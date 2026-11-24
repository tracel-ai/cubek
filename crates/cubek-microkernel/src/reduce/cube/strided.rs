use crate::reduce::read::{UnitStridedReduceMicrokernel, unit_strided_reduce_microkernel};
use crate::reduce::{Accumulable, ReduceWorkItem};
use cubecl::tile::Tile;
use cubecl::tile::space::UnitShare;
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// Lines that one cube reads together, its units in turns, and meets into one carrier its
/// first unit is left with.
///
/// It is the read of [`CubeLineReduceMicrokernel`](crate::reduce::cube::CubeLineReduceMicrokernel),
/// whose rows it holds whole, and of
/// [`CubePartialReduceMicrokernel`](crate::reduce::cube::CubePartialReduceMicrokernel), whose part.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStridedReduceMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    #[cube(comptime)]
    lanes: usize,
    #[cube(comptime)]
    planes: usize,
}

impl CubeStridedReduceMicrokernel {
    pub fn new(item: ReduceWorkItem, lanes: usize, planes: usize) -> Self {
        Self {
            item,
            lanes,
            planes,
        }
    }
}

impl CubeStridedReduceMicrokernel {
    pub(crate) fn item(&self) -> ReduceWorkItem {
        self.item
    }
}

#[cube]
impl CubeStridedReduceMicrokernel {
    /// The first `count` lines of `input` accumulated by the cube into `C`, the line at `l` read
    /// as line `first_line + l` of its row (`cube_strided_reduce_microkernel`).
    pub fn reduce<In: Numeric, Acc: Numeric, N: Size, C: Accumulable<Acc, N>>(
        &self,
        input: &Tile,
        first_line: usize,
        count: usize,
    ) -> C {
        cube_strided_reduce_microkernel::<In, Acc, N, C>(
            input,
            first_line,
            count,
            comptime!(CubeStridedReduceMicrokernel {
                item: self.item,
                lanes: self.lanes,
                planes: self.planes
            }),
        )
    }
}

/// Consecutive units read consecutive lines, each plane's lanes meet in the plane's instructions and
/// the planes in shared memory ([`across_planes`]).
#[cube]
pub(crate) fn cube_strided_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    N: Size,
    C: Accumulable<Acc, N>,
>(
    input: &Tile,
    first_line: usize,
    count: usize,
    #[comptime] microkernel: CubeStridedReduceMicrokernel,
) -> C {
    let monoid = microkernel.item.operation.monoid();
    let unit = PLANE_POS as usize * microkernel.lanes + UNIT_POS_PLANE as usize;
    let acc = unit_strided_reduce_microkernel::<In, Acc, N, C>(
        input,
        unit,
        count,
        first_line,
        comptime!(UnitStridedReduceMicrokernel::new(
            microkernel.item,
            microkernel.lanes * microkernel.planes
        )),
    );

    let plane = C::across(acc, UnitShare::Plane, monoid);
    let one_column = comptime!(1usize);
    across_planes::<Acc, N, C>(
        plane,
        0,
        UNIT_POS_PLANE == 0,
        one_column,
        microkernel.planes,
        microkernel.item,
    )
}

/// Bytes of shared memory [`across_planes`] allocates for `partials` partials of `item`'s carrier, its
/// lines `width` lanes wide: the carriers, and one line in place of the indices of one that keeps none.
pub(crate) fn staged_bytes(item: ReduceWorkItem, partials: usize, width: usize) -> usize {
    let carrier = item.operation.carrier();
    let absent_indices = match carrier.carries_indices() {
        true => 0,
        false => width * size_of::<u32>(),
    };
    partials * carrier.bytes(width, item.accumulator.size()) + absent_indices
}

/// The partials of the cube's planes, each already met across its own lanes, met in shared
/// memory, one total for each of the `columns` lines of rows its units lie across: the unit of
/// each plane and column where `stages` is set stages its partial, `acc`, in that column's slot of
/// its plane, the cube meets, the first plane's units combine their column's slots, and the cube
/// meets again so the slots may be staged anew. Only the first plane's units are left with their
/// column's total; the others hold the identity.
#[cube]
pub(crate) fn across_planes<Acc: Numeric, N: Size, C: Accumulable<Acc, N>>(
    acc: C,
    column: usize,
    stages: bool,
    #[comptime] columns: usize,
    #[comptime] planes: usize,
    #[comptime] item: ReduceWorkItem,
) -> C {
    let operation = item.operation;
    // A partial is its carrier's lines, a slot each, the indices beside where it keeps any.
    let lines = planes * columns * operation.written_per_row();
    let index_lines = comptime!(match operation.carrier().carries_indices() {
        true => lines,
        false => 1,
    });
    let mut values = Shared::<[Vector<Acc, N>]>::new_slice(lines);
    let mut indices = Shared::<[Vector<u32, N>]>::new_slice(index_lines);
    if stages {
        stage_partial::<Acc, N, C>(
            &mut values,
            &mut indices,
            PLANE_POS as usize * columns + column,
            acc,
            item,
        );
    }

    sync_cube();

    let total =
        merged_by_first_plane::<Acc, N, C>(&values, &indices, column, columns, planes, item);

    sync_cube();
    total
}

/// `acc` staged as partial `partial` of `values` and `indices`, its slots' lines side by side.
#[cube]
fn stage_partial<Acc: Numeric, N: Size, C: Accumulable<Acc, N>>(
    values: &mut Shared<[Vector<Acc, N>]>,
    indices: &mut Shared<[Vector<u32, N>]>,
    partial: usize,
    acc: C,
    #[comptime] item: ReduceWorkItem,
) {
    let operation = item.operation;
    let slots = operation.written_per_row();
    let with_indices = operation.carrier().carries_indices();
    #[unroll]
    for slot in 0..slots {
        values[partial * slots + slot] = C::value_line(&acc, slot);
        if with_indices {
            indices[partial * slots + slot] = C::index_line(&acc, slot);
        }
    }
}

/// The partials of `column` the cube's planes staged in `values` and `indices`, `columns` a
/// plane, combined by the cube's first plane; every other unit is left with the identity.
#[cube]
fn merged_by_first_plane<Acc: Numeric, N: Size, C: Accumulable<Acc, N>>(
    values: &Shared<[Vector<Acc, N>]>,
    indices: &Shared<[Vector<u32, N>]>,
    column: usize,
    #[comptime] columns: usize,
    #[comptime] planes: usize,
    #[comptime] item: ReduceWorkItem,
) -> C {
    let mut total = C::identity(item.operation.monoid());
    if PLANE_POS == 0 {
        total = merged_partials::<Acc, N, C>(values, indices, column, columns, planes, item);
    }
    total
}

/// The partials of `column` the cube's planes staged in `values` and `indices`, `columns` a
/// plane, combined. A carrier of one slot unrolls its loop over the planes; a top-k's stays a
/// loop, as each of its partials is itself `k` lines.
#[cube]
fn merged_partials<Acc: Numeric, N: Size, C: Accumulable<Acc, N>>(
    values: &Shared<[Vector<Acc, N>]>,
    indices: &Shared<[Vector<u32, N>]>,
    column: usize,
    #[comptime] columns: usize,
    #[comptime] planes: usize,
    #[comptime] item: ReduceWorkItem,
) -> C {
    let operation = item.operation;
    let monoid = operation.monoid();
    let one_slot = operation.written_per_row() == 1;
    let mut total = C::identity(monoid);
    #[unroll(one_slot)]
    for plane in 0..planes {
        let partial = staged_partial::<Acc, N, C>(values, indices, plane * columns + column, item);
        total = C::combine(total, partial, monoid);
    }
    total
}

/// Partial `partial` of `values` and `indices` as [`stage_partial`] staged it.
#[cube]
fn staged_partial<Acc: Numeric, N: Size, C: Accumulable<Acc, N>>(
    values: &Shared<[Vector<Acc, N>]>,
    indices: &Shared<[Vector<u32, N>]>,
    partial: usize,
    #[comptime] item: ReduceWorkItem,
) -> C {
    let operation = item.operation;
    let slots = operation.written_per_row();
    let with_indices = operation.carrier().carries_indices();
    let mut staged = C::identity(operation.monoid());
    #[unroll]
    for slot in 0..slots {
        let index = match with_indices {
            true => indices[partial * slots + slot],
            false => Vector::new(0u32),
        };
        C::set_line(&mut staged, slot, values[partial * slots + slot], index);
    }
    staged
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::ReduceOperation;
    use crate::reduce::testing::{ComponentLaunch, LaunchedRows, MicrokernelLauncher, cube_lanes};
    use crate::reduce::write::{UnitRegionWriteMicrokernel, unit_region_write_microkernel};
    use crate::reduce::{BATCH, ReduceCarrierFamily};
    use cubecl::ir::ElemType;
    use cubecl::tile::launch::Decline;
    use cubecl::tile::{Levels, Partitioning, TileTensor};
    use cubek_std::Hardware;

    /// A cube reads its rows into a carrier, whose line its first unit writes.
    #[derive(Debug, Clone, Copy)]
    struct CubeStridedMicrokernelLauncher {
        planes: usize,
    }

    impl MicrokernelLauncher for CubeStridedMicrokernelLauncher {
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
            let elements = launch.elements();
            launch_component!(
                launch,
                cube_strided_reduce_kernel::<F>,
                sizes: [launch.rows.cells()],
                [launch.rows.reduced, launch.rows.row_lanes()],
                CubeStridedReduceMicrokernel::new(launch.rows.item, launch.rows.lanes, self.planes),
                elements,
            );
        }
    }

    /// Every cube's row, read by the cube into its carrier, which the region's write writes.
    ///
    /// Its operands' leaves:
    /// - `input`: the rows, laid along or across the reduced dimension.
    /// - `out`: one tensor of a cell a row (`k` for a top-k).
    #[cube(launch_unchecked)]
    fn cube_strided_reduce_kernel<
        In: Numeric,
        Acc: Numeric,
        Out: Numeric,
        N: Size,
        K: Size,
        F: ReduceCarrierFamily,
    >(
        input: &TileTensor<'_>,
        out: &mut TileTensor<'_>,
        partitioning: Partitioning,
        #[comptime] values: usize,
        #[comptime] row_lanes: usize,
        #[comptime] microkernel: CubeStridedReduceMicrokernel,
        #[define(In, Acc, Out)] _elements: [ElemType; 3],
    ) {
        let x = input.tile(&partitioning);
        let o = out.tile_mut(&partitioning);

        for cube in &partitioning {
            let row = x.at(&cube);
            let count = row.flat::<In, N>().shape();
            let carrier = microkernel
                .runtime()
                .reduce::<In, Acc, N, F::Carrier<Acc, N, K>>(&row, 0, count);
            unit_region_write_microkernel::<Out, Out, Acc, N, F::Carrier<Acc, N, K>>(
                &cube,
                carrier,
                &o,
                &ComptimeOption::new_None(),
                values,
                UNIT_POS == 0,
                comptime!(UnitRegionWriteMicrokernel::new(microkernel.item, row_lanes)),
            );
        }
    }

    /// The cube's read into a carrier, rows 300 values long along the reduced dimension, four values a
    /// line.
    mod strided_along_300_long_four_wide {
        use super::*;
        every_operation!(CubeStridedMicrokernelLauncher { planes: 4 }, Along, 300, 4);
    }

    /// The same across the reduced dimension, each lane its own row.
    mod strided_across_300_long_four_wide {
        use super::*;
        every_operation!(CubeStridedMicrokernelLauncher { planes: 4 }, Across, 300, 4);
    }

    const PARTIALS: usize = 4;
    const WIDTH: usize = 4;

    /// A line of f32 values a partial, and a line in place of the indices.
    #[test]
    fn values_alone_stage_a_line_each_and_one_more() {
        let item = ReduceWorkItem::along_f32(ReduceOperation::Sum);
        let line = WIDTH * size_of::<f32>();
        assert_eq!(staged_bytes(item, PARTIALS, WIDTH), PARTIALS * line + line);
    }

    #[test]
    fn indexed_values_stage_two_lines_each() {
        let item = ReduceWorkItem::along_f32(ReduceOperation::ArgMax);
        let line = WIDTH * size_of::<f32>();
        assert_eq!(staged_bytes(item, PARTIALS, WIDTH), 2 * PARTIALS * line);
    }
}
