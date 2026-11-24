use crate::reduce::testing::{F32, LaunchedRows, bound};
use crate::reduce::write::UnitLinesWriteMicrokernel;
use crate::reduce::{Accumulable, BATCH, ReduceOperation, ReduceWorkItem};
use cubecl::client::Client;
use cubecl::ir::{ElemType, IntKind};
use cubecl::tile::kind::Boundary;
use cubecl::tile::launch::{Decline, Grid};
use cubecl::tile::{Carrier, Indexed, Launcher, Monoid, Partitioning, TileTensor};
use cubecl::zspace::shape;
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeChecker, TestInput, Written};

const I32: ElemType = ElemType::Int(IntKind::I32);
/// Rows side by side in a line, each lane its own.
const LINE_LANE_COUNT: usize = 4;

/// Checks the lines' write of a carrier of values and indices, the values to one output and the indices
/// to another beside it.
pub(crate) struct BothHalvesScopeChecker {
    hw: Hardware,
}

impl BothHalvesScopeChecker {
    /// The checker on the test device.
    pub(crate) fn new() -> Self {
        Self {
            hw: Hardware::from_client(&cubecl::test_device().client()),
        }
    }
}

impl ScopeChecker for BothHalvesScopeChecker {
    /// The index every value is written with.
    type Case = u32;

    fn launch(&self, client: &Client, index: &Self::Case) -> Result<Vec<Written>, Decline> {
        let rows = LaunchedRows {
            item: ReduceWorkItem {
                operation: ReduceOperation::ArgMax,
                input: F32,
                accumulator: F32,
                lines_along: BATCH,
                with_indices: true,
            },
            row_count: LINE_LANE_COUNT,
            value_count: 1,
            line_lane_count: LINE_LANE_COUNT,
            plane_lane_count: self.hw.plane_dim().unwrap_or(1) as usize,
            out: F32,
        };
        let of = |element, value: f32| {
            TestInput::builder(client.clone(), shape![1, LINE_LANE_COUNT])
                .dtype(element)
                .custom(vec![value; LINE_LANE_COUNT])
                .generate_without_host_data()
        };
        let (input, out, indices) = (of(F32, 3.0), of(F32, -7.0), of(I32, -7.0));
        let space = rows.space();
        let partitioning = Partitioning::new(space.clone(), rows.unit_levels().build());
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(rows.plane_lane_count as u32),
        };
        let launcher = Launcher::new(client, partitioning, &space, grid)?;
        let lines = rows.input_dimensions();
        let cells = rows.out_dimensions();
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and the
        // kernel indexes no buffer but through its tiles.
        unsafe {
            both_halves_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                rows.line_lane_count,
                rows.bound_input(input.binding().into_tensor_arg(), &lines),
                rows.bound_out(out.clone().binding()),
                bound(
                    indices.clone().binding().into_tensor_arg(),
                    I32,
                    rows.line_lane_count,
                    &cells,
                    Boundary::Zero,
                ),
                launcher.partitioning_arg(),
                *index,
                UnitLinesWriteMicrokernel::new(rows.item),
                [F32, F32, I32],
            );
        }
        Ok(vec![
            Written {
                name: "values",
                tensor: out,
                host: HostDataType::F32,
            },
            Written {
                name: "indices",
                tensor: indices,
                host: HostDataType::F32,
            },
        ])
    }

    fn reference(&self, index: &Self::Case) -> Vec<HostData> {
        [3.0, *index as f32]
            .map(|value| {
                let line = HostDataVec::F32(vec![value; LINE_LANE_COUNT]);
                HostData::row_major(line, shape![1, LINE_LANE_COUNT])
            })
            .to_vec()
    }

    fn tolerance(&self, _index: &Self::Case) -> f32 {
        0.0
    }
}

/// Every unit's one line of `input`, each value with `index`, written to `out` and the indices to
/// `indices`.
///
/// Its operands' leaves:
/// - `input`: one line a unit.
/// - `out`, `indices`: one tensor each of a cell a row.
#[cube(launch_unchecked)]
fn both_halves_kernel<Acc: Numeric, Out: Numeric, Idx: Numeric, N: Size>(
    input: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    indices: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] index: u32,
    #[comptime] microkernel: UnitLinesWriteMicrokernel,
    #[define(Acc, Out, Idx)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);
    let i = indices.tile_mut(&partitioning);

    for unit in &partitioning {
        let row = x.at(&unit);
        let mut row_out = o.at(&unit);
        let slot = 0usize;
        let mut carrier = Indexed::<Acc, N>::identity(Monoid::Max);
        Indexed::<Acc, N>::set_line(
            &mut carrier,
            slot,
            row.flat::<Acc, N>().read(slot),
            Vector::new(index),
        );
        microkernel
            .runtime()
            .write::<Out, Idx, Acc, N, N, Indexed<Acc, N>>(
                carrier,
                &mut row_out,
                &ComptimeOption::new_Some(i.at(&unit)),
                1,
            );
    }
}
