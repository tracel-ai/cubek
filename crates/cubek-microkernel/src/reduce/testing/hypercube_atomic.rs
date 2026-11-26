use crate::reduce::cube::CubeStagedReduceMicrokernel;
use crate::reduce::hypercube::HypercubeAtomicReduceMicrokernel;
use crate::reduce::testing::{
    ComponentLaunch, LANE_SLOT_COUNT, LaunchedRows, MicrokernelLauncher, cube_lanes,
};
use crate::reduce::{BATCH, REDUCED, ReduceCarrierFamily, ReduceWorkItem};
use cubecl::ir::features::AtomicUsage;
use cubecl::ir::{ElemType, Type};
use cubecl::tile::kind::Boundary;
use cubecl::tile::launch::{AccumulateArg, AccumulateArgLaunch, BoundaryPolicy, Decline};
use cubecl::tile::{Levels, Partitioning, Space, TileSpec, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::Hardware;

/// `parts` cubes of `planes` planes, each adding its part of the row into an output that holds zero.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HypercubeAtomicMicrokernelLauncher {
    planes: usize,
    parts: usize,
}

impl HypercubeAtomicMicrokernelLauncher {
    pub(crate) const fn new(planes: usize, parts: usize) -> Self {
        Self { planes, parts }
    }
}

/// Why a device without an atomic add of the output's element runs none.
const NO_ATOMIC_ADD: Decline = Decline::Unserved {
    why: "the device has no atomic add for the output's element",
};

impl MicrokernelLauncher for HypercubeAtomicMicrokernelLauncher {
    fn plane_width(&self, hw: &Hardware, item: &ReduceWorkItem) -> Result<usize, Decline> {
        cube_lanes(hw, item, self.planes)
    }

    /// The kernel adds into the output's element.
    fn serves_out(&self, hw: &Hardware, out: ElemType) -> Result<(), Decline> {
        let adds = hw
            .properties
            .atomic_type_usage(Type::atomic(out))
            .contains(AtomicUsage::Add);
        match adds {
            true => Ok(()),
            false => Err(NO_ATOMIC_ADD),
        }
    }

    fn unwritten(&self) -> f32 {
        0.0
    }

    fn launch_plane_width(&self, _hw: &Hardware, rows: &LaunchedRows) -> usize {
        rows.plane_lane_count * self.planes
    }

    fn space(&self, rows: &LaunchedRows) -> Space {
        rows.space()
    }

    fn levels(&self, _hw: &Hardware, rows: &LaunchedRows) -> Levels {
        let [(batch, owned), slots] = rows.owner();
        let part = rows.value_count / self.parts;
        Levels::leaf(&[(batch, owned), (REDUCED, part), slots]).cubes(&[BATCH, REDUCED])
    }

    fn launch_into<F: ReduceCarrierFamily>(&self, launch: ComponentLaunch<'_>) {
        let rows = launch.rows;
        let dimensions = rows.out_dimensions();
        let boundary = BoundaryPolicy::Every(Boundary::Zero);
        let out = AccumulateArgLaunch::new(
            launch.out.into_tensor_arg(),
            rows.owned(),
            TileSpec::direct(&dimensions).boundary(boundary),
        );
        let part = CubeStagedReduceMicrokernel::new(
            rows.item,
            rows.plane_lane_count,
            self.planes,
            LANE_SLOT_COUNT,
        );
        // SAFETY: both operands are bound at the problem's own extents, the kernel indexes no buffer
        // but through its tiles, and the output only receives atomic adds.
        unsafe {
            hypercube_atomic_reduce_kernel::launch_unchecked::<F>(
                launch.client,
                launch.launcher.cube_count(),
                launch.launcher.cube_dim(),
                rows.line_lane_count,
                rows.cells(),
                launch.input,
                out,
                launch.launcher.partitioning_arg(),
                HypercubeAtomicReduceMicrokernel::new(part),
                [rows.item.input, rows.item.accumulator, rows.out],
            )
        }
    }
}

/// Every cube's part of its rows, its sum added into `out`.
///
/// Its operands' leaves:
/// - `input`: the hypercube atomic's rows, laid along or across the reduced dimension.
/// - `out`: the hypercube atomic's cells, one a row, at zero.
#[cube(launch_unchecked)]
fn hypercube_atomic_reduce_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    K: Size,
    F: ReduceCarrierFamily,
>(
    input: &TileTensor<'_>,
    out: &mut AccumulateArg<'_, Out>,
    partitioning: Partitioning,
    #[comptime] microkernel: HypercubeAtomicReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = input.tile(&partitioning);
    let o = out.tile_mut(&partitioning);

    for cube in &partitioning {
        microkernel
            .runtime()
            .reduce::<In, Acc, Out, N, F::Carrier<Acc, N, K>>(&cube, &x, &o);
    }
}
