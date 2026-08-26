use super::geometry::TileGeometry;
use cubecl::{Runtime, client::ComputeClient, ir::ElemType};
use cubek_tile::{
    Axis, Buffering, Compaction, ComputeScope, Coverage, CubeAxis, Cut, Distribution, Instruction,
    Operand, PhysicalAxisMap, Projection, RegisterBlock, Residence, Space, Spread, Tiling,
    WalkOrder,
};

pub const BATCH: Axis = Axis(0);
pub const OUTPUT_H: Axis = Axis(1);
pub const OUTPUT_W: Axis = Axis(2);
pub const TAP_H: Axis = Axis(3);
pub const TAP_W: Axis = Axis(4);
pub const CHANNEL: Axis = Axis(5);

/// The instruction leaf, which the device decides.
pub fn instruction<R: Runtime>(client: &ComputeClient<R>) -> Instruction {
    match client.properties().hardware.num_cpu_cores {
        Some(_) => Instruction::Registers {
            config: RegisterBlock::new(256).split_edge(),
        },
        None => Instruction::Registers {
            config: RegisterBlock::new(64).lane_fanout(),
        },
    }
}

/// Every level runs a single-slot ring. CHANNEL is the cube walk's only moving axis, and it moves
/// only past `lanes * 4` channels; below that the walk is one region, where a deeper pipeline has
/// no fill to hide and pays the ring's prologue and drain for nothing.
#[allow(clippy::too_many_arguments)]
pub fn interpolate_space(
    batch: usize,
    height: usize,
    width: usize,
    channels: usize,
    lanes: usize,
    taps: usize,
    geometry: TileGeometry,
    instruction: Instruction,
    dtype: ElemType,
    residence: Residence,
) -> (Space, Operand) {
    assert!(
        geometry.lane_cols * geometry.lane_channels == lanes,
        "interpolate_space: the lane split covers {} of the plane's {lanes} lanes",
        geometry.lane_cols * geometry.lane_channels
    );
    let channels_per_cube = geometry.channels_per_cube();
    let mut in_operand = (Operand::new(
        &[BATCH, OUTPUT_H, OUTPUT_W, TAP_H, TAP_W, CHANNEL],
        dtype,
    ),);
    let space = Tiling::over(
        &mut in_operand,
        &[
            (BATCH, batch),
            (OUTPUT_H, height),
            (OUTPUT_W, width),
            (TAP_H, taps),
            (TAP_W, taps),
            (CHANNEL, channels),
        ],
    )
    .level(WalkOrder::RowMajor, Buffering::SINGLE, |level, input| {
        level
            .axis(BATCH, Cut::cube(CubeAxis::Z, 1))
            .axis(OUTPUT_H, Cut::cube(CubeAxis::Y, geometry.rows_per_cube()))
            .axis(OUTPUT_W, Cut::cube(CubeAxis::X, geometry.cols_per_cube()))
            .axis(TAP_H, Cut::sequential(taps))
            .axis(TAP_W, Cut::sequential(taps))
            .axis(CHANNEL, Cut::sequential(channels_per_cube));
        input.0.stage(residence);
    })
    .level(WalkOrder::RowMajor, Buffering::SINGLE, |level, _| {
        level
            .axis(BATCH, Cut::sequential(1))
            .axis(OUTPUT_H, Cut::plane(geometry.rows_per_plane))
            .axis(OUTPUT_W, Cut::sequential(geometry.cols_per_cube()))
            .axis(TAP_H, Cut::sequential(taps))
            .axis(TAP_W, Cut::sequential(taps))
            .axis(CHANNEL, Cut::sequential(channels_per_cube));
    })
    .instruction(instruction, |level, _| {
        level
            .axis(BATCH, Cut::sequential(1))
            .axis(OUTPUT_H, Cut::sequential(geometry.rows_per_plane))
            .axis(
                OUTPUT_W,
                lanes_over(geometry.lane_cols, geometry.cols_per_lane),
            )
            .axis(TAP_H, Cut::sequential(taps))
            .axis(TAP_W, Cut::sequential(taps))
            .axis(
                CHANNEL,
                lanes_over(geometry.lane_channels, geometry.channel_block),
            );
    })
    .build();

    (space, in_operand.0)
}

/// `edge`-sized tiles dealt to `instances` lanes of the plane. [`Cut::unit`] claims the whole plane
/// for one axis; the interpolation splits it across two, so the count is stated outright. One lane
/// is no split at all, and stays sequential so the walk keeps its comptime coordinate.
fn lanes_over(instances: usize, edge: usize) -> Cut {
    match instances {
        1 => Cut::sequential(edge),
        n => Cut::new(
            edge,
            Distribution::Spatial {
                scope: ComputeScope::Unit,
                spread: Spread::Contiguous,
                coverage: Coverage::Instances(n),
            },
        ),
    }
}

pub fn input_projection(
    row: super::coordinate::Rational,
    col: super::coordinate::Rational,
    radius: usize,
) -> Projection {
    Projection::new(
        &[BATCH, OUTPUT_H, OUTPUT_W, TAP_H, TAP_W, CHANNEL],
        &[
            PhysicalAxisMap::of(BATCH),
            row.tap_axis(OUTPUT_H, TAP_H, radius),
            col.tap_axis(OUTPUT_W, TAP_W, radius),
            PhysicalAxisMap::of(CHANNEL),
        ],
    )
}

/// The number of bytes the gathered input would require if staged into shared memory.
pub fn stage_window_bytes(
    row: super::coordinate::Rational,
    col: super::coordinate::Rational,
    taps: usize,
    radius: usize,
    geometry: TileGeometry,
    vector_size: usize,
    elem_size: usize,
) -> usize {
    let extent_of = |axis| match axis {
        BATCH => 1,
        OUTPUT_H => geometry.rows_per_cube(),
        OUTPUT_W => geometry.cols_per_cube(),
        TAP_H | TAP_W => taps,
        CHANNEL => geometry.channels_per_cube(),
        other => panic!("stage_window_bytes: {other:?} is not an axis of the interpolation space"),
    };
    let window_vectors: usize =
        Compaction::of(&input_projection(row, col, radius), vector_size, extent_of)
            .extents()
            .iter()
            .product();
    window_vectors * vector_size * elem_size
}
