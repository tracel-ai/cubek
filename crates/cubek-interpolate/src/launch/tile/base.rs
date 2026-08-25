use super::{
    coordinate::Rational,
    filter::{BicubicFilter, BilinearFilter, Lanczos3Filter, NearestFilter, SeparableFilterFamily},
    geometry::TileGeometry,
    kernel::interpolate_tile_kernel,
    space::{self, CHANNEL},
};
use crate::{
    InterpolateError,
    definition::{InterpolateMode, InterpolateOptions, get_transform},
};
use cubecl::{Runtime, client::ComputeClient, ir::ElemType, prelude::*};
use cubek_tile::{Operand, Residence};

/// Every choice the tile-backed interpolation launch makes.
///
/// Nothing here is inferred. The launch has no default and no derivation: a caller states the
/// geometry and the gathered input's residence, and gets exactly that. Only the lane split is
/// solved, by [`TileGeometry::from_config`], because the space asserts an exact plane cover.
///
/// [`channel_block`](Self::channel_block) is the one choice inside that split a caller may still
/// pin. It is the lane's channel run, so it is the accumulator's innermost extent and sets `nr`
/// in the contraction: the separable schedule's cost is per tap, and `nr` multiplies it. Solving
/// it only ever reaches the widest divisor one line holds, which leaves the other splits of a
/// deep channel axis unreachable.
///
/// The output is always written directly to global memory. Only the gathered input can be staged,
/// so `InPlace` makes the whole tile operation in-place while `Smem` stages that input. Which one
/// a problem wants swings both ways by up to 4x, which is why it is stated rather than guessed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileConfig {
    pub input_residence: Residence,
    pub planes_per_cube: usize,
    pub rows_per_plane: usize,
    pub cols_per_lane: usize,
    /// The lane's channel run, `None` to solve it with the rest of the lane split.
    pub channel_block: Option<usize>,
}

impl TileConfig {
    pub const fn new(
        input_residence: Residence,
        planes_per_cube: usize,
        rows_per_plane: usize,
        cols_per_lane: usize,
    ) -> Self {
        Self {
            input_residence,
            planes_per_cube,
            rows_per_plane,
            cols_per_lane,
            channel_block: None,
        }
    }

    /// Pin the lane's channel run rather than solving it. Must divide the channel count.
    pub const fn with_channel_block(self, block: usize) -> Self {
        Self {
            channel_block: Some(block),
            ..self
        }
    }

    /// The geometry these choices describe, over a plane of `lanes`.
    pub fn geometry(&self, channels: usize, lanes: usize) -> TileGeometry {
        TileGeometry::from_config(
            channels,
            lanes,
            self.planes_per_cube,
            self.rows_per_plane,
            self.cols_per_lane,
            self.channel_block,
        )
    }
}

/// Launch the tile-backed interpolation implementation for NHWC tensors.
///
/// The config is required: this path is under evaluation and every choice it makes is meant to be
/// stated by whatever is measuring it.
pub fn interpolate_tile_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    options: InterpolateOptions,
    dtype: ElemType,
    config: TileConfig,
) -> Result<(), InterpolateError> {
    let geometry = config.geometry(
        output.shape[3],
        client.properties().hardware.plane_size_max as usize,
    );
    match options.mode {
        InterpolateMode::Nearest(_) => {
            launch::<R, NearestFilter>(client, input, output, options, dtype, geometry, config)
        }
        InterpolateMode::Bilinear => {
            launch::<R, BilinearFilter>(client, input, output, options, dtype, geometry, config)
        }
        InterpolateMode::Bicubic => {
            launch::<R, BicubicFilter>(client, input, output, options, dtype, geometry, config)
        }
        InterpolateMode::Lanczos3 => {
            launch::<R, Lanczos3Filter>(client, input, output, options, dtype, geometry, config)
        }
    }
}

fn launch<R: Runtime, F: SeparableFilterFamily>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    options: InterpolateOptions,
    dtype: ElemType,
    geometry: TileGeometry,
    config: TileConfig,
) -> Result<(), InterpolateError> {
    let (input_h, input_w, output_h, output_w) = (
        input.shape[1],
        input.shape[2],
        output.shape[1],
        output.shape[2],
    );
    assert!(
        input_h <= i32::MAX as usize
            && input_w <= i32::MAX as usize
            && output_h <= i32::MAX as usize
            && output_w <= i32::MAX as usize
    );
    let row = Rational::of(get_transform(input_h, output_h, options));
    let col = Rational::of(get_transform(input_w, output_w, options));
    let lanes = client.properties().hardware.plane_size_max as usize;
    let space = space::interpolate_space(
        output.shape[0],
        output_h,
        output_w,
        output.shape[3],
        lanes,
        F::TAPS,
        geometry,
        space::instruction(client),
    );
    let launch = space.launcher_over(client, &[]);

    let vector_size = launch.vector_size(
        CHANNEL,
        &[(&input, &[CHANNEL]), (&output, &[CHANNEL])],
        dtype.size(),
    );

    let in_bounds = tap_range_in_bounds(row, output_h, input_h, F::TAPS, F::radius())
        && tap_range_in_bounds(col, output_w, input_w, F::TAPS, F::radius());

    // The channel block is the lane's channel run, so it is the width the contraction wants its
    // lines in. Where the tensor's own channel count cannot serve them (`C = 3` has no 4-aligned
    // row start, so `vector_size` above is 1), a shared-memory stage still can: it pads the axis
    // out to whole lines, and the contraction runs `4` wide against a scalar output. Only a width
    // the device actually serves is worth asking for, and only over a stage that exists.
    let residence = config.input_residence;
    let stage_width = (residence == Residence::Smem
        && geometry.channel_block != vector_size
        && client
            .io_optimized_vector_sizes(dtype.size())
            .any(|v| v == geometry.channel_block))
    .then_some(geometry.channel_block);

    // A padded stage reads the lanes past the real channel count, so those reads have to be the
    // masked kind whatever the taps do: unchecked they would take the next pixel's channels, and
    // run off the buffer entirely on the last one. Their values never reach the output (the sink's
    // own overhang mask drops those columns), but the reads still have to be in bounds.
    let checked = !in_bounds || stage_width.is_some();

    let max_smem = client.properties().hardware.max_shared_memory_size;
    let requested_smem = space::stage_window_bytes(
        row,
        col,
        F::TAPS,
        F::radius(),
        geometry,
        stage_width.unwrap_or(vector_size),
        dtype.size(),
    );

    // The residence is stated, so the only thing left to decide is whether the device can serve
    // it. Capacity is a hard limit rather than a preference, so it refuses instead of falling back.
    if residence == Residence::Smem && requested_smem > max_smem {
        return Err(InterpolateError::SharedMemoryLimitExceeded {
            requested: requested_smem,
            available: max_smem,
        });
    }
    let mut in_operand = Operand::new(
        &[
            space::BATCH,
            space::OUTPUT_H,
            space::OUTPUT_W,
            space::TAP_H,
            space::TAP_W,
            space::CHANNEL,
        ],
        dtype,
    );
    in_operand.stage(residence);
    in_operand.stage(Residence::InPlace);
    in_operand.stage(Residence::InPlace);

    let mut input_arg = launch
        .arg(input)
        .operand(&in_operand)
        .gathered(space::input_projection(row, col, F::radius()))
        .checked(checked)
        .with_boundary(checked.then_some(F::BOUNDARY))
        .vectorize(vector_size);
    if let Some(width) = stage_width {
        input_arg = input_arg.stage_width(width);
    }
    let input_arg = input_arg.build();
    let output_arg = launch
        .arg(output)
        .subspace(&[space::BATCH, space::OUTPUT_H, space::OUTPUT_W, CHANNEL])
        .vectorize(vector_size)
        .build();
    interpolate_tile_kernel::launch::<F, R>(
        client,
        launch.cube_count(),
        launch.cube_dim(),
        vector_size,
        input_arg.arg(),
        output_arg.arg(),
        row.scale as u32,
        row.offset as i32,
        row.divisor as u32,
        col.scale as u32,
        col.offset as i32,
        col.divisor as u32,
        F::radius(),
        launch.space().clone(),
        dtype,
    );
    Ok(())
}

fn tap_range_in_bounds(
    source: Rational,
    outputs: usize,
    input: usize,
    taps: usize,
    radius: usize,
) -> bool {
    let first = source.offset.div_euclid(source.divisor as isize) - radius as isize;
    let numerator = (outputs.saturating_sub(1) as isize) * source.scale as isize + source.offset;
    let last = numerator.div_euclid(source.divisor as isize) + (taps - 1 - radius) as isize;
    first >= 0 && last < input as isize
}
