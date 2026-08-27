use super::{
    compute::interpolate_tile_kernel,
    coordinate::Rational,
    filter::{BicubicFilter, BilinearFilter, Lanczos3Filter, NearestFilter, SeparableFilterFamily},
    geometry::TileGeometry,
    space::{self, CHANNEL},
};
use crate::{
    InterpolateError, InterpolateStrategy,
    definition::{InterpolateForwardProblem, InterpolateMode, InterpolateOptions, get_transform},
};
use cubecl::{Runtime, client::ComputeClient, ir::ElemType, prelude::*};
use cubek_tile::Residence;

/// Launch the tile-backed interpolation implementation for NHWC tensors.
///
/// Resolves the strategy against the device and the problem, then dispatches on the mode.
pub(crate) fn interpolate_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    options: InterpolateOptions,
    dtype: ElemType,
    strategy: InterpolateStrategy,
) -> Result<(), InterpolateError> {
    let hardware = &client.properties().hardware;
    let problem = InterpolateForwardProblem::from_input_output_shapes(
        &input.shape,
        &[output.shape[1], output.shape[2]],
        options,
    );
    let blueprint = strategy.blueprint(hardware, &problem);
    blueprint.validate()?;

    if blueprint.input_residence == Residence::Smem && hardware.num_cpu_cores.is_some() {
        return Err(InterpolateError::SharedMemoryUnsupportedOnCpu);
    }

    // Where the input reads from when the device cannot hold the staged window, or `None` to
    // refuse instead.
    //
    // The window is sized from the real extents, which an autotune key only buckets: two problems
    // that share a key can want stages several-fold apart, so a cached result that staged when it
    // was measured can meet a device that will not serve it when it is reused. An intent is a
    // preference, so it reads in place there and stays launchable whatever it is handed. A stated
    // blueprint keeps the refusal, because a sweep that asked for a stage wants to be told it did
    // not get one rather than to record the in-place kernel under the staged name.
    let fallback = match strategy {
        InterpolateStrategy::Forced(_) => None,
        _ => Some(Residence::InPlace),
    };

    let geometry =
        TileGeometry::from_blueprint(blueprint, output.shape[3], hardware.plane_size_max as usize);
    let residence = blueprint.input_residence;
    match options.mode {
        InterpolateMode::Nearest(_) => launch::<R, NearestFilter>(
            client, input, output, options, dtype, geometry, residence, fallback,
        ),
        InterpolateMode::Bilinear => launch::<R, BilinearFilter>(
            client, input, output, options, dtype, geometry, residence, fallback,
        ),
        InterpolateMode::Bicubic => launch::<R, BicubicFilter>(
            client, input, output, options, dtype, geometry, residence, fallback,
        ),
        InterpolateMode::Lanczos3 => launch::<R, Lanczos3Filter>(
            client, input, output, options, dtype, geometry, residence, fallback,
        ),
    }
}

/// Dispatch `residence`, falling back where the device cannot hold the stage it asked for.
///
/// Capacity is only knowable once the space is built and its vectorization solved, so the fallback
/// reads the refusal rather than predicting it. Nothing has been dispatched by then.
#[allow(clippy::too_many_arguments)]
fn launch<R: Runtime, F: SeparableFilterFamily>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    options: InterpolateOptions,
    dtype: ElemType,
    geometry: TileGeometry,
    residence: Residence,
    fallback: Option<Residence>,
) -> Result<(), InterpolateError> {
    let Some(fallback) = fallback.filter(|_| residence == Residence::Smem) else {
        return dispatch::<R, F>(client, input, output, options, dtype, geometry, residence);
    };

    match dispatch::<R, F>(
        client,
        input.clone(),
        output.clone(),
        options,
        dtype,
        geometry,
        residence,
    ) {
        Err(InterpolateError::SharedMemoryLimitExceeded { .. }) => {
            dispatch::<R, F>(client, input, output, options, dtype, geometry, fallback)
        }
        result => result,
    }
}

fn dispatch<R: Runtime, F: SeparableFilterFamily>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    options: InterpolateOptions,
    dtype: ElemType,
    geometry: TileGeometry,
    residence: Residence,
) -> Result<(), InterpolateError> {
    let (input_h, input_w, output_h, output_w) = (
        input.shape[1],
        input.shape[2],
        output.shape[1],
        output.shape[2],
    );
    let row = Rational::of(get_transform(input_h, output_h, options));
    let col = Rational::of(get_transform(input_w, output_w, options));
    let lanes = client.properties().hardware.plane_size_max as usize;

    // Cheap to check before any of the space/vectorization work below: a cube this wide is
    // refused outright rather than built and then rejected by the device at dispatch.
    let max_units = client.properties().hardware.max_units_per_cube as usize;
    let units_per_cube = geometry.planes_per_cube.saturating_mul(lanes);
    if units_per_cube > max_units {
        return Err(InterpolateError::UnitsPerCubeExceeded {
            requested: units_per_cube,
            available: max_units,
        });
    }

    let (space, in_operand) = space::interpolate_space(
        output.shape[0],
        output_h,
        output_w,
        output.shape[3],
        lanes,
        F::mode_properties().taps,
        geometry,
        space::instruction(client),
        dtype,
        residence,
    );
    let launch = space.launcher_over(client, &[]);

    let vector_size = launch.vector_size(
        CHANNEL,
        &[(&input, &[CHANNEL]), (&output, &[CHANNEL])],
        dtype.size(),
    );

    let properties = F::mode_properties();
    let in_bounds = tap_range_in_bounds(row, output_h, input_h, properties.taps, F::radius())
        && tap_range_in_bounds(col, output_w, input_w, properties.taps, F::radius());

    // The channel block is the lane's channel run, so it is the width the contraction wants its
    // lines in. Where the tensor's own channel count cannot serve them (`C = 3` has no 4-aligned
    // row start, so `vector_size` above is 1), a shared-memory stage still can: it pads the axis
    // out to whole lines, and the contraction runs `4` wide against a scalar output. Only a width
    // the device actually serves is worth asking for, and only over a stage that exists.
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

    // Capacity is a hard limit, so it refuses here rather than trimming the window to fit. Whether
    // that refusal ends the launch or sends it back in place is the caller's to decide.
    if residence == Residence::Smem {
        let available = client.properties().hardware.max_shared_memory_size;
        let requested = space::stage_window_bytes(
            row,
            col,
            properties.taps,
            F::radius(),
            geometry,
            stage_width.unwrap_or(vector_size),
            dtype.size(),
        );
        if requested > available {
            return Err(InterpolateError::SharedMemoryLimitExceeded {
                requested,
                available,
            });
        }
    }

    let mut input_arg = launch
        .arg(input)
        .operand(&in_operand)
        .gathered(space::input_projection(row, col, F::radius()))
        .checked(checked)
        .with_boundary(checked.then_some(properties.boundary))
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
