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
use cubek_tile::*;

/// The staging policy for the tile-backed interpolation input.
///
/// The output is always written directly to global memory. Only the gathered input can be staged,
/// so `InPlace` makes the whole tile operation in-place while `Smem` stages that input. Which one
/// a problem wants swings both ways by up to 4x, so the default derives it
/// ([`stage_input`](space::stage_input)) and a stated one is an override, for measuring the
/// derivation against the alternative.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TileConfig {
    input_residence: Option<Residence>,
}

impl TileConfig {
    /// Let the tap window decide where the input lives.
    pub const fn auto() -> Self {
        Self {
            input_residence: None,
        }
    }

    /// Pin the input's residence, whatever the tap window says.
    pub const fn forced(input_residence: Residence) -> Self {
        Self {
            input_residence: Some(input_residence),
        }
    }
}

/// Launch the tile-backed interpolation implementation for NHWC tensors.
pub fn interpolate_tile_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    options: InterpolateOptions,
    dtype: ElemType,
) -> Result<(), InterpolateError> {
    interpolate_tile_launch_configured(client, input, output, options, dtype, TileConfig::default())
}

/// [`interpolate_tile_launch`] with an explicit input staging policy.
pub fn interpolate_tile_launch_configured<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    options: InterpolateOptions,
    dtype: ElemType,
    config: TileConfig,
) -> Result<(), InterpolateError> {
    let geometry = TileGeometry::heuristic(
        output.shape[3],
        client.properties().hardware.plane_size_max as usize,
    );
    interpolate_tile_launch_with_config(client, input, output, options, dtype, geometry, config)
}

/// [`interpolate_tile_launch`] with the tile geometry stated rather than inferred.
pub fn interpolate_tile_launch_with<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    options: InterpolateOptions,
    dtype: ElemType,
    geometry: TileGeometry,
) -> Result<(), InterpolateError> {
    interpolate_tile_launch_with_config(
        client,
        input,
        output,
        options,
        dtype,
        geometry,
        TileConfig::default(),
    )
}

/// [`interpolate_tile_launch_with`] with an explicit input staging policy.
pub fn interpolate_tile_launch_with_config<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    options: InterpolateOptions,
    dtype: ElemType,
    geometry: TileGeometry,
    config: TileConfig,
) -> Result<(), InterpolateError> {
    match options.mode {
        InterpolateMode::Nearest(_) => {
            launch_with::<R, NearestFilter>(client, input, output, options, dtype, geometry, config)
        }
        InterpolateMode::Bilinear => launch_with::<R, BilinearFilter>(
            client, input, output, options, dtype, geometry, config,
        ),
        InterpolateMode::Bicubic => {
            launch_with::<R, BicubicFilter>(client, input, output, options, dtype, geometry, config)
        }
        InterpolateMode::Lanczos3 => launch_with::<R, Lanczos3Filter>(
            client, input, output, options, dtype, geometry, config,
        ),
    }
}

fn launch_with<R: Runtime, F: SeparableFilterFamily>(
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
    );
    let launch = space.launcher_over(client, &[]);

    let vector_size = launch.vector_size(
        CHANNEL,
        &[(&input, &[CHANNEL]), (&output, &[CHANNEL])],
        dtype.size(),
    );

    let in_bounds = tap_range_in_bounds(row, output_h, input_h, F::TAPS, F::radius())
        && tap_range_in_bounds(col, output_w, input_w, F::TAPS, F::radius());

    let residence = config.input_residence.unwrap_or_else(|| {
        space::stage_input(row, col, F::TAPS, F::radius(), geometry, vector_size)
    });
    let leaf = space::leaf(client);
    let input_arg = launch
        .arg(input, leaf)
        .gathered(space::input_projection(row, col, F::radius()))
        .checked(!in_bounds)
        .with_boundary((!in_bounds).then_some(F::BOUNDARY))
        .vectorize(vector_size)
        .residence(&[residence, Residence::InPlace, Residence::InPlace])
        .build();
    let output_arg = launch
        .arg(output, leaf)
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
