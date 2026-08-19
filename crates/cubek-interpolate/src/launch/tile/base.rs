use super::{
    coordinate::Rational,
    filter::{BicubicFilter, BilinearFilter, Lanczos3Filter, NearestFilter, SeparableFilterFamily},
    geometry::TileGeometry,
    kernel::interpolate_tile_kernel,
    space::{self, CHANNEL, LEAF},
};
use crate::{
    InterpolateError,
    definition::{InterpolateMode, InterpolateOptions, get_transform},
};
use cubecl::{Runtime, client::ComputeClient, ir::ElemType, prelude::*};
use cubek_tile::*;

/// Launch the tile-backed interpolation implementation for NHWC tensors.
pub fn interpolate_tile_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    options: InterpolateOptions,
    dtype: ElemType,
) -> Result<(), InterpolateError> {
    let geometry = TileGeometry::heuristic(
        output.shape[3],
        client.properties().hardware.plane_size_max as usize,
    );
    interpolate_tile_launch_with(client, input, output, options, dtype, geometry)
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
    match options.mode {
        InterpolateMode::Nearest(_) => {
            launch_with::<R, NearestFilter>(client, input, output, options, dtype, geometry)
        }
        InterpolateMode::Bilinear => {
            launch_with::<R, BilinearFilter>(client, input, output, options, dtype, geometry)
        }
        InterpolateMode::Bicubic => {
            launch_with::<R, BicubicFilter>(client, input, output, options, dtype, geometry)
        }
        InterpolateMode::Lanczos3 => {
            launch_with::<R, Lanczos3Filter>(client, input, output, options, dtype, geometry)
        }
    }
}

fn launch_with<R: Runtime, F: SeparableFilterFamily>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    options: InterpolateOptions,
    dtype: ElemType,
    geometry: TileGeometry,
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
    // Clamped and zero-padded windows are scalar-only. Retain channel vectorization when the
    // complete tap range is provably in bounds, avoiding an unnecessary boundary window.
    let in_bounds = tap_range_in_bounds(row, output_h, input_h, F::TAPS, F::radius())
        && tap_range_in_bounds(col, output_w, input_w, F::TAPS, F::radius());
    let v = if in_bounds
        && space_has_no_overhang(output_h, output_w, lanes, geometry)
        && geometry.channel_block.is_multiple_of(4)
        && launch.vector_size(
            CHANNEL,
            &[(&input, &[CHANNEL]), (&output, &[CHANNEL])],
            dtype.size(),
        ) >= 4
    {
        4
    } else {
        1
    };
    let input_arg = launch
        .arg(input, LEAF)
        .gathered(space::input_projection(row, col, F::radius()))
        .checked(!in_bounds)
        .with_boundary((!in_bounds).then_some(F::BOUNDARY))
        .vectorize(v)
        .residence(&[Residence::Smem, Residence::InPlace])
        .build();
    let output_arg = launch
        .arg(output, LEAF)
        .subspace(&[space::BATCH, space::OUTPUT_H, space::OUTPUT_W, CHANNEL])
        .vectorize(v)
        .build();
    interpolate_tile_kernel::launch::<F, R>(
        client,
        launch.cube_count(),
        launch.cube_dim(),
        v,
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

fn space_has_no_overhang(
    height: usize,
    width: usize,
    lanes: usize,
    geometry: TileGeometry,
) -> bool {
    let cols_per_cube = if geometry.lanes_on_channels {
        geometry.cols_per_lane
    } else {
        lanes * geometry.cols_per_lane
    };
    height.is_multiple_of(geometry.rows_per_cube) && width.is_multiple_of(cols_per_cube)
}
