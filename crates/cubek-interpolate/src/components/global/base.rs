use crate::{
    components::{
        global::{TileSize, tile_absolute_coords},
        readers::{GlobalMemoryReader, Reader, ReaderExpand, SharedMemoryReader},
        writers::Writer,
    },
    definition::{
        InterpolateMode, InterpolateOptions, InterpolatePrecision, MemoryStrategy, NearestMode,
        compute_weights, get_halo,
    },
};
use cubecl::{prelude::*, std::FastDivmod};

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn interpolate_kernel<EI: Float, EA: Float, N: Size>(
    input: &Tensor<Vector<EI, N>>,
    output: &mut Tensor<Vector<EI, N>>,
    cube_shape: Sequence<FastDivmod<usize>>,
    #[comptime] output_tile_size: TileSize,
    #[comptime] options: InterpolateOptions,
    #[comptime] memory_strategy: MemoryStrategy,
    #[comptime] smem_width: usize,
    #[comptime] smem_height: usize,
    #[define(EI)] _dtype: StorageType,
    #[define(EA)] _acc_dtype: StorageType,
) {
    interpolate_kernel_inner::<(EI, EA), N>(
        input,
        output,
        cube_shape,
        output_tile_size,
        options,
        memory_strategy,
        smem_width,
        smem_height,
    );
}

#[cube]
fn interpolate_kernel_inner<P: InterpolatePrecision, N: Size>(
    input: &Tensor<Vector<P::EI, N>>,
    output: &mut Tensor<Vector<P::EI, N>>,
    cube_shape: Sequence<FastDivmod<usize>>,
    #[comptime] output_tile_size: TileSize,
    #[comptime] options: InterpolateOptions,
    #[comptime] memory_strategy: MemoryStrategy,
    #[comptime] smem_width: usize,
    #[comptime] smem_height: usize,
) {
    let (batch, cube_pos, unit_pos, channel_group) = decompose_index(ABSOLUTE_POS, cube_shape);

    let (output_width, output_height) = (output.shape(2), output.shape(1));
    let (input_width, input_height) = (input.shape(2), input.shape(1));

    let (x, y) = tile_absolute_coords(output_width, cube_pos, unit_pos, output_tile_size);

    if x >= output_width || y >= output_height {
        terminate!();
    }

    let (mapped_x, mapped_y) = compute_input_coords::<P::EA>(
        x,
        y,
        input_width,
        input_height,
        output_width,
        output_height,
        options,
    );

    let base_x_floor = mapped_x.floor();
    let base_y_floor = mapped_y.floor();

    let (frac_x, frac_y) = (mapped_x - base_x_floor, mapped_y - base_y_floor);

    let (base_x, base_y) = (
        isize::cast_from(base_x_floor),
        isize::cast_from(base_y_floor),
    );

    let (weights_x, weights_y) = (
        compute_weights(frac_x, options),
        compute_weights(frac_y, options),
    );

    let vector_size = input.vector_size();

    let final_value = match comptime!(memory_strategy) {
        MemoryStrategy::Global => {
            let reader = <GlobalMemoryReader as Reader<P::EA, N>>::init::<P::EI>(
                input,
                batch,
                channel_group,
                vector_size,
                input_width,
                input_height,
                0,
                0,
                comptime!(smem_width),
                comptime!(smem_height),
            );
            compute_value_reader::<P, N, GlobalMemoryReader>(
                input,
                input_width,
                input_height,
                base_x,
                base_y,
                weights_x,
                weights_y,
                options,
                &reader,
            )
        }
        MemoryStrategy::Shared => {
            let halo = comptime!(get_halo(options.mode));
            let radius_offset = (halo - 1) / 2;

            let min_input_x = isize::cast_from(base_x) - radius_offset as isize;
            let min_input_y = isize::cast_from(base_y) - radius_offset as isize;

            let min_x = min_input_x.max(0) as usize;
            let min_y = min_input_y.max(0) as usize;

            let reader = <SharedMemoryReader<P::EA, N> as Reader<P::EA, N>>::init::<P::EI>(
                input,
                batch,
                channel_group,
                vector_size,
                input_width,
                input_height,
                min_x,
                min_y,
                comptime!(smem_width),
                comptime!(smem_height),
            );
            compute_value_reader::<P, N, SharedMemoryReader<P::EA, N>>(
                input,
                input_width,
                input_height,
                base_x,
                base_y,
                weights_x,
                weights_y,
                options,
                &reader,
            )
        }
    };

    let final_value = Vector::cast_from(final_value);

    let writer = Writer::new(channel_group);

    writer.write(output, batch, x, y, vector_size, final_value);
}

#[cube]
fn decompose_index(
    index: usize,
    cube_shape: Sequence<FastDivmod<usize>>,
) -> (usize, usize, usize, usize) {
    let (rem, channel_group) = cube_shape[0].div_mod(index);
    let (rem, unit_pos) = cube_shape[1].div_mod(rem);
    let (batch, cube_pos) = cube_shape[2].div_mod(rem);
    (batch, cube_pos, unit_pos, channel_group)
}

// Computes the input coordinates corresponding to an output coordinates.
#[cube]
fn compute_input_coords<EA: Float>(
    x: usize,
    y: usize,
    input_width: usize,
    input_height: usize,
    output_width: usize,
    output_height: usize,
    #[comptime] options: InterpolateOptions,
) -> (EA, EA) {
    let mapped_x = get_input_coord::<EA>(x, input_width, output_width, options);
    let mapped_y = get_input_coord::<EA>(y, input_height, output_height, options);
    (mapped_x, mapped_y)
}

#[cube]
fn get_input_coord<EA: Float>(
    x: usize,
    input_size: usize,
    output_size: usize,
    #[comptime] options: InterpolateOptions,
) -> EA {
    match options.mode {
        InterpolateMode::Nearest(nearest_mode) => match nearest_mode {
            NearestMode::Exact => {
                (EA::cast_from(x) + EA::new(0.5)) * EA::cast_from(input_size)
                    / EA::cast_from(output_size)
            }
            NearestMode::Floor => {
                (EA::cast_from(x) * EA::cast_from(input_size)) / EA::cast_from(output_size)
            }
        },
        _ => {
            if options.align_corners {
                let is_valid_output = (output_size > 1) as usize;
                let safe_denominator = (output_size - 1).max(1);

                EA::cast_from(x * (input_size - 1) * is_valid_output)
                    / EA::cast_from(safe_denominator)
            } else {
                (EA::cast_from(x) + EA::new(0.5)) * EA::cast_from(input_size)
                    / EA::cast_from(output_size)
                    - EA::new(0.5)
            }
        }
    }
}

#[cube]
fn compute_value_reader<P: InterpolatePrecision, N: Size, R: Reader<P::EA, N>>(
    input: &Tensor<Vector<P::EI, N>>,
    input_width: usize,
    input_height: usize,
    base_x: isize,
    base_y: isize,
    weights_x: Array<Vector<P::EA, N>>,
    weights_y: Array<Vector<P::EA, N>>,
    #[comptime] options: InterpolateOptions,
    reader: &R,
) -> Vector<P::EA, N> {
    let halo = comptime!(get_halo(options.mode));
    let radius_offset = (halo - 1) / 2;

    let mut final_value = Vector::zeroed();
    let mut total_weight = Vector::zeroed();

    #[unroll]
    for i in 0..halo {
        let mut row_value = Vector::zeroed();
        let mut row_weight_sum = Vector::zeroed();

        let y = base_y + i as isize - radius_offset as isize;

        #[unroll]
        for j in 0..halo {
            let x = base_x + j as isize - radius_offset as isize;

            let is_in_bounds =
                is_in_bounds(x, input_width, options) && is_in_bounds(y, input_height, options);

            let clamped_y = y.max(0).min(input_height as isize - 1) as usize;
            let clamped_x = x.max(0).min(input_width as isize - 1) as usize;
            let weight_x = weights_x[j];

            row_value += select(
                is_in_bounds,
                reader.read_weighted::<P::EI>(input, clamped_y, clamped_x, weight_x),
                Vector::zeroed(),
            );
            row_weight_sum += select(is_in_bounds, weight_x, Vector::zeroed());
        }

        let weight_y = weights_y[i];
        final_value += row_value * weight_y;
        total_weight += row_weight_sum * weight_y;
    }

    let epsilon = Vector::cast_from(P::EA::new(1e-7));

    final_value / total_weight.max(epsilon)
}

// Only used for bounds checking in Lanczos3 mode.
#[cube]
fn is_in_bounds(value: isize, size: usize, #[comptime] options: InterpolateOptions) -> bool {
    match options.mode {
        InterpolateMode::Lanczos3 => value >= 0 && value < size as isize,
        _ => true,
    }
}
