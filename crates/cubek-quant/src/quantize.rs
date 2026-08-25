use cubecl::{
    calculate_cube_count_elemwise,
    features::TypeUsage,
    ir::{ElemType, types::Fp8Format},
    post_processing::minifloat::f32_to_fp8_bits,
    prelude::*,
    std::{
        quant::{check_scale_bindings, fp4::float_to_e2m1_bits, round::round_up_to_dtype},
        tensor::{
            View, ViewMut, into_contiguous,
            layout::linear::{LinearView, LinearViewMut, linear_view},
        },
    },
    tensor_vector_size_parallel,
};

use crate::{
    layout::{ScalesLayout, ScalesViewMut, scales_view},
    scale::{GlobalScale, Scale, split_levels},
    utils::{check_block_size_compat, packed_storage_elem},
};
use crate::{
    layout::{ScalesView, scales_layout},
    scheme::{QuantMode, QuantScheme, QuantStore, QuantValue, ScaleDtype},
};

/// Scale a value into its quantization range, ready for the store to encode.
///
/// `Vector::round` is the *integer* formats' rounding, and only theirs: it snaps to whole numbers,
/// which is the grid `q8`/`q4`/`q2` store. A minifloat's grid is not the integers — `e2m1` holds
/// `0.5` and `1.5`, `e4m3` holds far more — so rounding here first would quantize it twice, the
/// second time onto the wrong grid, and the format would silently degrade to an integer one of the
/// same width. Those formats round where they are encoded, against the codes they actually have.
#[cube]
fn quantize_symmetric<F: Float, N: Size, FS: CubePrimitive>(
    value: Vector<F, N>,
    scale: Scale<FS>,
    range_min: F,
    range_max: F,
    #[comptime] quant: QuantValue,
) -> Vector<F, N> {
    let scaled = scale.quantize_symmetric::<F, N>(value);
    let snapped = if comptime![rounds_to_integers(quant)] {
        Vector::round(scaled)
    } else {
        scaled
    };
    clamp(snapped, Vector::new(range_min), Vector::new(range_max))
}

/// Whether the format's representable values are the integers in its range, which is what makes
/// [`Vector::round`] the right rounding for it.
fn rounds_to_integers(quant: QuantValue) -> bool {
    match quant {
        QuantValue::Q8F
        | QuantValue::Q8S
        | QuantValue::Q4F
        | QuantValue::Q4S
        | QuantValue::Q2F
        | QuantValue::Q2S => true,
        QuantValue::E5M2 | QuantValue::E4M3 | QuantValue::E2M1 => false,
    }
}

#[cube]
fn quantize_symmetric_q<F: Float, N: Size, FS: CubePrimitive, Q: Scalar>(
    value: Vector<F, N>,
    scale: Scale<FS>,
    range_min: F,
    range_max: F,
    #[comptime] quant: QuantValue,
) -> Vector<Q, N> {
    Vector::cast_from(quantize_symmetric::<F, N, FS>(
        value, scale, range_min, range_max, quant,
    ))
}

#[cube]
fn quantize_packed_value<F: Float, N: Size, FS: CubePrimitive, QS: Int>(
    value: Vector<F, N>,
    scale: Scale<FS>,
    range_min: F,
    range_max: F,
    #[comptime] scheme: QuantScheme,
) -> QS {
    let value = quantize_symmetric::<F, N, FS>(value, scale, range_min, range_max, scheme.value);
    pack_q::<F, N, QS>(value, scheme.value)
}

/// Pack a vector of quantized values into a single integer (the stored quantization type),
/// according to the specified quantization input type.
///
/// The field a value occupies is the format's *code*, not its magnitude, and the two only coincide
/// for the integer formats — which is why a minifloat is encoded rather than cast here. Casting
/// `e2m1` would truncate `0.5` and `1.5` to whole numbers and leave a format reconstructing on the
/// integer grid `{0, 1, 2, 3, 4, 5, 6}`; casting `e4m3` would do the same to its fractions and
/// then wrap, since its codes reach ±448 while a byte read as two's complement stops at ±127.
#[cube]
fn pack_q<F: Float, N: Size, QS: Int>(value: Vector<F, N>, #[comptime] quant: QuantValue) -> QS {
    let size_quant = quant.size_bits();

    let size_store = QS::size_bits().comptime();
    let num_quants = size_store / size_quant;

    let mask = (1 << size_quant) - 1;
    let mut packed = QS::from_int(0);

    let fields = match quant {
        QuantValue::E2M1 | QuantValue::E4M3 | QuantValue::E5M2 => {
            encode_minifloat::<F, N>(value, quant)
        }
        QuantValue::Q8F
        | QuantValue::Q8S
        | QuantValue::Q4F
        | QuantValue::Q4S
        | QuantValue::Q2F
        | QuantValue::Q2S => integer_fields::<F, N>(value),
    };

    // Shift and combine into QS (using i32 for sign extension)
    #[unroll]
    for position in 0..num_quants {
        let offset = QS::cast_from(position * size_quant);
        let shifted = QS::cast_from(i32::cast_from(fields.extract(position)) & mask) << offset;
        packed |= shifted;
    }

    packed
}

/// The field an integer format's value occupies, which is the value itself: its codes and its
/// magnitudes are the same numbers, and the mask the caller applies does the sign truncation.
#[cube]
fn integer_fields<F: Float, N: Size>(value: Vector<F, N>) -> Vector<u32, N> {
    Vector::<u32, N>::reinterpret(Vector::<i32, N>::cast_from(value))
}

/// The code a minifloat value occupies in its packed field.
///
/// Software throughout, so a field packs the same on a backend with no narrow float type as on one
/// with the intrinsic — and in this path there is no such type in play anyway, since the codes ride
/// in an integer rather than in a vector the backend could convert. Both codecs round to nearest
/// with ties to even and saturate, which is what the host types do.
#[cube]
fn encode_minifloat<F: Float, N: Size>(
    value: Vector<F, N>,
    #[comptime] quant: QuantValue,
) -> Vector<u32, N> {
    match quant {
        QuantValue::E2M1 => float_to_e2m1_bits::<F, N>(value),
        QuantValue::E4M3 => f32_to_fp8_bits::<N>(Vector::cast_from(value), Fp8Format::E4M3),
        QuantValue::E5M2 => f32_to_fp8_bits::<N>(Vector::cast_from(value), Fp8Format::E5M2),
        _ => comptime!(unreachable!("{quant:?} is not a minifloat")),
    }
}

#[cube]
fn write_scale<F: Float, FS: CubePrimitive>(
    in_pos: usize,
    scale: View<F, usize>,
    mut out_scale: ViewMut<FS, usize>,
    global: &GlobalScale,
    scales_layout: ScalesLayout,
    #[comptime] dtype: ScaleDtype,
) -> Scale<FS> {
    // Rounded up rather than cast to nearest, which can land below the scale calibration asked for
    // and clip every value at the block maximum. The CPU backends round up too, and both have to,
    // or a tensor quantized on one reconstructs differently on the other.
    let inner = FS::cast_from(round_up_to_dtype::<F>(scale.read(in_pos), dtype));

    // Write the scale into the output buffer
    if scales_layout.is_block_start(in_pos) {
        out_scale.write(in_pos, inner);
    }

    global.at::<FS>(inner)
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn quantize_symmetric_native_kernel<F: Float, N: Size, FS: Numeric, Q: Numeric>(
    input: LinearView<'_, Vector<F, N>>,
    scale: ScalesView<'_, F>,
    global: ComptimeOption<LinearView<'_, f32>>,
    range_min: InputScalar,
    range_max: InputScalar,
    mut output: LinearViewMut<'_, Vector<Q, N>>,
    out_scale: ScalesViewMut<'_, FS>,
    out_global: ComptimeOption<LinearViewMut<'_, f32>>,
    scales_layout: ScalesLayout,
    #[comptime] scheme: QuantScheme,
    #[define(F, FS, Q)] _dtypes: [ElemType; 3],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let native_packing = Q::packing_factor();
    let in_pos = ABSOLUTE_POS * input.vector_size() * native_packing;
    let global = GlobalScale::read(global);
    global.write(out_global);
    let scale = write_scale(
        in_pos,
        scale,
        out_scale,
        &global,
        scales_layout,
        scheme.scale_dtype(),
    );

    output.write(
        ABSOLUTE_POS,
        quantize_symmetric_q::<F, N, FS, Q>(
            input.read(ABSOLUTE_POS),
            scale,
            range_min.get::<F>(),
            range_max.get::<F>(),
            scheme.value,
        ),
    );
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn quantize_symmetric_packed_kernel<F: Float, N: Size, FS: Numeric, QS: Int>(
    input: LinearView<'_, Vector<F, N>>,
    scale: ScalesView<'_, F>,
    global: ComptimeOption<LinearView<'_, f32>>,
    range_min: InputScalar,
    range_max: InputScalar,
    mut output: LinearViewMut<'_, QS>,
    out_scale: ScalesViewMut<'_, FS>,
    out_global: ComptimeOption<LinearViewMut<'_, f32>>,
    scales_layout: ScalesLayout,
    #[comptime] scheme: QuantScheme,
    #[define(F, FS, QS)] _dtypes: [ElemType; 3],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let num_quants = scheme.num_quants();
    let packed_pos = ABSOLUTE_POS * num_quants;
    let global = GlobalScale::read(global);
    global.write(out_global);
    let scale = write_scale(
        packed_pos,
        scale,
        out_scale,
        &global,
        scales_layout,
        scheme.scale_dtype(),
    );

    if input.vector_size().comptime() == num_quants {
        output.write(
            ABSOLUTE_POS,
            quantize_packed_value::<F, N, FS, QS>(
                input.read(ABSOLUTE_POS),
                scale,
                range_min.get::<F>(),
                range_max.get::<F>(),
                scheme,
            ),
        );
    } else {
        // Input vector size = 1
        let size!(NQ) = num_quants;
        let mut values = Vector::<F, NQ>::empty();
        #[unroll]
        for i in 0..num_quants {
            values.insert(i, input.read(packed_pos + i).extract(0usize));
        }
        output.write(
            ABSOLUTE_POS,
            quantize_packed_value::<F, NQ, FS, QS>(
                values,
                scale,
                range_min.get::<F>(),
                range_max.get::<F>(),
                scheme,
            ),
        );
    }
}

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    scales: &[TensorBinding<R>],
    out_scales: &[TensorBinding<R>],
    scheme: &QuantScheme,
    input_elem: ElemType,
) -> Result<(), LaunchError> {
    check_scale_bindings(scheme, scales.len());
    check_scale_bindings(scheme, out_scales.len());

    let scale_dtype = ElemType::from_scale_dtype(scheme.scale_dtype());
    let (scale, global) = split_levels(scales);
    let (out_scale, out_global) = split_levels(out_scales);

    match scheme {
        QuantScheme {
            store: QuantStore::PackedU32(_),
            ..
        } => quantize_packed(
            client,
            input,
            scheme,
            scale,
            global,
            out_scale,
            out_global,
            output,
            input_elem,
            scale_dtype,
        ),
        QuantScheme {
            value: QuantValue::Q8F | QuantValue::Q8S | QuantValue::E4M3 | QuantValue::E5M2,
            store: QuantStore::Native,
            ..
        } => {
            if !i8::supported_uses(client).contains(TypeUsage::Conversion) {
                panic!(
                    "{:?} is not supported for native quantization",
                    scheme.value
                );
            }

            quantize_native(
                client,
                input,
                scheme,
                scale,
                global,
                out_scale,
                out_global,
                output,
                input_elem,
                scale_dtype,
            )
        }
        QuantScheme {
            store: QuantStore::Native | QuantStore::PackedNative(_),
            value,
            ..
        } => {
            panic!("{value:?} is not supported for native quantization");
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn quantize_native<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    scheme: &QuantScheme,
    scale: TensorBinding<R>,
    global: Option<TensorBinding<R>>,
    out_scale: TensorBinding<R>,
    out_global: Option<TensorBinding<R>>,
    output: TensorBinding<R>,
    input_dtype: ElemType,
    scale_dtype: ElemType,
) -> Result<(), LaunchError> {
    let num_elems: usize = input.shape.iter().product();
    let output_dtype = ElemType::from_quant_value(scheme.value);

    let candidates = client.io_optimized_vector_sizes(input_dtype.size().max(output_dtype.size()));
    let vector_size = tensor_vector_size_parallel(
        candidates,
        &input.shape,
        &input.strides,
        input.shape.len() - 1,
    );

    let working_units = num_elems / vector_size as usize;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);
    let (range_min, range_max) = scheme.value.range();

    match scheme {
        QuantScheme {
            mode: QuantMode::Symmetric,
            store: QuantStore::Native,
            ..
        } => {
            // We could use vector_size = block_size if it's in the supported vector sizes.. but let's keep it simple
            check_block_size_compat(scheme, vector_size as usize);

            let address_type = input
                .required_address_type(input_dtype.size())
                .max(scale.required_address_type(scale_dtype.size()))
                .max(output.required_address_type(output_dtype.size()));

            let scales_layout = scales_layout(&output, &scale, 1, scheme);

            unsafe {
                quantize_symmetric_native_kernel::launch_unchecked(
                    client,
                    cube_count,
                    cube_dim,
                    address_type,
                    vector_size,
                    linear_view(input),
                    // scale is computed based on input float dtype, but stored based on qparams precision
                    scales_view(output.clone(), scale, 1, scheme),
                    global.map(linear_view).into(),
                    InputScalar::new(range_min, input_dtype),
                    InputScalar::new(range_max, input_dtype),
                    linear_view(output.clone()),
                    scales_view(output, out_scale, 1, scheme),
                    out_global.map(linear_view).into(),
                    scales_layout,
                    *scheme,
                    [input_dtype, scale_dtype, output_dtype],
                )
            }
        }
        _ => panic!("Unsupported quantization scheme {scheme:?}"),
    };

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn quantize_packed<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    scheme: &QuantScheme,
    scale: TensorBinding<R>,
    global: Option<TensorBinding<R>>,
    out_scale: TensorBinding<R>,
    out_global: Option<TensorBinding<R>>,
    output: TensorBinding<R>,
    input_dtype: ElemType,
    scale_dtype: ElemType,
) -> Result<(), LaunchError> {
    let num_elems: usize = input.shape.iter().product();

    // Determine if we can use vectorized packing
    let mut can_vectorize = match scheme {
        QuantScheme {
            mode: QuantMode::Symmetric,
            store: QuantStore::PackedU32(dim),
            ..
        } => {
            // Check if packing dim is contiguous
            let ndims = input.shape.len();
            input.strides[ndims - 1 - *dim] == 1
        }
        QuantScheme { .. } => panic!("Unsupported quantization scheme {scheme:?}"),
    };
    // For larger tensors, copying to contiguous memory should be faster than scalar reads.
    // 2048 is a conservative floor for the threshold, could be tuned.
    let num_quants = scheme.num_quants();
    let input = if !can_vectorize && num_elems >= 2048 {
        can_vectorize = true;
        into_contiguous(client, input, input_dtype).binding()
    } else {
        input
    };

    // Elements to pack are strided, require scalar reads + manual gather
    let vector_size = if can_vectorize { num_quants } else { 1 };

    let working_units = num_elems.div_ceil(vector_size);
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);
    let (range_min, range_max) = scheme.value.range();
    let output_dtype = packed_storage_elem(scheme);

    let address_type = input
        .required_address_type(input_dtype.size())
        .max(scale.required_address_type(scale_dtype.size()))
        .max(output.required_address_type(output_dtype.size()));

    check_block_size_compat(scheme, num_quants); // 32 / 8 = 4

    let scales_layout = scales_layout(&output, &scale, 1, scheme);

    unsafe {
        quantize_symmetric_packed_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            address_type,
            vector_size,
            linear_view(input),
            // scale is computed based on input float dtype, but stored based on qparams precision
            scales_view(output.clone(), scale, 1, scheme),
            global.map(linear_view).into(),
            InputScalar::new(range_min, input_dtype),
            InputScalar::new(range_max, input_dtype),
            linear_view(output.clone()),
            scales_view(output, out_scale, 1, scheme),
            out_global.map(linear_view).into(),
            scales_layout,
            *scheme,
            [input_dtype, scale_dtype, output_dtype],
        )
    };

    Ok(())
}
