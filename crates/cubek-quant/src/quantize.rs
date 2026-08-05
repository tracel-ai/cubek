use cubecl::std::quant::round::round_up_to_param;
use cubecl::{
    calculate_cube_count_elemwise,
    features::TypeUsage,
    ir::ElemType,
    prelude::*,
    std::tensor::{
        View, ViewMut, into_contiguous,
        layout::linear::{LinearView, LinearViewMut, linear_view},
    },
    tensor_vector_size_parallel,
};

use crate::{
    global_scale::{quantize_symmetric_scaled, read_global, write_global},
    layout::{ScalesLayout, ScalesViewMut, scales_view},
    utils::{
        check_block_size_compat, check_global_bindings, check_param_supported, global_dtype,
        packed_storage_elem, scale_dtype,
    },
};
use crate::{
    layout::{ScalesView, scales_layout},
    scheme::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue},
};

#[cube]
fn quantize_symmetric<F: Float, N: Size, FG: Numeric, FS: CubePrimitive>(
    value: Vector<F, N>,
    block: FS,
    global: ComptimeOption<FG>,
    range_min: F,
    range_max: F,
) -> Vector<F, N> {
    clamp(
        Vector::round(quantize_symmetric_scaled::<F, FG, FS, N>(
            value, block, global,
        )),
        Vector::new(range_min),
        Vector::new(range_max),
    )
}

#[cube]
fn quantize_symmetric_q<F: Float, N: Size, FG: Numeric, FS: CubePrimitive, Q: Scalar>(
    value: Vector<F, N>,
    block: FS,
    global: ComptimeOption<FG>,
    range_min: F,
    range_max: F,
) -> Vector<Q, N> {
    Vector::cast_from(quantize_symmetric::<F, N, FG, FS>(
        value, block, global, range_min, range_max,
    ))
}

#[cube]
fn quantize_packed_value<F: Float, N: Size, FG: Numeric, FS: CubePrimitive, QS: Int>(
    value: Vector<F, N>,
    block: FS,
    global: ComptimeOption<FG>,
    range_min: F,
    range_max: F,
    #[comptime] scheme: QuantScheme,
) -> QS {
    let value = quantize_symmetric::<F, N, FG, FS>(value, block, global, range_min, range_max);
    pack_q::<F, N, QS>(value, scheme.value)
}

/// Pack a vector of quantized floating-point values into a single integer (the stored quantization type),
/// according to the specified quantization input type.
#[allow(clippy::explicit_counter_loop)]
#[cube]
fn pack_q<F: Float, N: Size, QS: Int>(value: Vector<F, N>, #[comptime] quant: QuantValue) -> QS {
    let size_quant = quant.size_bits();

    let size_store = QS::type_size_bits().comptime();
    let num_quants = size_store / size_quant;

    let mask = (1 << size_quant) - 1;
    let mut packed = QS::from_int(0);

    // Shift and combine into QS (using i32 for sign extension)
    #[unroll]
    for position in 0..num_quants {
        let offset = QS::cast_from(position * size_quant);
        let shifted = QS::cast_from(i32::cast_from(value.extract(position)) & mask) << offset;
        packed |= shifted;
    }

    packed
}

#[cube]
fn write_scale<F: Float, FS: CubePrimitive>(
    in_pos: usize,
    scale: View<F, usize>,
    mut out_scale: ViewMut<FS, usize>,
    scales_layout: ScalesLayout,
    #[comptime] param: QuantParam,
) -> FS {
    let scale = FS::cast_from(round_up_to_param::<F>(scale.read(in_pos), param));

    // Write the scale into the output buffer
    if scales_layout.is_block_start(in_pos) {
        out_scale.write(in_pos, scale);
    }

    scale
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn quantize_symmetric_native_kernel<F: Float, N: Size, FS: Numeric, FG: Numeric, Q: Numeric>(
    input: LinearView<'_, Vector<F, N>>,
    scale: ScalesView<'_, F>,
    global: ComptimeOption<LinearView<'_, FG>>,
    range_min: InputScalar,
    range_max: InputScalar,
    mut output: LinearViewMut<'_, Vector<Q, N>>,
    out_scale: ScalesViewMut<'_, FS>,
    out_global: ComptimeOption<LinearViewMut<'_, FG>>,
    scales_layout: ScalesLayout,
    #[comptime] param: QuantParam,
    #[define(F, FS, FG, Q)] _dtypes: [StorageType; 4],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let native_packing = Q::packing_factor();
    let in_pos = ABSOLUTE_POS * input.vector_size() * native_packing;
    let block = write_scale(in_pos, scale, out_scale, scales_layout, param);
    let global = read_global::<FG>(global);
    write_global::<FG>(global, out_global);

    output.write(
        ABSOLUTE_POS,
        quantize_symmetric_q::<F, N, FG, FS, Q>(
            input.read(ABSOLUTE_POS),
            block,
            global,
            range_min.get::<F>(),
            range_max.get::<F>(),
        ),
    );
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn quantize_symmetric_packed_kernel<F: Float, N: Size, FS: Numeric, FG: Numeric, QS: Int>(
    input: LinearView<'_, Vector<F, N>>,
    scale: ScalesView<'_, F>,
    global: ComptimeOption<LinearView<'_, FG>>,
    range_min: InputScalar,
    range_max: InputScalar,
    mut output: LinearViewMut<'_, QS>,
    out_scale: ScalesViewMut<'_, FS>,
    out_global: ComptimeOption<LinearViewMut<'_, FG>>,
    scales_layout: ScalesLayout,
    #[comptime] scheme: QuantScheme,
    #[define(F, FS, FG, QS)] _dtypes: [StorageType; 4],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let num_quants = scheme.num_quants();
    let packed_pos = ABSOLUTE_POS * num_quants;
    let block = write_scale(packed_pos, scale, out_scale, scales_layout, scheme.param);
    let global = read_global::<FG>(global);
    write_global::<FG>(global, out_global);

    if input.vector_size().comptime() == num_quants {
        output.write(
            ABSOLUTE_POS,
            quantize_packed_value::<F, N, FG, FS, QS>(
                input.read(ABSOLUTE_POS),
                block,
                global,
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
            values.insert(i, input.read(packed_pos + i).extract(0));
        }
        output.write(
            ABSOLUTE_POS,
            quantize_packed_value::<F, NQ, FG, FS, QS>(
                values,
                block,
                global,
                range_min.get::<F>(),
                range_max.get::<F>(),
                scheme,
            ),
        );
    }
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    scale: TensorBinding<R>,
    global: Option<TensorBinding<R>>,
    out_scale: TensorBinding<R>,
    out_global: Option<TensorBinding<R>>,
    scheme: &QuantScheme,
    input_elem: ElemType,
) -> Result<(), LaunchError> {
    check_global_bindings(scheme, global.is_some(), "global");
    check_global_bindings(scheme, out_global.is_some(), "out_global");
    check_param_supported(scheme);

    match scheme {
        QuantScheme {
            store: QuantStore::PackedU32(_),
            ..
        } => quantize_packed(
            client, input, scheme, scale, global, out_scale, out_global, output, input_elem,
        ),
        QuantScheme {
            value: QuantValue::Q8F | QuantValue::Q8S | QuantValue::E4M3 | QuantValue::E5M2,
            store: QuantStore::Native,
            ..
        }
        | QuantScheme {
            value: QuantValue::E2M1,
            store: QuantStore::PackedNative(_),
            ..
        } => {
            if !i8::supported_uses(client).contains(TypeUsage::Conversion) {
                panic!(
                    "{:?} is not supported for native quantization",
                    scheme.value
                );
            }

            quantize_native(
                client, input, scheme, scale, global, out_scale, out_global, output, input_elem,
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
) -> Result<(), LaunchError> {
    let scale_dtype = scale_dtype(scheme);
    let global_dtype = global_dtype(scheme);

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
            level: QuantLevel::Tensor | QuantLevel::Block(_) | QuantLevel::BlockTensor { .. },
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
                    scheme.param,
                    [
                        input_dtype.into(),
                        scale_dtype.into(),
                        global_dtype.into(),
                        output_dtype.into(),
                    ],
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
) -> Result<(), LaunchError> {
    let scale_dtype = scale_dtype(scheme);
    let global_dtype = global_dtype(scheme);

    let num_elems: usize = input.shape.iter().product();

    // Determine if we can use vectorized packing
    let mut can_vectorize = match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_) | QuantLevel::BlockTensor { .. },
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
        into_contiguous(client, input, input_dtype.into()).binding()
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
            [
                input_dtype.into(),
                scale_dtype.into(),
                global_dtype.into(),
                output_dtype.into(),
            ],
        )
    };

    Ok(())
}
