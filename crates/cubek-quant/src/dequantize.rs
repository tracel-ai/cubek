#![allow(missing_docs)] // pub cube modules

use cubecl::{calculate_cube_count_elemwise, ir::ElemType};
use cubecl::{features::TypeUsage, tensor_vector_size_parallel};
use cubecl::{prelude::*, std::tensor::layout::linear::LinearViewMut};

use crate::{
    layout::{ScalesView, scales_view},
    per_tensor::{apply_global, read_global},
    scheme::{QuantLevel, QuantMode, QuantScheme, QuantStore, QuantValue},
    utils::{check_global_bindings, global_dtype, packed_storage_elem},
};
use cubecl::std::tensor::{
    View,
    layout::linear::{LinearView, linear_view},
};

/// Dequantize a vector of values into floating-point values using the provided scale.
#[cube]
pub fn dequantize_symmetric<F: Float, FS: CubePrimitive, N: Size>(
    value: Vector<F, N>,
    scale: FS,
) -> Vector<F, N> {
    // x = scale * x_q, formed in f32 so that a scale which is subnormal in `F` still scales the
    // values it is representable against, instead of flushing the whole block to zero.
    Vector::cast_from(Vector::<f32, N>::cast_from(value) * Vector::<f32, N>::cast_from(scale))
}

/// Dequantize the value at a specified position using the provided quantization scheme.
///
/// Returns a vector of floating-point values. The number of values in the vector depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_symmetric_packed_values<
    F: Float,
    NF: Size,
    FS: CubePrimitive,
    FG: Numeric,
    QI: Int,
    NQ: Size,
>(
    position: usize,
    values: &View<Vector<QI, NQ>, usize>,
    scales: &View<FS, usize>,
    global: ComptimeOption<LinearView<'_, FG>>,
    #[comptime] scheme: QuantScheme,
) -> Array<Vector<F, NF>> {
    dequantize_symmetric_packed_value_at::<F, NF, FS, FG, QI, NQ>(
        position,
        values.read(position),
        scales,
        global,
        scheme,
    )
}

/// Dequantize a single value using the scale at the specified position.
///
/// Returns a vector of floating-point values. The number of values in the vector depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_symmetric_packed_value_at<
    F: Float,
    NF: Size,
    FS: CubePrimitive,
    FG: Numeric,
    QI: Int,
    NQ: Size,
>(
    position: usize,
    values: Vector<QI, NQ>,
    scales: &View<FS, usize>,
    global: ComptimeOption<LinearView<'_, FG>>,
    #[comptime] scheme: QuantScheme,
) -> Array<Vector<F, NF>> {
    dequantize_symmetric_packed_value::<F, NF, FS, FG, QI, NQ>(
        values, scales, global, position, scheme,
    )
}

/// Dequantize a single packed value using the scale provided.
///
/// Returns a vector of floating-point values. The number of values in the vector depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_symmetric_packed_value<
    F: Float,
    NF: Size,
    FS: CubePrimitive,
    FG: Numeric,
    QS: Int,
    NQ: Size,
>(
    values: Vector<QS, NQ>,
    scales: &View<FS, usize>,
    global: ComptimeOption<LinearView<'_, FG>>,
    position: usize,
    #[comptime] scheme: QuantScheme,
) -> Array<Vector<F, NF>> {
    let vector_size_values = values.vector_size();
    let num_quants = scheme.num_quants();
    let mut tmp = Array::new(vector_size_values);

    let global = read_global::<FG>(global);

    #[unroll]
    for i in 0..vector_size_values {
        let floats = unpack_q::<F, NF, QS>(values.extract(i), scheme.value, scheme.store);
        let block = scales.read((position * vector_size_values) + i * num_quants);
        let scale = apply_global::<FG, FS>(block, global);
        let values = dequantize_symmetric::<F, f32, NF>(floats, scale);
        tmp[i] = values;
    }

    tmp
}

/// Unpack a quantized integer into a vector of floating-point values, according to the specified quantization input type.
///
/// This handles types where multiple quantized values are packed into a single integer (the stored quantization type).
#[allow(clippy::explicit_counter_loop)]
#[cube]
fn unpack_q<F: Float, NF: Size, QS: Int>(
    value: QS,
    #[comptime] quant: QuantValue,
    #[comptime] store: QuantStore,
) -> Vector<F, NF> {
    let size_quant = quant.size_bits();
    let size_store = store.size_bits(&quant);
    let num_quant = size_store / size_quant;

    let mut output = Vector::empty();

    let mask = QS::from_int((1 << size_quant) - 1);
    let sign_bit = QS::from_int(1 << (size_quant - 1));
    let two_pow_n = 1 << size_quant;

    #[unroll]
    for position in 0..num_quant {
        let offset = QS::cast_from(position * size_quant);
        let raw = (value >> offset) & mask;

        // Branchless two's complement conversion
        // If raw >= 2^(n-1), then result = raw - 2^n
        let raw_i32 = i32::cast_from(raw);
        let is_negative = i32::cast_from(raw >= sign_bit); // 1 if negative, 0 if positive
        let signed_value = raw_i32 - (is_negative * two_pow_n);

        output.insert(position, F::cast_from(signed_value));
    }

    output
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn dequantize_symmetric_packed_kernel<
    F: Float,
    NF: Size,
    FS: Numeric,
    FG: Numeric,
    QS: Int,
    NQ: Size,
>(
    input: LinearView<'_, Vector<QS, NQ>>,
    scales: ScalesView<'_, FS>,
    global: ComptimeOption<LinearView<'_, FG>>,
    mut output: LinearViewMut<'_, Vector<F, NF>>,
    #[comptime] scheme: QuantScheme,
    #[define(F, FS, FG, QS)] _dtypes: [StorageType; 4],
) {
    if !input.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let vector_size_in = input.vector_size();
    let vector_size_out = output.vector_size();

    comptime! {
        assert_eq!(vector_size_out, scheme.num_quants());
    }

    let values = input.read(ABSOLUTE_POS);
    let packed_pos = ABSOLUTE_POS * scheme.num_quants();

    let out = dequantize_symmetric_packed_value::<F, NF, FS, FG, QS, NQ>(
        values, &scales, global, packed_pos, scheme,
    );

    #[unroll]
    for i in 0..vector_size_in {
        output.write(ABSOLUTE_POS * vector_size_in + i, out[i]);
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn dequantize_symmetric_native_kernel<F: Float, N: Size, FS: Numeric, FG: Numeric, Q: Numeric>(
    input: LinearView<'_, Vector<Q, N>>,
    scale: ScalesView<'_, FS>,
    global: ComptimeOption<LinearView<'_, FG>>,
    mut output: LinearViewMut<'_, Vector<F, N>>,
    #[define(F, FS, FG, Q)] _dtypes: [StorageType; 4],
) {
    if !input.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let native_packing = Q::packing_factor();
    // Absolute pos represents the logical block (scale) used to dequantize, not layout
    let block = scale.read(ABSOLUTE_POS * input.vector_size() * native_packing);
    let scale = apply_global::<FG, FS>(block, read_global::<FG>(global));

    output.write(
        ABSOLUTE_POS,
        dequantize_symmetric::<F, f32, N>(Vector::cast_from(input.read(ABSOLUTE_POS)), scale),
    );
}

#[allow(clippy::result_large_err)]
/// Convert the tensor back to a higher precision data type.
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    scale: TensorBinding<R>,
    global: Option<TensorBinding<R>>,
    scheme: &QuantScheme,
    output_dtype: StorageType,
) -> Result<(), LaunchError> {
    let scale_dtype: StorageType = ElemType::from_quant_param(scheme.param).into();
    let global_dtype: StorageType = global_dtype(scheme).into();

    check_global_bindings(scheme, global.is_some());

    match scheme {
        QuantScheme {
            store: QuantStore::PackedU32(_),
            ..
        } => dequantize_packed(
            client,
            input,
            *scheme,
            scale,
            global,
            output,
            output_dtype,
            scale_dtype,
            global_dtype,
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

            dequantize_native(
                client,
                input,
                *scheme,
                scale,
                global,
                output,
                output_dtype,
                scale_dtype,
                global_dtype,
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
fn dequantize_packed<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    scheme: QuantScheme,
    scale: TensorBinding<R>,
    global: Option<TensorBinding<R>>,
    output: TensorBinding<R>,
    output_dtype: StorageType,
    scale_dtype: StorageType,
    global_dtype: StorageType,
) -> Result<(), LaunchError> {
    let num_elems_input: usize = input.shape.iter().product();
    let input_dtype = packed_storage_elem(&scheme);

    let mut vector_size_in = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(input_dtype.size()),
        &input.shape,
        &input.strides,
        input.shape.len() - 1,
    );
    let num_quants = scheme.num_quants();
    let vector_size_out = num_quants;
    let rank = output.shape.len();

    if !output.shape[rank - 1].is_multiple_of(vector_size_out) {
        vector_size_in = 1;
    }

    let num_elems = num_elems_input / vector_size_in as usize;
    let cube_dim = CubeDim::new(client, num_elems);
    let cube_count = calculate_cube_count_elemwise(client, num_elems, cube_dim);
    let address_type = input
        .required_address_type(input_dtype.size())
        .max(scale.required_address_type(scale_dtype.size()))
        .max(output.required_address_type(output_dtype.size()));

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_) | QuantLevel::BlockTensor { .. },
            store: QuantStore::PackedU32(_),
            mode: QuantMode::Symmetric,
            ..
        } => unsafe {
            dequantize_symmetric_packed_kernel::launch_unchecked(
                client,
                cube_count,
                cube_dim,
                address_type,
                vector_size_out,
                vector_size_in,
                linear_view(input.clone()),
                scales_view(input, scale, 1, &scheme),
                global.map(linear_view).into(),
                linear_view(output),
                scheme,
                [output_dtype, scale_dtype, global_dtype, input_dtype.into()],
            )
        },
        QuantScheme { .. } => panic!("Unsupported quantization scheme {scheme:?}"),
    };

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dequantize_native<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    scheme: QuantScheme,
    scale: TensorBinding<R>,
    global: Option<TensorBinding<R>>,
    output: TensorBinding<R>,
    output_dtype: StorageType,
    scale_dtype: StorageType,
    global_dtype: StorageType,
) -> Result<(), LaunchError> {
    let num_elems: usize = input.shape.iter().product();
    let input_dtype = ElemType::from_quant_value(scheme.value);

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

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_) | QuantLevel::BlockTensor { .. },
            mode: QuantMode::Symmetric,
            store: QuantStore::Native,
            ..
        } => {
            let address_type = input
                .required_address_type(input_dtype.size())
                .max(scale.required_address_type(scale_dtype.size()))
                .max(output.required_address_type(output_dtype.size()));

            unsafe {
                dequantize_symmetric_native_kernel::launch_unchecked(
                    client,
                    cube_count,
                    cube_dim,
                    address_type,
                    vector_size,
                    linear_view(input.clone()),
                    scales_view(input, scale, 1, &scheme),
                    global.map(linear_view).into(),
                    linear_view(output),
                    [output_dtype, scale_dtype, global_dtype, input_dtype.into()],
                )
            }
        }
        QuantScheme { .. } => panic!("Unsupported quantization scheme {scheme:?}"),
    };

    Ok(())
}
