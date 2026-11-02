#![allow(missing_docs)] // pub cube modules

use cubecl::{calculate_cube_count_elemwise, ir::ElemType};
use cubecl::{features::TypeUsage, tensor_vector_size_parallel};
use cubecl::{prelude::*, std::tensor::layout::linear::LinearViewMut};

use crate::{
    layout::{ScalesView, scales_view},
    scale::{Scales, split_levels},
    scheme::{QuantMode, QuantScheme, QuantStore, QuantValue},
    utils::packed_storage_elem,
};
use cubecl::std::quant::{check_scale_bindings, unpack_cast_u32};
use cubecl::std::tensor::{
    View,
    layout::linear::{LinearView, linear_view},
};

/// Dequantize the value at a specified position using the provided quantization scheme.
///
/// Returns a vector of floating-point values. The number of values in the vector depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_symmetric_packed_values<
    F: Float,
    NF: Size,
    FS: CubePrimitive,
    QI: Int,
    NQ: Size,
>(
    position: usize,
    values: &View<Vector<QI, NQ>, usize>,
    scales: &Scales<'_, FS>,
    #[comptime] scheme: QuantScheme,
) -> Array<Vector<F, NF>> {
    dequantize_symmetric_packed_value_at::<F, NF, FS, QI, NQ>(
        position,
        values.read(position),
        scales,
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
    QI: Int,
    NQ: Size,
>(
    position: usize,
    values: Vector<QI, NQ>,
    scales: &Scales<'_, FS>,
    #[comptime] scheme: QuantScheme,
) -> Array<Vector<F, NF>> {
    dequantize_symmetric_packed_value::<F, NF, FS, QI, NQ>(values, scales, position, scheme)
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
    QS: Int,
    NQ: Size,
>(
    values: Vector<QS, NQ>,
    scales: &Scales<'_, FS>,
    position: usize,
    #[comptime] scheme: QuantScheme,
) -> Array<Vector<F, NF>> {
    let vector_size_values = values.vector_size();
    let num_quants = scheme.num_quants();
    let mut tmp = Array::new(vector_size_values);

    #[unroll]
    for i in 0..vector_size_values {
        let floats = unpack_cast_u32::<F, Const<1>, NF>(
            Vector::<u32, Const<1>>::new(u32::cast_from(values.extract(i))),
            ComptimeOption::new_None(),
            scheme,
        );
        let scale = scales.read((position * vector_size_values) + i * num_quants);
        tmp[i] = scale.dequantize_symmetric::<F, NF>(floats);
    }

    tmp
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn dequantize_symmetric_packed_kernel<F: Float, NF: Size, FS: Numeric, QS: Int, NQ: Size>(
    input: LinearView<'_, Vector<QS, NQ>>,
    scales: ScalesView<'_, FS>,
    global: ComptimeOption<LinearView<'_, f32>>,
    mut output: LinearViewMut<'_, Vector<F, NF>>,
    #[comptime] scheme: QuantScheme,
    #[define(F, FS, QS)] _dtypes: [ElemType; 3],
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

    let scales = Scales::<FS>::new(&scales, global);
    let out =
        dequantize_symmetric_packed_value::<F, NF, FS, QS, NQ>(values, &scales, packed_pos, scheme);

    #[unroll]
    for i in 0..vector_size_in {
        output.write(ABSOLUTE_POS * vector_size_in + i, out[i]);
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn dequantize_symmetric_native_kernel<F: Float, N: Size, FS: Numeric, Q: Numeric>(
    input: LinearView<'_, Vector<Q, N>>,
    scale: ScalesView<'_, FS>,
    global: ComptimeOption<LinearView<'_, f32>>,
    mut output: LinearViewMut<'_, Vector<F, N>>,
    #[define(F, FS, Q)] _dtypes: [ElemType; 3],
) {
    if !input.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let native_packing = Q::packing_factor();
    let scales = Scales::<FS>::new(&scale, global);
    // Absolute pos is the logical position the scale is looked up by, not a layout index
    let scale = scales.read(ABSOLUTE_POS * input.vector_size() * native_packing);

    output.write(
        ABSOLUTE_POS,
        scale.dequantize_symmetric::<F, N>(Vector::cast_from(input.read(ABSOLUTE_POS))),
    );
}

#[allow(clippy::result_large_err)]
/// Convert the tensor back to a higher precision data type.
pub fn launch_ref(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    scales: &[TensorBinding],
    scheme: &QuantScheme,
    output_dtype: ElemType,
) -> Result<(), LaunchError> {
    check_scale_bindings(scheme, scales.len());

    let scale_dtype: ElemType = ElemType::from_scale_dtype(scheme.scale_dtype());
    let (scale, global) = split_levels(scales);

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

            dequantize_native(
                client,
                input,
                *scheme,
                scale,
                global,
                output,
                output_dtype,
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
fn dequantize_packed(
    client: &Client,
    input: TensorBinding,
    scheme: QuantScheme,
    scale: TensorBinding,
    global: Option<TensorBinding>,
    output: TensorBinding,
    output_dtype: ElemType,
    scale_dtype: ElemType,
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
                [output_dtype, scale_dtype, input_dtype],
            )
        },
        QuantScheme { .. } => panic!("Unsupported quantization scheme {scheme:?}"),
    };

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dequantize_native(
    client: &Client,
    input: TensorBinding,
    scheme: QuantScheme,
    scale: TensorBinding,
    global: Option<TensorBinding>,
    output: TensorBinding,
    output_dtype: ElemType,
    scale_dtype: ElemType,
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
                    [output_dtype, scale_dtype, input_dtype],
                )
            }
        }
        QuantScheme { .. } => panic!("Unsupported quantization scheme {scheme:?}"),
    };

    Ok(())
}
