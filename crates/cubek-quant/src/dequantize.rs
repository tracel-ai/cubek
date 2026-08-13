#![allow(missing_docs)] // pub cube modules

use cubecl::tensor_vector_size_parallel;
use cubecl::{calculate_cube_count_elemwise, ir::ElemType};
use cubecl::{prelude::*, std::tensor::layout::linear::LinearViewMut};

use crate::{
    layout::{ScalesView, scales_view},
    scale::Scales,
    scheme::{BlockSize, QuantMode, QuantScheme, QuantStore, QuantValue, ScaleDtype},
    utils::packed_storage_elem,
};
use cubecl::std::quant::check_scale_bindings;
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
        let floats = unpack_q::<F, NF, QS>(values.extract(i), scheme.value, scheme.store);
        let scale = scales.read((position * vector_size_values) + i * num_quants);
        tmp[i] = scale.dequantize_symmetric::<F, NF>(floats);
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

/// Convert the tensor back to a higher precision data type.
///
/// `scales` holds one binding per scale level, innermost first. A packed store's `input` is
/// the stored binding, its packed axis counted in `u32` words.
///
/// # Errors
///
/// Propagates the kernel's [`LaunchError`]. A scheme neither path can serve panics on the
/// caller's thread instead, so the plan fails loudly.
#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    scales: &[TensorBinding<R>],
    scheme: &QuantScheme,
    output_dtype: ElemType,
) -> Result<(), LaunchError> {
    check_scale_bindings(scheme, scales.len());

    let widest_line = client
        .io_optimized_vector_sizes(size_of::<u32>().max(output_dtype.size()))
        .next()
        .unwrap_or(1);
    match dequant_path(scheme, widest_line, &input.strides, &output.strides) {
        DequantPath::Tile => launch_tile(client, input, output, scales, scheme, output_dtype),
        DequantPath::Legacy => launch_legacy(client, input, output, scales, scheme, output_dtype),
    }
}

/// Which implementation serves a launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DequantPath {
    /// The tile-engine kernel.
    Tile,
    /// The elementwise kernels, kept for what the tile path cannot express.
    Legacy,
}

/// Route a launch from host data alone, so the routing matrix is unit-tested.
///
/// The tile path serves f32-scale symmetric schemes over contiguous innermost dims, with
/// values stored natively or as innermost-packed `u32` words a device line can cover whole
/// (`widest_line` is the widest line over the storage word); a FULL dim inside a block is
/// refused because it would put a zero edge into the scale windowing. Everything else keeps
/// the legacy kernels.
fn dequant_path(
    scheme: &QuantScheme,
    widest_line: usize,
    input_strides: &[usize],
    output_strides: &[usize],
) -> DequantPath {
    let store_served = match scheme.store {
        QuantStore::Native => matches!(
            scheme.value,
            QuantValue::Q8F | QuantValue::Q8S | QuantValue::E4M3 | QuantValue::E5M2
        ),
        QuantStore::PackedU32(0) => scheme.num_quants() <= widest_line,
        _ => false,
    };
    let block_served = scheme
        .block_size()
        .is_none_or(|block| !block.as_slice().contains(&BlockSize::FULL));
    let contiguous = input_strides.last() == Some(&1) && output_strides.last() == Some(&1);

    if scheme.scale_dtype() == ScaleDtype::F32
        && scheme.mode == QuantMode::Symmetric
        && store_served
        && block_served
        && contiguous
    {
        DequantPath::Tile
    } else {
        DequantPath::Legacy
    }
}

#[allow(clippy::result_large_err)]
fn launch_tile<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    scales: &[TensorBinding<R>],
    scheme: &QuantScheme,
    output_dtype: ElemType,
) -> Result<(), LaunchError> {
    let values = match scheme.store {
        QuantStore::PackedU32(_) => {
            packed_binding_in_values(input, &output.shape, scheme.num_quants())
        }
        _ => input,
    };
    crate::dequantize_tiled::launch_ref(client, values, output, scales, scheme, output_dtype)
}

/// Re-declare a packed binding from storage to values: the caller counts the packed axis in
/// `u32` words, while the tile path counts the values they hold, so the innermost extent
/// widens by `num_quants` and every coarser stride re-expresses in values (the innermost
/// stays 1; the buffer keeps its storage width).
fn packed_binding_in_values<R: Runtime>(
    stored: TensorBinding<R>,
    output_shape: &[usize],
    num_quants: usize,
) -> TensorBinding<R> {
    let mut values = stored;
    let rank = values.shape.len();
    values.shape[rank - 1] *= num_quants;
    assert_eq!(
        &values.shape[..],
        output_shape,
        "dequantize: the stored shape widened by the packing factor must match the output"
    );
    for stride in &mut values.strides[..rank - 1] {
        *stride *= num_quants;
    }
    values
}

/// The elementwise kernels; [`launch_ref`] falls back here when the tile path cannot serve a
/// launch.
#[allow(clippy::result_large_err)]
pub(crate) fn launch_legacy<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    scales: &[TensorBinding<R>],
    scheme: &QuantScheme,
    output_dtype: ElemType,
) -> Result<(), LaunchError> {
    let scale_dtype: ElemType = ElemType::from_scale_dtype(scheme.scale_dtype());
    let (scale, global) = (scales[0].clone(), scales.get(1).cloned());

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
        }
        | QuantScheme {
            value: QuantValue::E2M1,
            store: QuantStore::PackedNative(_),
            ..
        } => {
            crate::utils::check_i8_supported(client, scheme);

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
fn dequantize_packed<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    scheme: QuantScheme,
    scale: TensorBinding<R>,
    global: Option<TensorBinding<R>>,
    output: TensorBinding<R>,
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
fn dequantize_native<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    scheme: QuantScheme,
    scale: TensorBinding<R>,
    global: Option<TensorBinding<R>>,
    output: TensorBinding<R>,
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

#[cfg(test)]
mod tests {
    use super::*;

    const CONTIGUOUS: &[usize] = &[64, 1];
    const WIDEST_LINE: usize = 4;

    fn path(scheme: QuantScheme) -> DequantPath {
        dequant_path(&scheme, WIDEST_LINE, CONTIGUOUS, CONTIGUOUS)
    }

    fn native_q8s() -> QuantScheme {
        QuantScheme::default()
            .with_store(QuantStore::Native)
            .with_value(QuantValue::Q8S)
    }

    #[test]
    fn native_per_tensor_takes_the_tile_path() {
        assert_eq!(path(native_q8s()), DequantPath::Tile);
    }

    #[test]
    fn native_block_takes_the_tile_path() {
        let scheme = native_q8s().per_block([32], ScaleDtype::F32);
        assert_eq!(path(scheme), DequantPath::Tile);
    }

    #[test]
    fn two_level_takes_the_tile_path() {
        let scheme = native_q8s()
            .per_block([16], ScaleDtype::F32)
            .per_tensor(ScaleDtype::F32);
        assert_eq!(path(scheme), DequantPath::Tile);
    }

    #[test]
    fn packed_within_the_device_line_takes_the_tile_path() {
        let scheme = QuantScheme::default().with_value(QuantValue::Q8S);
        assert_eq!(scheme.num_quants(), WIDEST_LINE);
        assert_eq!(path(scheme), DequantPath::Tile);
    }

    /// A served line covers whole `u32` words or nothing, so a packing factor past the
    /// device's widest line has no tile plan and must keep the legacy kernel.
    #[test]
    fn packed_past_the_device_line_falls_back() {
        let scheme = QuantScheme::default().with_value(QuantValue::Q4S);
        assert_eq!(path(scheme), DequantPath::Legacy);
        assert_eq!(
            dequant_path(&scheme, 8, CONTIGUOUS, CONTIGUOUS),
            DequantPath::Tile
        );
    }

    #[test]
    fn non_f32_scales_fall_back() {
        let scheme = native_q8s().per_tensor(ScaleDtype::F16);
        assert_eq!(path(scheme), DequantPath::Legacy);
    }

    #[test]
    fn native_sub_byte_values_fall_back() {
        let scheme = QuantScheme::default()
            .with_store(QuantStore::Native)
            .with_value(QuantValue::Q4S);
        assert_eq!(path(scheme), DequantPath::Legacy);
    }

    #[test]
    fn global_dim_packing_falls_back() {
        let scheme = QuantScheme::default()
            .with_store(QuantStore::PackedU32(1))
            .with_value(QuantValue::Q8S);
        assert_eq!(path(scheme), DequantPath::Legacy);
    }

    #[test]
    fn packed_native_falls_back() {
        let scheme = QuantScheme::default()
            .with_store(QuantStore::PackedNative(0))
            .with_value(QuantValue::E2M1);
        assert_eq!(path(scheme), DequantPath::Legacy);
    }

    #[test]
    fn a_full_dim_inside_a_block_falls_back() {
        let scheme = native_q8s().per_block([BlockSize::FULL, 32], ScaleDtype::F32);
        assert_eq!(path(scheme), DequantPath::Legacy);
    }

    #[test]
    fn a_strided_innermost_dim_falls_back() {
        let strided: &[usize] = &[128, 2];
        let scheme = native_q8s();
        assert_eq!(
            dequant_path(&scheme, WIDEST_LINE, strided, CONTIGUOUS),
            DequantPath::Legacy
        );
        assert_eq!(
            dequant_path(&scheme, WIDEST_LINE, CONTIGUOUS, strided),
            DequantPath::Legacy
        );
    }
}
