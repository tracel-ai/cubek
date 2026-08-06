use cubecl::{
    features::TypeUsage,
    ir::ElemType,
    prelude::*,
    quant::scheme::{QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue},
};
use cubek_tile::{
    Axis, ByAxis, Distribution, Partitioner, QuantTileArg, Space, StridedOperand, TileArg, Until,
};

// Input axes
const M: Axis = Axis(0);
const N: Axis = Axis(1);

/// Convert the tensor back to a higher precision data type.
/// Uses the tile-based implementation for dequantization.
/// Very WIP and naive implementation for now.
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    scales: TensorBinding<R>,
    scheme: &QuantScheme,
    output_dtype: StorageType,
) -> Result<(), LaunchError> {
    assert!(
        scheme.store == QuantStore::Native,
        "only native quantization is supported for now."
    );
    assert!(
        scheme.level == QuantLevel::Tensor,
        "only per tensor quantization is supported for now."
    );
    assert!(
        scheme.param == QuantParam::F32,
        "only f32 scales are supported for now."
    );
    check_i8_supported(client, scheme);

    // One space for the whole kernel; both operands span all of it. Geometry reads the
    // concrete extents; the kernel gets the dynamic form, so m and n resolve in-kernel
    // from the tensor's own shape and never fork the compiled kernel.
    let space = sequential_space(&[(M, input.shape[0]), (N, input.shape[1])]);
    let cube_count = space.cube_count();
    let cube_dim = space.cube_dim(client);
    let input_dtype = ElemType::from_quant_value(scheme.value).into();
    // Both operands through the source builder, which derives the storage from the binding's own
    // dims and validates the scheme against this space. One tile covers each axis, so nothing
    // overhangs and the checks stay off.
    let input_op = StridedOperand::source(input)
        .space(&space)
        .subspace(&[M, N])
        .checked(false)
        // Nothing stages this operand, so its read is what decodes it.
        .quantized(scales.into_tensor_arg(), *scheme, Until::Read)
        .build();
    let output_op = StridedOperand::source(output)
        .space(&space)
        .subspace(&[M, N])
        .checked(false)
        .build();
    dequantize::launch(
        client,
        cube_count,
        cube_dim,
        input_op.arg(),
        output_op.arg(),
        space.all_dynamic(),
        input_dtype,
        output_dtype,
    );

    Ok(())
}

/// A row-major space whose every axis is `Sequential`: a single cube walks all the tiles.
/// Each axis is one tile covering its full extent (one tile total).
fn sequential_space(extents: &[(Axis, usize)]) -> Space {
    let dists: Vec<(Axis, Distribution)> = extents
        .iter()
        .map(|&(a, _)| (a, Distribution::Sequential))
        .collect();
    let partitioner = Partitioner::row_major(ByAxis::new(extents), ByAxis::new(&dists)).direct();
    Space::new(extents).with_partitioner(partitioner)
}

fn check_i8_supported<R: Runtime>(client: &ComputeClient<R>, scheme: &QuantScheme) {
    match scheme {
        QuantScheme {
            value: QuantValue::Q8F | QuantValue::Q8S | QuantValue::E4M3 | QuantValue::E5M2,
            store: QuantStore::Native,
            ..
        }
        | QuantScheme {
            value: QuantValue::E2M1,
            store: QuantStore::PackedNative(_),
            ..
        } if !i8::supported_uses(client).contains(TypeUsage::Conversion) => {
            panic!(
                "{:?} is not supported for native quantization",
                scheme.value
            );
        }
        _ => {}
    }
}

#[cube(launch)]
/// The input tile serves `O` and dequantizes on read, so the body is a plain copy; `I` (the
/// storage element) only names the binding's element, the scheme recovers the served value.
/// Scales ride as an ordinary second tensor.
pub fn dequantize<I: Numeric, O: Numeric>(
    input: &QuantTileArg<'_, I, Const<1>>,
    output: &TileArg<'_, O, Const<1>>,
    #[comptime] space: Space,
    #[define(I)] _input_dtype: StorageType,
    #[define(O)] _output_dtype: StorageType,
) {
    let input = input.tile::<O>(comptime!(space.clone()));
    let mut output = output.tile(space);
    output.copy_from(&input);
}
