use cubecl::{
    features::TypeUsage,
    ir::ElemType,
    prelude::*,
    quant::scheme::{QuantLevel, QuantScheme, QuantStore, QuantValue},
};
use cubek_tile::{Axis, ByAxis, Distribution, Partitioner, Space, Storage, TileArg, TileArgLaunch};

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
    check_i8_supported(client, scheme);

    let input_space = sequential_space(&[(M, input.shape[0]), (N, input.shape[1])]);
    let input_storage = Storage::of(input.shape.len(), input_space.rank());
    let input_tilearg =
        TileArgLaunch::new(input.into_tensor_arg(), input_space.clone(), input_storage);

    // per-tensor scale: rank-1 [1] tensor, no reshape needed
    let scale_space = sequential_space(&[(M, 1usize)]);
    let scale_storage = Storage::of(scales.shape.len(), scale_space.rank());
    let scale_tilearg = TileArgLaunch::new(scales.into_tensor_arg(), scale_space, scale_storage);

    let output_space = sequential_space(&[(M, output.shape[0]), (N, output.shape[1])]);
    let output_storage = Storage::of(output.shape.len(), output_space.rank());
    let output_tilearg = TileArgLaunch::new(output.into_tensor_arg(), output_space, output_storage);

    let cube_count = input_space.cube_count();
    let cube_dim = input_space.cube_dim(client);

    let input_dtype = ElemType::from_quant_value(scheme.value).into();
    let scale_dtype = ElemType::from_quant_param(scheme.param).into();

    dequantize::launch(
        client,
        cube_count,
        cube_dim,
        input_tilearg,
        scale_tilearg,
        output_tilearg,
        input_dtype,
        scale_dtype,
        output_dtype,
        1usize,
        1usize,
        1usize,
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
/// input: the quantized input tensor
/// scales: the scale grid
/// output: the dequantized output tensor
pub fn dequantize<I: Numeric, S: Numeric, O: Numeric, IN: Size, SN: Size, ON: Size>(
    input: &TileArg<'_, I, IN>,
    scales: &TileArg<'_, S, SN>,
    output: &TileArg<'_, O, ON>,
    #[define(I)] _input_dtype: StorageType,
    #[define(S)] _scale_dtype: StorageType,
    #[define(O)] _output_dtype: StorageType,
    #[define(IN)] _input_size: usize,
    #[define(SN)] _scale_size: usize,
    #[define(ON)] _output_size: usize,
) {
    let input = input.tile();
    let scales = scales.tile();
    let mut output = output.tile();
    output.dequantize(&input, &scales);
}
