use cubecl::{
    ir::ElemType,
    prelude::*,
    quant::scheme::{QuantScheme, QuantStore, ScaleDtype},
};
use cubek_tile::{
    Axis, Launcher, Partitioning, Projection, Space, TileArg, TileArgLaunch, TileSpec, kind::Field,
    launch::Grid, layout::PhysicalAxisMap,
};

// Input axes
const M: Axis = Axis(0);
const N: Axis = Axis(1);

/// Convert the tensor back to a higher precision data type.
/// Uses the tile-based implementation for dequantization.
/// Very WIP and naive implementation for now.
///
/// The decode is the kernel's own statement: the values are fields of the scheme's element read
/// out of words, under their scale, and `output.dequantize_from(&values)` multiplies each by the
/// scale where it is written. Nothing decodes behind a read.
pub fn launch_ref(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    scales: TensorBinding,
    scheme: &QuantScheme,
    output_dtype: ElemType,
) -> Result<(), LaunchError> {
    assert!(
        scheme.store == QuantStore::Native,
        "only native quantization is supported for now."
    );
    assert!(
        scheme.num_levels() == 1 && scheme.block_size().is_none(),
        "only per tensor quantization is supported for now."
    );
    assert!(
        scheme.scale_dtype() == ScaleDtype::F32,
        "only f32 scales are supported for now."
    );
    assert!(
        input.shape.len() == 2 && output.shape.len() == 2,
        "dequantize_tiled: both operands must be plain 2-D tensors, got {:?} and {:?}",
        input.shape,
        output.shape
    );

    // A native store keeps one value a byte, so the buffer is read as words of fields: each
    // `u32` holds `per_word` of them, decoded by the scheme's own field. The binding still counts
    // values, as a packed operand's does.
    let field = Field::Quant(scheme.value);
    let per_word = field.per_word();
    let (rows, cols) = (input.shape[0], input.shape[1]);
    assert!(
        field.size_bits() == 8,
        "dequantize_tiled: a native store holds one value a byte, and {:?} is {} bits",
        scheme.value,
        field.size_bits()
    );
    assert!(
        input.strides[1] == 1 && input.strides[0] == cols && cols.is_multiple_of(per_word),
        "dequantize_tiled: the values are read as whole words, so each row must be contiguous \
         and a whole number of words ({per_word} values), got shape {:?} and strides {:?}",
        input.shape,
        input.strides
    );
    let max = client.properties().hardware.max_vector_size;
    assert!(
        per_word <= max,
        "dequantize_tiled: a word of {per_word} values is read as one line, and this device's \
         lines hold {max}"
    );

    // One tile covering every axis, walked by a single cube: no level cuts it, so there is
    // nothing to list and the grid is one cube.
    let space = Space::new(&[(M, rows), (N, cols)]);
    let plane_size = client.properties().hardware.plane_size_max;
    let (cube_count, cube_dim) = (CubeCount::Static(1, 1, 1), CubeDim::new_2d(plane_size, 1));
    let launch = Launcher::new(
        client,
        // Static: the decoding copy reads through the values' own axes, which it needs the
        // extents of at expansion, so the kernel is compiled per shape.
        Partitioning::new(space.clone(), vec![]),
        &space,
        Grid::Stated {
            cube_count: cube_count.clone(),
            cube_dim,
        },
    )
    .unwrap_or_else(|refusal| panic!("dequantize_tiled: {refusal}"));
    // One scale for the whole tensor: a scale tile over no axis of the values.
    let per_tensor = Projection::new(&[M, N], &[PhysicalAxisMap::broadcast()]);
    let values = TileArgLaunch::new(
        input.into_tensor_arg(),
        1,
        TileSpec::direct(&[M, N]).packed(field),
    )
    .scales(TileArgLaunch::new(
        scales.into_tensor_arg(),
        1,
        TileSpec::new(per_tensor),
    ));
    dequantize::launch(
        client,
        cube_count,
        cube_dim,
        values,
        TileArgLaunch::new(
            output.into_tensor_arg(),
            per_word,
            TileSpec::direct(&[M, N]),
        ),
        launch.partitioning_arg(),
        output_dtype,
    );

    Ok(())
}

#[cube(launch)]
/// The values are fields of the scheme's element read out of `u32` words, served as `O` under
/// their scale, which the copy that says so multiplies in.
pub fn dequantize<O: Numeric>(
    input: &TileArg<'_, O>,
    output: &TileArg<'_, O>,
    space: Partitioning,
    #[define(O)] _output_dtype: ElemType,
) {
    let input = input.tile(&space);
    let mut output = output.tile(&space);
    output.dequantize_from(&input);
}
