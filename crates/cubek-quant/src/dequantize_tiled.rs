use crate::quantized_arg;
use cubecl::tile::{
    Launcher, Partitioning, Projection, Space, TileSpec, TileTensor, TileTensorLaunch,
    kind::Field,
    launch::{Decline, Grid},
    layout::PhysicalAxisMap,
};
use cubecl::{
    ir::ElemType,
    prelude::*,
    quant::scheme::{BlockSize, QuantScheme, QuantStore, ScaleDtype},
};
use cubek_std::matmul_dimensions::{M, N};

/// A scheme whose values are not stored natively.
pub const DEQUANTIZE_TILED_STORE_NOT_NATIVE: Decline = Decline::Unserved {
    why: "the tiled dequantize reads values stored natively",
};

/// A scheme with more than one level of scales, or with block scales.
pub const DEQUANTIZE_TILED_SCALE_NOT_PER_TENSOR: Decline = Decline::Unserved {
    why: "the tiled dequantize reads one scale per tensor",
};

/// A scheme whose scales are not `f32`.
pub const DEQUANTIZE_TILED_SCALE_NOT_F32: Decline = Decline::Unserved {
    why: "the tiled dequantize reads f32 scales",
};

/// An input or an output that is not a matrix.
pub const DEQUANTIZE_TILED_OPERAND_NOT_A_MATRIX: Decline = Decline::Unserved {
    why: "the tiled dequantize reads and writes matrices",
};

/// Values other than one a byte.
pub const DEQUANTIZE_TILED_VALUE_NOT_A_BYTE: Decline = Decline::Unserved {
    why: "the tiled dequantize reads natively stored values one a byte",
};

/// An input whose rows are not contiguous runs of whole words.
pub const DEQUANTIZE_TILED_ROWS_NOT_WORDS: Decline = Decline::Unserved {
    why: "the tiled dequantize reads rows that are contiguous runs of whole u32 words",
};

/// An output of another shape than the input's.
pub const DEQUANTIZE_TILED_OUTPUT_UNREAD: Decline = Decline::Unserved {
    why: "the tiled dequantize writes an output of its input's shape",
};

/// Scales other than one value for the whole tensor.
pub const DEQUANTIZE_TILED_SCALES_NOT_ONE: Decline = Decline::Unserved {
    why: "the tiled dequantize reads one scale for the whole tensor",
};

/// A word of values wider than the device's widest line.
pub const DEQUANTIZE_TILED_WORD_PAST_LINE: Decline = Decline::Unserved {
    why: "the tiled dequantize reads a word of values as one line, wider than this device's lines",
};

/// Convert the tensor back to a higher precision data type.
/// Uses the tile-based implementation for dequantization.
/// Very WIP and naive implementation for now.
///
/// The decode is the kernel's own statement: the values are fields of the scheme's element read
/// out of words, under their scale, and `output.dequantize_from(&values)` multiplies each by the
/// scale where it is written. Nothing decodes behind a read.
///
/// # Errors
///
/// [`DEQUANTIZE_TILED_STORE_NOT_NATIVE`], [`DEQUANTIZE_TILED_SCALE_NOT_PER_TENSOR`],
/// [`DEQUANTIZE_TILED_SCALE_NOT_F32`] and [`DEQUANTIZE_TILED_VALUE_NOT_A_BYTE`] of `scheme`;
/// [`DEQUANTIZE_TILED_OPERAND_NOT_A_MATRIX`], [`DEQUANTIZE_TILED_OUTPUT_UNREAD`],
/// [`DEQUANTIZE_TILED_SCALES_NOT_ONE`] and [`DEQUANTIZE_TILED_ROWS_NOT_WORDS`] of the operands; [`DEQUANTIZE_TILED_WORD_PAST_LINE`] of the device; and why the tile engine cannot
/// launch or bind it on this device.
pub fn launch_ref(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    scales: TensorBinding,
    scheme: &QuantScheme,
    output_dtype: ElemType,
) -> Result<(), Decline> {
    if scheme.store != QuantStore::Native {
        return Err(DEQUANTIZE_TILED_STORE_NOT_NATIVE);
    }
    if scheme.levels().count() != 1 || scheme.finest().size != BlockSize::TENSOR {
        return Err(DEQUANTIZE_TILED_SCALE_NOT_PER_TENSOR);
    }
    if scheme.finest().dtype != ScaleDtype::F32 {
        return Err(DEQUANTIZE_TILED_SCALE_NOT_F32);
    }
    if input.shape.len() != 2 || output.shape.len() != 2 {
        return Err(DEQUANTIZE_TILED_OPERAND_NOT_A_MATRIX);
    }
    if input.shape != output.shape {
        return Err(DEQUANTIZE_TILED_OUTPUT_UNREAD);
    }
    if scales.shape.iter().product::<usize>() != 1 {
        return Err(DEQUANTIZE_TILED_SCALES_NOT_ONE);
    }

    // A native store keeps one value a byte, so the buffer is read as words of fields: each
    // `u32` holds `per_word` of them, decoded by the scheme's own field. The binding still counts
    // values, as a packed operand's does.
    let field = Field::Quant(scheme.value);
    let per_word = field.per_word();
    let (rows, cols) = (input.shape[0], input.shape[1]);
    if field.size_bits() != 8 {
        return Err(DEQUANTIZE_TILED_VALUE_NOT_A_BYTE);
    }
    if input.strides[1] != 1 || input.strides[0] != cols || !cols.is_multiple_of(per_word) {
        return Err(DEQUANTIZE_TILED_ROWS_NOT_WORDS);
    }
    if per_word > client.properties().hardware.max_vector_size {
        return Err(DEQUANTIZE_TILED_WORD_PAST_LINE);
    }

    // One tile covering every dimension, walked by a single cube: no level cuts it, so there is
    // nothing to list and the grid is one cube.
    let space = Space::new(&[(M, rows), (N, cols)]);
    let plane_size = client.properties().hardware.plane_size_max;
    let (cube_count, cube_dim) = (CubeCount::Static(1, 1, 1), CubeDim::new_2d(plane_size, 1));
    let launch = Launcher::new(
        client,
        // Static: the decoding copy reads through the values' own dimensions, which it needs the
        // extents of at expansion, so the kernel is compiled per shape.
        Partitioning::new(space.clone(), vec![]),
        &space,
        Grid::Stated {
            cube_count: cube_count.clone(),
            cube_dim,
        },
    )?;
    // One scale for the whole tensor: a scale tile over no dimension of the values.
    let per_tensor = Projection::new(&[M, N], &[PhysicalAxisMap::broadcast()]);
    let values = quantized_arg(
        scheme,
        launch
            .arg(input, u32::elem_type_native())
            .dimensions(&[M, N]),
        [launch
            .arg(scales, u32::elem_type_native())
            .gathered(per_tensor)],
    )?;
    dequantize::launch(
        client,
        cube_count,
        cube_dim,
        values,
        TileTensorLaunch::new(
            output.into_tensor_arg(),
            output_dtype,
            per_word,
            TileSpec::direct(&[M, N]),
        ),
        launch.partitioning_arg(),
    );

    Ok(())
}

#[cube(launch)]
/// The values are fields of the scheme's element read out of `u32` words, served as the output's
/// element under their scale, which the copy that says so multiplies in.
///
/// - `input`: the values, `u32` words of the scheme's fields bound over `[M, N]`, and their one
///   `f32` scale, broadcast over every value.
/// - `output`: the dequantized `[M, N]` matrix, in the output's element.
/// - `space`: the one tile over `[M, N]` the single cube walks.
pub fn dequantize(input: &TileTensor<'_>, output: &mut TileTensor<'_>, space: Partitioning) {
    let input = input.tile(&space);
    let mut output = output.tile_mut(&space);
    output.dequantize_from(&input);
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::quant::scheme::QuantValue;
    use cubecl::std::tensor::TensorHandle;

    /// The scheme the tiled dequantize reads: `q8s` values stored natively under one `f32` scale.
    fn native_per_tensor() -> QuantScheme {
        QuantScheme::default()
            .per_tensor(ScaleDtype::F32)
            .with_store(QuantStore::Native)
            .with_value(QuantValue::Q8S)
    }

    /// What the tiled dequantize of an `input` of `shape` into an `output` of `out` under `scales`
    /// of `scales` and `scheme` declines.
    fn declined(
        shape: &[usize],
        out: &[usize],
        scales: &[usize],
        scheme: QuantScheme,
    ) -> Option<Decline> {
        let client = cubecl::test_device().client();
        let u32 = u32::elem_type_native();
        let f32 = f32::elem_type_native();
        let tensor =
            |shape: &[usize], element| TensorHandle::zeros(&client, shape.to_vec(), element);
        launch_ref(
            &client,
            tensor(shape, u32).binding(),
            tensor(out, f32).binding(),
            tensor(scales, f32).binding(),
            &scheme,
            f32,
        )
        .err()
    }

    #[test]
    fn a_packed_store_declines() {
        let scheme = native_per_tensor().with_store(QuantStore::PackedU32(0));

        let declined = declined(&[8, 8], &[8, 8], &[1], scheme);

        assert_eq!(declined, Some(DEQUANTIZE_TILED_STORE_NOT_NATIVE));
    }

    #[test]
    fn block_scales_decline() {
        let scheme = native_per_tensor().per_block([8], ScaleDtype::F32);

        let declined = declined(&[8, 8], &[8, 8], &[1], scheme);

        assert_eq!(declined, Some(DEQUANTIZE_TILED_SCALE_NOT_PER_TENSOR));
    }

    #[test]
    fn half_scales_decline() {
        let scheme = native_per_tensor().per_tensor(ScaleDtype::F16);

        let declined = declined(&[8, 8], &[8, 8], &[1], scheme);

        assert_eq!(declined, Some(DEQUANTIZE_TILED_SCALE_NOT_F32));
    }

    #[test]
    fn an_operand_other_than_a_matrix_declines() {
        let declined = declined(&[64], &[64], &[1], native_per_tensor());

        assert_eq!(declined, Some(DEQUANTIZE_TILED_OPERAND_NOT_A_MATRIX));
    }

    #[test]
    fn an_output_of_another_shape_declines() {
        let declined = declined(&[8, 8], &[4, 8], &[1], native_per_tensor());

        assert_eq!(declined, Some(DEQUANTIZE_TILED_OUTPUT_UNREAD));
    }

    #[test]
    fn scales_of_more_than_one_value_decline() {
        let declined = declined(&[8, 8], &[8, 8], &[2], native_per_tensor());

        assert_eq!(declined, Some(DEQUANTIZE_TILED_SCALES_NOT_ONE));
    }

    #[test]
    fn values_other_than_a_byte_decline() {
        let scheme = native_per_tensor().with_value(QuantValue::Q4F);

        let declined = declined(&[8, 8], &[8, 8], &[1], scheme);

        assert_eq!(declined, Some(DEQUANTIZE_TILED_VALUE_NOT_A_BYTE));
    }

    #[test]
    fn rows_of_no_whole_words_decline() {
        let declined = declined(&[8, 6], &[8, 6], &[1], native_per_tensor());

        assert_eq!(declined, Some(DEQUANTIZE_TILED_ROWS_NOT_WORDS));
    }
}
