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
    quant::scheme::{QuantScheme, QuantStore, ScaleDtype},
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
/// [`DEQUANTIZE_TILED_OPERAND_NOT_A_MATRIX`] and [`DEQUANTIZE_TILED_ROWS_NOT_WORDS`] of the
/// operands; [`DEQUANTIZE_TILED_WORD_PAST_LINE`] of the device; and why the tile engine cannot
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
    if scheme.num_levels() != 1 || scheme.block_size().is_some() {
        return Err(DEQUANTIZE_TILED_SCALE_NOT_PER_TENSOR);
    }
    if scheme.scale_dtype() != ScaleDtype::F32 {
        return Err(DEQUANTIZE_TILED_SCALE_NOT_F32);
    }
    if input.shape.len() != 2 || output.shape.len() != 2 {
        return Err(DEQUANTIZE_TILED_OPERAND_NOT_A_MATRIX);
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
