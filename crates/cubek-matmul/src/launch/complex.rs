//! Modified from tensor4all/cubek `07567557` (MIT OR Apache-2.0):
//! use current upstream real matmul kernels without changing caller descriptors.

use crate::{
    definition::{MatmulElems, MatmulGlobalElems, MatmulSetupError, broadcast_batches},
    launch::launch_ref,
    strategy::Strategy,
};
use cubecl::{
    calculate_cube_count_elemwise,
    client::Client,
    ir::{ComplexKind, ElemType},
    prelude::*,
    zspace::{Shape, Strides, Tiling},
};
use cubek_std::{InputBinding, MatrixLayout};

type C32Parts = Vector<f32, Const<2>>;

/// Options for CubeK-owned `C32` matrix multiplication.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComplexMatmulOptions {
    pub lhs_conj: bool,
    pub rhs_conj: bool,
}

#[derive(Clone, Copy)]
enum C32Part {
    Real,
    Imag,
}

/// Launch a `C32` matmul by lowering to four real `F32` matmuls.
///
/// The output matrix shape must match the product, with NumPy-style batch
/// broadcasting; output reshaping is not supported by this C32 entry point.
#[allow(clippy::result_large_err)]
pub fn launch_c32_ref(
    strategy: &Strategy,
    client: &Client,
    lhs: InputBinding,
    rhs: InputBinding,
    out: TensorBinding,
    dtypes: &mut MatmulElems,
    options: ComplexMatmulOptions,
) -> Result<(), MatmulSetupError> {
    validate_c32_globals(dtypes)?;
    validate_normal_input("lhs", &lhs)?;
    validate_normal_input("rhs", &rhs)?;
    validate_rank("lhs", lhs.shape())?;
    validate_rank("rhs", rhs.shape())?;
    validate_rank("out", &out.shape)?;
    if out.tiling.is_tiled() {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "complex GEMM does not support storage-tiled output",
        )));
    }
    let lhs_shape = lhs.shape().as_slice();
    let rhs_shape = rhs.shape().as_slice();
    let out_shape = out.shape.as_slice();
    if lhs_shape[lhs_shape.len() - 1] != rhs_shape[rhs_shape.len() - 2]
        || lhs_shape[lhs_shape.len() - 2] != out_shape[out_shape.len() - 2]
        || rhs_shape[rhs_shape.len() - 1] != out_shape[out_shape.len() - 1]
        || broadcast_batches(
            &lhs_shape[..lhs_shape.len() - 2],
            &rhs_shape[..rhs_shape.len() - 2],
        )
        .as_deref()
            != Some(&out_shape[..out_shape.len() - 2])
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "complex GEMM contraction, output, or batch shape mismatch",
        )));
    }
    validate_storage("lhs", lhs.data())?;
    validate_storage("rhs", rhs.data())?;
    validate_storage("out", &out)?;
    validate_output_strides(&out)?;

    let mut lhs_real = extract_part(client, lhs.data(), C32Part::Real, MatrixLayout::RowMajor)?;
    let mut lhs_imag = extract_part(client, lhs.data(), C32Part::Imag, MatrixLayout::RowMajor)?;
    let mut rhs_real = extract_part(client, rhs.data(), C32Part::Real, MatrixLayout::ColMajor)?;
    let mut rhs_imag = extract_part(client, rhs.data(), C32Part::Imag, MatrixLayout::ColMajor)?;
    // The real matmul batch layout expects one stride per output batch axis.
    for binding in [&mut lhs_real, &mut lhs_imag, &mut rhs_real, &mut rhs_imag] {
        while binding.shape.len() < out.shape.len() {
            binding.shape.insert(0, 1);
            binding.strides.insert(0, 0);
        }
    }

    let out_shape = out.shape.clone();
    let out_strides = dense_matrix_strides(out_shape.as_slice(), MatrixLayout::RowMajor)?;
    let real_pos = scratch_like(client, &out_shape, &out_strides)?;
    let real_neg = scratch_like(client, &out_shape, &out_strides)?;
    let imag_left = scratch_like(client, &out_shape, &out_strides)?;
    let imag_right = scratch_like(client, &out_shape, &out_strides)?;

    let f32_type = f32::elem_type_native();
    let mut real_dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: f32_type,
        rhs: f32_type,
        out: f32_type,
    });

    launch_ref(
        strategy,
        client,
        InputBinding::new(lhs_real.clone(), f32_type),
        InputBinding::new(rhs_real.clone(), f32_type),
        real_pos.clone(),
        &mut real_dtypes,
    )?;
    launch_ref(
        strategy,
        client,
        InputBinding::new(lhs_imag.clone(), f32_type),
        InputBinding::new(rhs_imag.clone(), f32_type),
        real_neg.clone(),
        &mut real_dtypes,
    )?;
    launch_ref(
        strategy,
        client,
        InputBinding::new(lhs_real, f32_type),
        InputBinding::new(rhs_imag, f32_type),
        imag_left.clone(),
        &mut real_dtypes,
    )?;
    launch_ref(
        strategy,
        client,
        InputBinding::new(lhs_imag, f32_type),
        InputBinding::new(rhs_real, f32_type),
        imag_right.clone(),
        &mut real_dtypes,
    )?;

    compose_parts(
        client,
        &out,
        &real_pos,
        &real_neg,
        &imag_left,
        &imag_right,
        options,
    )
}

#[cube(launch_unchecked)]
fn extract_c32_part_kernel(
    input_meta: &Tensor<f32>,
    input_parts: &[C32Parts],
    out_parts: &mut Tensor<f32>,
    #[comptime] part: u32,
    #[comptime] rank: usize,
) {
    if ABSOLUTE_POS >= out_parts.len() {
        terminate!();
    }
    let mut input_offset = 0usize;
    #[unroll]
    for axis in 0..rank {
        let coord = out_parts.coordinate(ABSOLUTE_POS, axis);
        input_offset += coord * input_meta.stride(axis);
    }
    let value = input_parts[input_offset];
    out_parts[ABSOLUTE_POS] = if part == 0 {
        value.extract(0usize)
    } else {
        value.extract(1usize)
    };
}

#[cube(launch_unchecked)]
fn compose_c32_parts_kernel(
    out_meta: &Tensor<f32>,
    out_parts: &mut [C32Parts],
    real_pos: &Tensor<f32>,
    real_neg: &Tensor<f32>,
    imag_left: &Tensor<f32>,
    imag_right: &Tensor<f32>,
    #[comptime] lhs_imag_sign: i32,
    #[comptime] rhs_imag_sign: i32,
    #[comptime] rank: usize,
) {
    if ABSOLUTE_POS >= real_pos.len() {
        terminate!();
    }
    let mut out_offset = 0usize;
    let mut dense_offset = 0usize;
    #[unroll]
    for axis in 0..rank {
        let coord = real_pos.coordinate(ABSOLUTE_POS, axis);
        out_offset += coord * out_meta.stride(axis);
        dense_offset += coord * real_pos.stride(axis);
    }
    let lhs_sign = f32::cast_from(lhs_imag_sign);
    let rhs_sign = f32::cast_from(rhs_imag_sign);
    let real = real_pos[dense_offset] - lhs_sign * rhs_sign * real_neg[dense_offset];
    let imag = rhs_sign * imag_left[dense_offset] + lhs_sign * imag_right[dense_offset];
    let mut value = Vector::<f32, Const<2>>::empty();
    value.insert(0usize, real);
    value.insert(1usize, imag);
    out_parts[out_offset] = value;
}

fn extract_part(
    client: &Client,
    input: &TensorBinding,
    part: C32Part,
    layout: MatrixLayout,
) -> Result<TensorBinding, MatmulSetupError> {
    let strides = dense_matrix_strides(input.shape.as_slice(), layout)?;
    let out = scratch_like(client, &input.shape, &strides)?;
    let len = logical_len(input.shape.as_slice())?;
    let cube_dim = CubeDim::new(client, len);
    let cube_count = calculate_cube_count_elemwise(client, len, cube_dim);
    let part_value = match part {
        C32Part::Real => 0,
        C32Part::Imag => 1,
    };
    let input_parts = c32_array_arg(input);
    unsafe {
        extract_c32_part_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            input.clone().into_tensor_arg(),
            input_parts,
            out.clone().into_tensor_arg(),
            part_value,
            input.shape.len(),
        )
    };
    Ok(out)
}

fn compose_parts(
    client: &Client,
    out: &TensorBinding,
    real_pos: &TensorBinding,
    real_neg: &TensorBinding,
    imag_left: &TensorBinding,
    imag_right: &TensorBinding,
    options: ComplexMatmulOptions,
) -> Result<(), MatmulSetupError> {
    let len = logical_len(out.shape.as_slice())?;
    let cube_dim = CubeDim::new(client, len);
    let cube_count = calculate_cube_count_elemwise(client, len, cube_dim);
    let lhs_sign = if options.lhs_conj { -1 } else { 1 };
    let rhs_sign = if options.rhs_conj { -1 } else { 1 };
    let out_meta = metadata_with_backing(out, real_pos);
    let out_parts = c32_array_arg(out);
    unsafe {
        compose_c32_parts_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            out_meta.into_tensor_arg(),
            out_parts,
            real_pos.clone().into_tensor_arg(),
            real_neg.clone().into_tensor_arg(),
            imag_left.clone().into_tensor_arg(),
            imag_right.clone().into_tensor_arg(),
            lhs_sign,
            rhs_sign,
            out.shape.len(),
        )
    };
    Ok(())
}

fn validate_normal_input(name: &'static str, input: &InputBinding) -> Result<(), MatmulSetupError> {
    match input {
        InputBinding::Normal(binding, dtype)
            if *dtype == c32_elem_type() && !binding.tiling.is_tiled() =>
        {
            Ok(())
        }
        InputBinding::Normal(binding, _) if binding.tiling.is_tiled() => {
            Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "complex GEMM {name} does not support storage-tiled input"
            ))))
        }
        InputBinding::Normal(_, dtype) => Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "complex GEMM {name} must use C32 storage, got {dtype:?}"
        )))),
        InputBinding::Quantized { .. } => Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "complex GEMM {name} does not support quantized input"
        )))),
    }
}

fn validate_c32_globals(dtypes: &MatmulElems) -> Result<(), MatmulSetupError> {
    let c32_type = c32_elem_type();
    if dtypes.lhs_global != c32_type
        || dtypes.rhs_global != c32_type
        || dtypes.acc_global != c32_type
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "complex GEMM dtypes must be C32, got lhs={:?}, rhs={:?}, out={:?}",
            dtypes.lhs_global, dtypes.rhs_global, dtypes.acc_global
        ))));
    }
    Ok(())
}

fn validate_rank(name: &'static str, shape: &Shape) -> Result<(), MatmulSetupError> {
    if shape.len() < 2 {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "complex GEMM {name} must have rank at least 2"
        ))));
    }
    Ok(())
}

fn validate_storage(name: &str, tensor: &TensorBinding) -> Result<(), MatmulSetupError> {
    let invalid = || {
        MatmulSetupError::InvalidConfig(Box::new(format!(
            "complex GEMM {name} shape/strides exceed C32 binding"
        )))
    };
    let shape = tensor.shape.as_slice();
    let strides = &tensor.strides[..];
    let binding = &tensor.handle;
    let start = binding.offset_start.unwrap_or(0);
    let end = binding.offset_end.unwrap_or(0);
    if shape.len() != strides.len()
        || shape.contains(&0)
        || !start.is_multiple_of(8)
        || !end.is_multiple_of(8)
        || start
            .checked_add(end)
            .is_none_or(|total| total > binding.size)
    {
        return Err(invalid());
    }
    let available = (binding.size - start - end) / 8;
    let highest = shape
        .iter()
        .zip(strides)
        .try_fold(0usize, |pos, (&dim, &stride)| {
            pos.checked_add((dim - 1).checked_mul(stride)?)
        });
    if highest
        .and_then(|pos| pos.checked_add(1))
        .is_none_or(|needed| needed as u64 > available)
    {
        return Err(invalid());
    }
    Ok(())
}

fn validate_output_strides(out: &TensorBinding) -> Result<(), MatmulSetupError> {
    let mut axes: Vec<_> = out.shape.iter().zip(out.strides.iter()).collect();
    axes.sort_by_key(|(_, stride)| **stride);
    // The occupied span of the axes processed so far, not `stride * dim`:
    // a `[2, 2]` output with strides `[3, 2]` reaches `{0, 2, 3, 5}` and is
    // valid, while `stride * dim` would report the second axis as overlapping.
    let mut span = 1usize;
    for (&dim, &stride) in axes {
        if dim > 1 {
            if stride < span {
                return Err(MatmulSetupError::InvalidConfig(Box::new(
                    "complex GEMM output strides overlap",
                )));
            }
            span = (dim - 1)
                .checked_mul(stride)
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| {
                    MatmulSetupError::InvalidConfig(Box::new(
                        "complex GEMM output strides overflow",
                    ))
                })?;
        }
    }
    Ok(())
}

fn scratch_like(
    client: &Client,
    shape: &Shape,
    strides: &Strides,
) -> Result<TensorBinding, MatmulSetupError> {
    let len = logical_len(shape.as_slice())?;
    let bytes = len
        .checked_mul(f32::elem_type_native().size())
        .ok_or_else(|| {
            MatmulSetupError::InvalidConfig(Box::new("complex GEMM scratch size overflow"))
        })?;
    let handle = client.empty(bytes).binding();
    Ok(TensorBinding {
        handle,
        strides: strides.clone(),
        shape: shape.clone(),
        tiling: Tiling::UNTILED,
    })
}

fn dense_matrix_strides(
    shape: &[usize],
    layout: MatrixLayout,
) -> Result<Strides, MatmulSetupError> {
    let rank = shape.len();
    let mut strides = vec![0usize; rank];
    let rows = shape[rank - 2];
    let cols = shape[rank - 1];
    match layout {
        MatrixLayout::RowMajor => {
            strides[rank - 2] = cols;
            strides[rank - 1] = 1;
        }
        MatrixLayout::ColMajor => {
            strides[rank - 2] = 1;
            strides[rank - 1] = rows;
        }
    }
    let mut batch_stride = rows.checked_mul(cols).ok_or_else(|| {
        MatmulSetupError::InvalidConfig(Box::new("complex GEMM batch stride overflow"))
    })?;
    for axis in (0..rank - 2).rev() {
        strides[axis] = batch_stride;
        batch_stride = batch_stride.checked_mul(shape[axis]).ok_or_else(|| {
            MatmulSetupError::InvalidConfig(Box::new("complex GEMM batch stride overflow"))
        })?;
    }
    Ok(Strides::new(&strides))
}

fn c32_array_arg(binding: &TensorBinding) -> BufferArg {
    let len = binding.handle.size_in_used() as usize / c32_elem_type().size();
    unsafe { BufferArg::from_raw_parts_binding(binding.handle.clone(), len) }
}

fn metadata_with_backing(metadata: &TensorBinding, backing: &TensorBinding) -> TensorBinding {
    TensorBinding {
        handle: backing.handle.clone(),
        strides: metadata.strides.clone(),
        shape: metadata.shape.clone(),
        tiling: metadata.tiling,
    }
}

fn logical_len(shape: &[usize]) -> Result<usize, MatmulSetupError> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            MatmulSetupError::InvalidConfig(Box::new("complex GEMM logical length overflow"))
        })
    })
}

fn c32_elem_type() -> ElemType {
    ElemType::Complex(ComplexKind::C32)
}
