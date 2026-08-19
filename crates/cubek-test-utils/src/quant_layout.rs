//! The shapes and element types a [`QuantScheme`] implies.
//!
//! Every quantized tensor a test builds has to agree with the kernels about where the values and
//! the scales sit, and that agreement is the scheme's to make, not each test's. Geometry only:
//! whether a scheme is supported at all is the caller's question, since a reference quantizer and
//! a kernel refuse different things.

use cubecl::ir::{ElemType, UIntKind};
use cubecl_common::quant::scheme::{QuantScheme, QuantStore};

/// Per-axis block extents, the number of elements each scale covers along each axis. A scheme with
/// no block level has one scale for the whole tensor, so its block spans everything.
pub fn block_dims(scheme: &QuantScheme, shape: &[usize]) -> Vec<usize> {
    match scheme.block_size() {
        Some(block) => block
            .to_dim_vec(shape.len())
            .into_iter()
            .map(usize::from)
            .collect(),
        None => shape.to_vec(),
    }
}

/// The block scale grid: each axis divided by its block extent, so one entry per block. A
/// two-level scheme grids exactly like a one-level block scheme; its outer scale is a separate
/// one-element tensor and does not appear here.
pub fn scales_shape(scheme: &QuantScheme, shape: &[usize]) -> Vec<usize> {
    scales_grid(shape, &block_dims(scheme, shape))
}

/// The same grid, over block extents the caller already holds.
pub fn scales_grid(shape: &[usize], block_dims: &[usize]) -> Vec<usize> {
    assert_eq!(
        shape.len(),
        block_dims.len(),
        "shape/block_dims rank mismatch"
    );
    shape
        .iter()
        .zip(block_dims)
        .map(|(&dim, &block)| {
            assert!(
                block > 0 && dim.is_multiple_of(block),
                "axis of {dim} elements does not divide into {block}-element blocks"
            );
            dim / block
        })
        .collect()
}

/// The element type the quantized values are stored at.
pub fn values_dtype(scheme: &QuantScheme) -> ElemType {
    match scheme.store {
        QuantStore::PackedU32(_) => ElemType::UInt(UIntKind::U32),
        QuantStore::PackedNative(_) | QuantStore::Native => {
            ElemType::from_quant_value(scheme.value)
        }
    }
}

/// The shape the quantized values occupy. A packed store carries `num_quants` of them per stored
/// integer along the innermost axis, so the shape is from the point of view of those integers;
/// a native store holds one value per element and keeps `shape`.
pub fn values_shape(scheme: &QuantScheme, shape: &[usize]) -> Vec<usize> {
    let mut packed = shape.to_vec();

    match scheme.store {
        QuantStore::PackedU32(_) | QuantStore::PackedNative(_) => {
            let last = packed.len() - 1;
            packed[last] /= scheme.num_quants();
        }
        QuantStore::Native => {}
    }

    packed
}

/// The element type the innermost level's scales are stored at.
pub fn scales_dtype(scheme: &QuantScheme) -> ElemType {
    ElemType::from_scale_dtype(scheme.scale_dtype())
}
