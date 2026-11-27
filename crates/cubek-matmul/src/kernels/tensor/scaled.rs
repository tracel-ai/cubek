//! How 8-bit floats under one level of scales constant along the contraction are read: the facts
//! of a scheme that says so, which [`StoredTensor`](super::StoredTensor) asks.

use super::PlainTensor;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::TensorBinding;
use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue};
use cubecl::tile::Dimension;
use cubecl::tile::launch::Decline;
use cubek_std::layout::stored::Operand;
use cubek_std::matmul_dimensions::K;

/// What [`StoredTensor::of_quantized`](super::StoredTensor::of_quantized) declines: a scheme it does not read
pub const SCALES_WITHIN_CONTRACTION: Decline = Decline::Unserved {
    why: "a matrix under scales alone is 8-bit floats a byte each, under one level of scales, \
          each over the whole contraction and over one position or all of the other dimension",
};

/// What [`StoredTensor::of_quantized`](super::StoredTensor::of_quantized) declines: 8-bit floats packed four to a word in tiles,
/// whose strides count words where a tile's pieces count values.
pub const PACKED_FP8_TILES_UNREAD: Decline = Decline::Unserved {
    why: "8-bit floats stored in tiles are read a byte a value, not packed in words",
};

/// Whether `scheme` stores a matrix over `dimensions` as 8-bit floats under one level of scales
/// constant along the contraction, but for how its values lie.
pub(crate) fn reads(scheme: &QuantScheme, dimensions: [Dimension; 2]) -> bool {
    element(scheme).is_some() && scaled_along(scheme, dimensions).is_some()
}

/// The 8-bit float `scheme` stores its values as, a byte each or four to a word along the
/// innermost dimension. `None` for any other.
pub(super) fn element(scheme: &QuantScheme) -> Option<ElemType> {
    let stored = matches!(scheme.store, QuantStore::Native | QuantStore::PackedU32(0));
    match scheme.value {
        QuantValue::E4M3 if stored => Some(ElemType::Float(FloatKind::E4M3)),
        QuantValue::E5M2 if stored => Some(ElemType::Float(FloatKind::E5M2)),
        _ => None,
    }
}

/// The dimension the scales of `scheme`'s one level vary along over `dimensions`, one position of
/// it a scale, or `Some(None)` for one scale over the matrix. `None` for a scheme of more levels
/// than one, or whose scales vary otherwise.
pub(super) fn scaled_along(
    scheme: &QuantScheme,
    dimensions: [Dimension; 2],
) -> Option<Option<Dimension>> {
    let contracted = dimensions.iter().position(|&dimension| dimension == K)?;
    let along = Operand::constant_along(&scheme.levels().next()?, contracted)?;
    (scheme.levels().count() == 1).then(|| along.map(|position| dimensions[position]))
}

/// `values`, stored as `scheme` says, as the `element` their bytes are: a word of four counted
/// as four values along the innermost dimension.
///
/// # Errors
///
/// [`PACKED_FP8_TILES_UNREAD`].
pub(super) fn bytes_of(
    values: TensorBinding,
    scheme: &QuantScheme,
    element: ElemType,
) -> Result<PlainTensor, Decline> {
    match (scheme.store, values.tiling.is_tiled()) {
        (QuantStore::PackedU32(_), true) => Err(PACKED_FP8_TILES_UNREAD),
        (QuantStore::PackedU32(_), false) => {
            Ok(PlainTensor::new(unpacked(values, scheme), element))
        }
        _ => Ok(PlainTensor::new(values, element)),
    }
}

/// `values`, words of `scheme`'s values along the innermost dimension, bound in values.
fn unpacked(mut values: TensorBinding, scheme: &QuantScheme) -> TensorBinding {
    let innermost = values.shape.rank() - 1;
    for dim in 0..innermost {
        values.strides[dim] *= scheme.num_quants();
    }
    values
}

#[cfg(test)]
mod tests {
    use super::reads;
    use cubecl::quant::scheme::{BlockScale, QuantScheme, QuantStore, QuantValue, ScaleDtype};
    use cubek_std::matmul_dimensions::{K, M};

    #[test]
    fn one_scale_a_row_of_8_bit_floats_is_read() {
        assert!(reads(&QuantScheme::fp8_per_row(), [M, K]));
    }

    #[test]
    fn one_scale_a_row_across_the_contraction_is_not_read() {
        assert!(!reads(&QuantScheme::fp8_per_row(), [K, M]));
    }

    #[test]
    fn blocks_along_the_contraction_are_not_read() {
        let blocks = QuantScheme::default()
            .with_value(QuantValue::E4M3)
            .with_store(QuantStore::PackedU32(0))
            .with_levels([BlockScale::per_block([32], ScaleDtype::UE8M0)]);
        assert!(!reads(&blocks, [M, K]));
    }

    #[test]
    fn values_other_than_8_bit_floats_are_not_read() {
        let integers = QuantScheme::fp8_per_row().with_value(QuantValue::Q8S);
        assert!(!reads(&integers, [M, K]));
    }
}
