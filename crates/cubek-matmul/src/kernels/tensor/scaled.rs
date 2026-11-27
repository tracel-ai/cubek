//! A matrix of 8-bit floats under one level of scales constant along the contraction.

use super::PlainTensor;
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::TensorBinding;
use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue};
use cubecl::tile::launch::Decline;
use cubecl::tile::layout::PhysicalAxisMap;
use cubecl::tile::{Dimension, Launcher, Projection, TileTensorLaunch};
use cubecl::zspace::metadata::Metadata;
use cubek_microkernel::matmul::MatmulBlocks;
use cubek_std::layout::stored::Operand;
use cubek_std::matmul_dimensions::{K, blocked};
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Breakable};

/// What [`ScaledTensor::of_quantized`] declines: a scheme it does not read
/// ([`ScaledTensor::reads`]).
pub const SCALES_WITHIN_CONTRACTION: Decline = Decline::Unserved {
    why: "a matrix under scales alone is 8-bit floats a byte each, under one level of scales, \
          each over the whole contraction and over one position or all of the other dimension",
};

/// What [`ScaledTensor::of_quantized`] declines: 8-bit floats packed four to a word in tiles,
/// whose strides count words where a tile's pieces count values.
pub const PACKED_FP8_TILES_UNREAD: Decline = Decline::Unserved {
    why: "8-bit floats stored in tiles are read a byte a value, not packed in words",
};

/// The strides of a run of scales, one a row of a single column.
const ONE_COLUMN_STRIDES: [usize; 2] = [1, 1];

/// A matrix of 8-bit floats as a kernel reads it: its values where they lie, and the scales they are
/// under, each constant along the contraction, so a sum takes them once.
#[derive(Debug, Clone)]
pub struct ScaledTensor {
    /// The values, at the 8-bit float they are stored in.
    pub values: PlainTensor,
    /// One scale for each position of `scaled_along`, or one for all of it.
    pub scales: PlainTensor,
    pub scheme: QuantScheme,
    /// The dimension the scales vary along, where they vary.
    pub scaled_along: Option<Dimension>,
}

impl ScaledTensor {
    /// A quantized matrix over `dimensions`, in the order its shape states them last: `values`, its
    /// buffer bound under its logical shape and strides, stored as `scheme` says, under its
    /// `scales`.
    ///
    /// # Errors
    ///
    /// [`SCALES_WITHIN_CONTRACTION`] and [`PACKED_FP8_TILES_UNREAD`].
    pub fn of_quantized(
        client: &Client,
        values: TensorBinding,
        scheme: QuantScheme,
        scales: PlainTensor,
        dimensions: [Dimension; 2],
    ) -> Result<Self, Decline> {
        let scaled_along =
            Self::scaled_along(&scheme, dimensions).ok_or(SCALES_WITHIN_CONTRACTION)?;
        let element = Self::element(&scheme).ok_or(SCALES_WITHIN_CONTRACTION)?;
        Ok(Self {
            values: Self::bytes_of(values, &scheme, element)?,
            scales: scales.into_contiguous(client),
            scheme,
            scaled_along,
        })
    }

    /// Whether [`of_quantized`](Self::of_quantized) reads a matrix over `dimensions` stored as
    /// `scheme` says, but for how its values lie.
    pub fn reads(scheme: &QuantScheme, dimensions: [Dimension; 2]) -> bool {
        Self::element(scheme).is_some() && Self::scaled_along(scheme, dimensions).is_some()
    }

    /// `values`, bound as a tile, under the scales: one operand the kernel factors them out of.
    ///
    /// # Errors
    ///
    /// [`Decline::Broken`] where the tile engine refuses the scales as the routine prepared them.
    pub fn under_scales(
        &self,
        launcher: &Launcher,
        blocks: &MatmulBlocks,
        values: TileTensorLaunch,
    ) -> Result<TileTensorLaunch, Decline> {
        // A dimension no scale groups is one block, the position inside it the whole of it.
        let (position, axis, count) = match self.scaled_along {
            Some(dimension) => Self::varying(blocks, dimension),
            None => (blocked(K).1, PhysicalAxisMap::broadcast(), 1),
        };
        let scales = self
            .scales
            .bound_as(Metadata::new([count, 1], ONE_COLUMN_STRIDES));
        let scales = launcher
            .arg(scales, self.scales.elem)
            .gathered(Projection::new(
                &[position],
                &[axis, PhysicalAxisMap::broadcast()],
            ))
            .build()
            .or_broken(ADMITTED_PLAN_RUNS)?;
        Ok(values.scales(scales.arg()))
    }

    /// The position inside `dimension`'s one block, the axis of scales along it, and how many.
    fn varying(blocks: &MatmulBlocks, dimension: Dimension) -> (Dimension, PhysicalAxisMap, usize) {
        let position = blocked(dimension).1;
        (
            position,
            PhysicalAxisMap::of(position),
            blocks.extent(dimension),
        )
    }

    /// The 8-bit float `scheme` stores its values as, a byte each or four to a word along the
    /// innermost dimension. `None` for any other.
    fn element(scheme: &QuantScheme) -> Option<ElemType> {
        let stored = matches!(scheme.store, QuantStore::Native | QuantStore::PackedU32(0));
        match scheme.value {
            QuantValue::E4M3 if stored => Some(ElemType::Float(FloatKind::E4M3)),
            QuantValue::E5M2 if stored => Some(ElemType::Float(FloatKind::E5M2)),
            _ => None,
        }
    }

    /// The dimension its scales vary along, over one position of it a scale. `None` for a scheme of
    /// more levels than one, or whose scales vary otherwise.
    fn scaled_along(scheme: &QuantScheme, dimensions: [Dimension; 2]) -> Option<Option<Dimension>> {
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
    fn bytes_of(
        values: TensorBinding,
        scheme: &QuantScheme,
        element: ElemType,
    ) -> Result<PlainTensor, Decline> {
        match (scheme.store, values.tiling.is_tiled()) {
            (QuantStore::PackedU32(_), true) => Err(PACKED_FP8_TILES_UNREAD),
            (QuantStore::PackedU32(_), false) => {
                Ok(PlainTensor::new(Self::unpacked(values, scheme), element))
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
}

#[cfg(test)]
mod tests {
    use super::ScaledTensor;
    use cubecl::quant::scheme::{BlockScale, QuantScheme, QuantStore, QuantValue, ScaleDtype};
    use cubek_std::matmul_dimensions::{K, M};

    #[test]
    fn one_scale_a_row_of_8_bit_floats_is_read() {
        assert!(ScaledTensor::reads(&QuantScheme::fp8_per_row(), [M, K]));
    }

    #[test]
    fn one_scale_a_row_across_the_contraction_is_not_read() {
        assert!(!ScaledTensor::reads(&QuantScheme::fp8_per_row(), [K, M]));
    }

    #[test]
    fn blocks_along_the_contraction_are_not_read() {
        let blocks = QuantScheme::default()
            .with_value(QuantValue::E4M3)
            .with_store(QuantStore::PackedU32(0))
            .with_levels([BlockScale::per_block([32], ScaleDtype::UE8M0)]);
        assert!(!ScaledTensor::reads(&blocks, [M, K]));
    }

    #[test]
    fn values_other_than_8_bit_floats_are_not_read() {
        let integers = QuantScheme::fp8_per_row().with_value(QuantValue::Q8S);
        assert!(!ScaledTensor::reads(&integers, [M, K]));
    }
}
