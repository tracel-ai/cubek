//! A matrix of 8-bit floats under one level of scales constant along the contraction.

use super::PlainTensor;
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::TensorBinding;
use cubecl::quant::scheme::{BlockExtent, QuantScheme, QuantStore, QuantValue};
use cubecl::tile::launch::Decline;
use cubecl::tile::layout::PhysicalAxisMap;
use cubecl::tile::{Dimension, Launcher, Projection, TileTensorLaunch};
use cubecl::zspace::metadata::Metadata;
use cubek_microkernel::matmul::MatmulBlocks;
use cubek_std::matmul_dimensions::{K, blocked};
use cubek_std::routine::ADMITTED_PLAN_RUNS;

/// What [`ScaledTensor::of_quantized`] declines: values other than 8-bit floats a byte each, or a
/// scale that varies along the contraction, or along part of another dimension.
pub const SCALES_WITHIN_CONTRACTION: Decline = Decline::Unserved {
    why: "a matrix under scales alone is 8-bit floats a byte each, under one level of scales, \
          each over the whole contraction and over one position or all of the other dimension",
};

/// A matrix of 8-bit floats as a kernel reads it: its values where they lie, and the scales they are
/// under, each constant along the contraction, so a sum takes them once.
#[derive(Debug, Clone)]
pub struct ScaledTensor {
    values: PlainTensor,
    scales: PlainTensor,
    scheme: QuantScheme,
    /// The dimension the scales vary along, where they vary.
    scaled_along: Option<Dimension>,
}

impl ScaledTensor {
    /// A quantized matrix over `dimensions`, in the order its shape states them last: `values`, its
    /// buffer bound under its logical shape and strides, stored as `scheme` says, under its
    /// `scales`.
    ///
    /// # Errors
    ///
    /// [`SCALES_WITHIN_CONTRACTION`].
    pub fn of_quantized(
        client: &Client,
        values: TensorBinding,
        scheme: QuantScheme,
        scales: PlainTensor,
        dimensions: [Dimension; 2],
    ) -> Result<Self, Decline> {
        let scaled_along =
            Self::scaled_along(&scheme, dimensions).ok_or(SCALES_WITHIN_CONTRACTION)?;
        Ok(Self {
            values: Self::bytes_of(values, &scheme).ok_or(SCALES_WITHIN_CONTRACTION)?,
            scales: scales.into_contiguous(client),
            scheme,
            scaled_along,
        })
    }

    /// Whether [`of_quantized`](Self::of_quantized) reads a matrix over `dimensions` stored as
    /// `scheme` says, its words along its innermost dimension.
    pub fn reads(scheme: &QuantScheme, dimensions: [Dimension; 2]) -> bool {
        matches!(scheme.value, QuantValue::E4M3 | QuantValue::E5M2)
            && matches!(scheme.store, QuantStore::Native | QuantStore::PackedU32(0))
            && Self::scaled_along(scheme, dimensions).is_some()
    }

    /// The dimension its scales vary along, over one position of it a scale. `None` for a scheme of
    /// another number of levels than one, or whose scales vary otherwise.
    fn scaled_along(scheme: &QuantScheme, dimensions: [Dimension; 2]) -> Option<Option<Dimension>> {
        let level = scheme.finest();
        let [first, second] = dimensions;
        match (scheme.levels().count(), level.size.extents_in::<2>()) {
            (1, [BlockExtent::WholeAxis, BlockExtent::WholeAxis]) => Some(None),
            (1, [BlockExtent::VALUE, BlockExtent::WholeAxis]) if first != K => Some(Some(first)),
            (1, [BlockExtent::WholeAxis, BlockExtent::VALUE]) if second != K => Some(Some(second)),
            _ => None,
        }
    }

    /// The values, at the element they are stored in.
    pub fn values(&self) -> &PlainTensor {
        &self.values
    }

    /// The scales.
    pub fn scales(&self) -> &PlainTensor {
        &self.scales
    }

    /// The scheme it is stored under.
    pub fn scheme(&self) -> QuantScheme {
        self.scheme
    }

    /// This matrix with `values` in place of its own: the same ones, laid otherwise.
    pub fn with_values(self, values: PlainTensor) -> Self {
        Self { values, ..self }
    }

    /// `values`, bound as a tile, under the scales: one operand the kernel factors them out of.
    pub fn under_scales(
        &self,
        launcher: &Launcher,
        blocks: &MatmulBlocks,
        values: TileTensorLaunch,
    ) -> TileTensorLaunch {
        // A dimension no scale groups is one block, the position inside it the whole of it.
        let (position, axis, count) = match self.scaled_along {
            Some(dimension) => {
                let position = blocked(dimension).1;
                let axis = PhysicalAxisMap::of(position);
                (position, axis, blocks.extent(dimension))
            }
            None => (blocked(K).1, PhysicalAxisMap::broadcast(), 1),
        };
        let scales = self.scales.bound_as(Metadata::new([count, 1], [1, 1]));
        let scales = launcher
            .arg(scales, self.scales.elem)
            .gathered(Projection::new(
                &[position],
                &[axis, PhysicalAxisMap::broadcast()],
            ))
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        values.scales(scales.arg())
    }

    /// `values`, stored as `scheme` says, as the 8-bit floats their bytes are. `None` where they are
    /// no 8-bit floats, or packed in words another way than along the innermost dimension of rows.
    fn bytes_of(values: TensorBinding, scheme: &QuantScheme) -> Option<PlainTensor> {
        let element = match scheme.value {
            QuantValue::E4M3 => ElemType::Float(FloatKind::E4M3),
            QuantValue::E5M2 => ElemType::Float(FloatKind::E5M2),
            _ => return None,
        };
        let mut values = values;
        match scheme.store {
            QuantStore::Native => {}
            QuantStore::PackedU32(0) if !values.tiling.is_tiled() => {
                let innermost = values.shape.rank() - 1;
                for dim in 0..innermost {
                    values.strides[dim] *= scheme.num_quants();
                }
            }
            QuantStore::PackedU32(_) | QuantStore::PackedNative(_) => return None,
        }
        Some(PlainTensor::new(values, element))
    }
}
