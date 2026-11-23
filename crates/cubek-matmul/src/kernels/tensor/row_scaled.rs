use super::PlainTensor;
use cubecl::ir::{ElemType, FloatKind, UIntKind};
use cubecl::quant::scheme::{QuantScheme, QuantValue, ScaleDtype};
use cubecl::tile::launch::Decline;
use cubecl::tile::{Dimension, Launcher, TileTensorLaunch, kind::Field};
use cubecl::tile::{Projection, layout::PhysicalAxisMap};
use cubek_std::routine::ADMITTED_PLAN_RUNS;

/// What a row-scaled tensor declines: a scheme of other than one per-row level over 8-bit float
/// values, values not of its element in dense rows, or scales other than one `f32` a row.
pub const ROW_SCALED_UNREAD: Decline = Decline::Unserved {
    why: "a row-scaled operand is 8-bit float values in dense rows under one f32 scale a row, \
          its scheme's only level",
};

/// What a product declines of a row-scaled operand anywhere but as the lhs beside a packed rhs:
/// the one place its row scales come out of the sum as the output rows' factor.
pub const ROW_SCALED_UNSERVED: Decline = Decline::Unserved {
    why: "a row-scaled operand is read as the lhs of a product against a packed rhs",
};

/// A matrix of 8-bit float values, each row under one scale over the whole row, its scheme's
/// only level ([`QuantScheme::per_row`]): `e4m3` activations quantized a token at a time
/// (FP8-row, [`QuantScheme::fp8_per_row`]). A kernel reads its values as they are, an input of
/// their element, and its scales come out of the sum as a factor of each output row, applied once.
#[derive(Debug, Clone)]
pub struct RowScaledTensor {
    values: PlainTensor,
    scales: PlainTensor,
    scheme: QuantScheme,
}

impl RowScaledTensor {
    /// `values`, `[rows, k]` of the element `scheme`'s values are, each row under its scale of
    /// `scales`, one `f32` a row.
    ///
    /// # Errors
    ///
    /// [`ROW_SCALED_UNREAD`].
    pub fn new(
        values: PlainTensor,
        scales: PlainTensor,
        scheme: QuantScheme,
    ) -> Result<Self, Decline> {
        let element = match scheme.value {
            QuantValue::E4M3 => ElemType::Float(FloatKind::E4M3),
            QuantValue::E5M2 => ElemType::Float(FloatKind::E5M2),
            _ => return Err(ROW_SCALED_UNREAD),
        };
        let one_level = scheme.num_levels() == 1 && scheme.row_scale().is_some();
        let shape = values.shape();
        let rows = shape.num_elements() / shape[shape.rank() - 1].max(1);
        let one_a_row =
            scales.shape().num_elements() == rows && scales.elem == ElemType::Float(FloatKind::F32);
        if !one_level || values.elem != element || !values.is_contiguous() || !one_a_row {
            return Err(ROW_SCALED_UNREAD);
        }
        Ok(Self {
            values,
            scales,
            scheme,
        })
    }

    /// Its values, of their own element.
    pub fn values(&self) -> &PlainTensor {
        &self.values
    }

    /// Its scales, one a row.
    pub fn scales(&self) -> &PlainTensor {
        &self.scales
    }

    /// The scheme its values and scales are stored under.
    pub fn scheme(&self) -> QuantScheme {
        self.scheme
    }

    /// Its scales under `launcher`, one a row of the dimension `rows` its values' rows are bound
    /// over, broadcast along every other: the level a kernel takes out of its sum.
    pub fn bind_scales(&self, launcher: &Launcher, rows: Dimension) -> TileTensorLaunch {
        let scales = self.scales.as_rows(self.scales.shape().num_elements(), 1);
        launcher
            // Bound at its words, as a packed operand's scale level is: an `f32` a word.
            .arg(scales.binding, ElemType::UInt(UIntKind::U32))
            .gathered(Projection::new(
                &[rows],
                &[PhysicalAxisMap::of(rows), PhysicalAxisMap::broadcast()],
            ))
            .packed(Field::of_scale(ScaleDtype::F32))
            .build()
            .expect(ADMITTED_PLAN_RUNS)
            .arg()
    }
}
