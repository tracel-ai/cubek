use super::PlainTensor;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::quant::scheme::{QuantScheme, QuantValue};
#[cfg(doc)]
use cubecl::tile::kind::Field;
use cubecl::tile::launch::Decline;

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

/// What a product declines of a row-scaled lhs against a packed rhs whose values are not all
/// values of the lhs's element ([`Field::values_exact_in`]): an 8-bit integer's, or a float
/// format of more mantissa bits or a wider exponent, which the decode into it would round.
pub const ROW_SCALED_RHS_VALUES_INEXACT: Decline = Decline::Unserved {
    why: "a row-scaled lhs is read against packed values its element holds exactly",
};

/// What a product declines of a row-scaled lhs against a packed rhs whose block scales are no
/// powers of two (`ue8m0`): a `ue4m3`'s or a float's mantissa times a code is no value of the
/// lhs's 8-bit element.
pub const ROW_SCALED_RHS_SCALES_INEXACT: Decline = Decline::Unserved {
    why: "a row-scaled lhs is read against packed values under power-of-two block scales",
};

/// What a product declines of a packed rhs with a per-row level against anything but a
/// row-scaled lhs: its row scales come out of the sum beside the lhs's, which no other read
/// verifies.
pub const ROW_SCALED_WEIGHT_UNSERVED: Decline = Decline::Unserved {
    why: "a packed rhs with a per-row level is read against a row-scaled lhs alone",
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
    /// `scales`, one contiguous `f32` a row.
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
        let Some(&k) = shape.as_slice().last() else {
            return Err(ROW_SCALED_UNREAD);
        };
        let rows = shape.num_elements() / k.max(1);
        let one_a_row = scales.shape().num_elements() == rows
            && scales.elem == ElemType::Float(FloatKind::F32)
            && scales.is_contiguous();
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::quant::scheme::ScaleDtype;

    const ROWS: usize = 4;
    const DEPTH: usize = 64;
    const F32: ElemType = ElemType::Float(FloatKind::F32);
    const E4M3: ElemType = ElemType::Float(FloatKind::E4M3);

    fn zeros(shape: [usize; 2], elem: ElemType) -> PlainTensor {
        PlainTensor::zeros(&cubek_test_utils::client(), &shape, [shape[1], 1], elem)
    }

    fn new(values: PlainTensor, scales: PlainTensor, scheme: QuantScheme) -> Option<Decline> {
        RowScaledTensor::new(values, scales, scheme).err()
    }

    /// `e4m3` values in dense rows under one contiguous `f32` a row are FP8-row.
    #[test]
    fn e4m3_rows_under_a_scale_a_row_are_row_scaled() {
        let (values, scales) = (zeros([ROWS, DEPTH], E4M3), zeros([ROWS, 1], F32));
        assert_eq!(new(values, scales, QuantScheme::fp8_per_row()), None);
    }

    /// A per-tensor level in place of the row's is refused.
    #[test]
    fn a_per_tensor_level_is_refused() {
        let (values, scales) = (zeros([ROWS, DEPTH], E4M3), zeros([ROWS, 1], F32));
        let scheme = QuantScheme::fp8_per_row().per_tensor(ScaleDtype::F32);
        assert_eq!(new(values, scales, scheme), Some(ROW_SCALED_UNREAD));
    }

    /// One scale short of the rows is refused.
    #[test]
    fn a_scale_short_is_refused() {
        let (values, scales) = (zeros([ROWS, DEPTH], E4M3), zeros([ROWS - 1, 1], F32));
        assert_eq!(
            new(values, scales, QuantScheme::fp8_per_row()),
            Some(ROW_SCALED_UNREAD)
        );
    }

    /// Values of another element than the scheme's are refused.
    #[test]
    fn values_of_another_element_are_refused() {
        let (values, scales) = (zeros([ROWS, DEPTH], F32), zeros([ROWS, 1], F32));
        assert_eq!(
            new(values, scales, QuantScheme::fp8_per_row()),
            Some(ROW_SCALED_UNREAD)
        );
    }

    /// Scales of another element than `f32` are refused.
    #[test]
    fn scales_of_another_element_are_refused() {
        let (values, scales) = (zeros([ROWS, DEPTH], E4M3), zeros([ROWS, 1], E4M3));
        assert_eq!(
            new(values, scales, QuantScheme::fp8_per_row()),
            Some(ROW_SCALED_UNREAD)
        );
    }
}
