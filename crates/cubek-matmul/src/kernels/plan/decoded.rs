use super::{MatmulCall, MatmulCallInput};
use crate::kernels::PlaneDequantizeMatmulRoutine;
use crate::kernels::tensor::{PackedTensor, StoredTensor};
use cubecl::tile::launch::Refusal;
use cubek_std::plan::{Declined, Step};

/// Every packed input decoded into a float one of the call's element by this crate's decode
/// ([`PlaneDequantizeMatmulRoutine`]), or by the framework's dequantize where the decode declines
/// it. A float input passes through as it is.
pub struct Decoded;

/// Every packed input dequantized by the framework into a float one of the call's element: what a
/// scheme no kernel here reads goes through. A float input passes through as it is.
pub struct FrameworkDequantized;

/// A packed input the framework is not asked about.
const NO_FRAMEWORK_DEQUANTIZE: Refusal = Refusal::Declined {
    why: "the caller states no framework dequantize for this packed input",
};

impl Step<MatmulCall> for Decoded {
    /// # Panics
    ///
    /// Where the decode declines an input stored in tiles, which the framework's dequantize reads
    /// by its strides alone and would answer wrong.
    fn apply(&self, call: &MatmulCall) -> Result<MatmulCall, Declined> {
        let decoded = |input: &MatmulCallInput| match &input.tensor {
            StoredTensor::Plain(_) => Ok(input.clone()),
            StoredTensor::Packed(packed) => decode(call, packed)
                .or_else(|why| match packed.words().is_tiled() {
                    true => panic!(
                        "the decode declines this input stored in tiles ({why}); store it untiled"
                    ),
                    false => framework_dequantized(input),
                })
                .map_err(|why| Declined {
                    plan: self.name(),
                    why: why.to_string(),
                }),
        };
        Ok(MatmulCall {
            lhs: decoded(&call.lhs)?,
            rhs: decoded(&call.rhs)?,
            ..call.clone()
        })
    }

    fn name(&self) -> String {
        "decoded".to_string()
    }
}

impl Step<MatmulCall> for FrameworkDequantized {
    fn apply(&self, call: &MatmulCall) -> Result<MatmulCall, Declined> {
        let dequantized = |input: &MatmulCallInput| match &input.tensor {
            StoredTensor::Plain(_) => Ok(input.clone()),
            StoredTensor::Packed(_) => framework_dequantized(input).map_err(|why| Declined {
                plan: self.name(),
                why: why.to_string(),
            }),
        };
        Ok(MatmulCall {
            lhs: dequantized(&call.lhs)?,
            rhs: dequantized(&call.rhs)?,
            ..call.clone()
        })
    }

    fn name(&self) -> String {
        "framework-dequantized".to_string()
    }
}

/// `packed` decoded by this crate's decode into a float input of the call's element.
fn decode(call: &MatmulCall, packed: &PackedTensor) -> Result<MatmulCallInput, Refusal> {
    PlaneDequantizeMatmulRoutine
        .launch_on(
            &call.hw,
            &call.client,
            packed.clone(),
            call.out,
            &PlaneDequantizeMatmulRoutine::SETTINGS,
        )
        .map(MatmulCallInput::replaced_by)
}

/// `input` dequantized by the framework, as its caller bound it.
fn framework_dequantized(input: &MatmulCallInput) -> Result<MatmulCallInput, Refusal> {
    let dequantize = input
        .framework_dequantize
        .as_ref()
        .ok_or(NO_FRAMEWORK_DEQUANTIZE)?;
    dequantize().map(MatmulCallInput::plain)
}
