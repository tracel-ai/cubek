use super::{MatmulCall, MatmulCallInput};
use crate::kernels::PlaneDequantizeMatmulRoutine;
use crate::kernels::tensor::{PackedTensor, PlainTensor, StoredTensor};
use cubecl::tile::launch::Refusal;
use cubek_std::matmul_axes::{K, M, N};
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

impl Decoded {
    /// `input`, one of `call`'s, decoded where it is packed: what a candidate that decodes one
    /// input and reads the other as stored decodes.
    ///
    /// # Errors
    ///
    /// Where the decode declines the input and the caller states no framework dequantize for it.
    ///
    /// # Panics
    ///
    /// Where the decode declines an input stored in tiles, which the framework's dequantize reads
    /// by its strides alone and would answer wrong.
    pub fn input(
        &self,
        call: &MatmulCall,
        input: &MatmulCallInput,
    ) -> Result<MatmulCallInput, Declined> {
        match &input.tensor {
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
        }
    }
}

impl Step<MatmulCall> for Decoded {
    /// # Panics
    ///
    /// As [`Decoded::input`].
    fn apply(&self, call: &MatmulCall) -> Result<MatmulCall, Declined> {
        Ok(MatmulCall {
            lhs: self.input(call, &call.lhs)?,
            rhs: self.input(call, &call.rhs)?,
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
///
/// The decode names a packed matrix's axes `K` and `N`: an `[m, k]` input's rows are its `N`, and
/// the `[k, rows]` it writes is read back transposed.
fn decode(call: &MatmulCall, packed: &PackedTensor) -> Result<MatmulCallInput, Refusal> {
    let matrix = packed.clone().renamed(M, N);
    let laid_along_k = matrix.axes()[0] == K;
    let handle = PlaneDequantizeMatmulRoutine.launch_on(
        &call.hw,
        &call.client,
        matrix,
        call.out,
        &PlaneDequantizeMatmulRoutine::SETTINGS,
    )?;
    let decoded = PlainTensor::of_handle(handle);
    Ok(MatmulCallInput::plain(match laid_along_k {
        true => decoded,
        false => decoded.transposed(),
    }))
}

/// `input` dequantized by the framework, as its caller bound it.
fn framework_dequantized(input: &MatmulCallInput) -> Result<MatmulCallInput, Refusal> {
    let dequantize = input
        .framework_dequantize
        .as_ref()
        .ok_or(NO_FRAMEWORK_DEQUANTIZE)?;
    dequantize().map(MatmulCallInput::plain)
}
