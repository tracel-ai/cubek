//! The measured plan the selector's tests hand it: a node named for the query positions it
//! measures, which no test launches.

use super::problem::QueryPositions;
use crate::kernels::plan::{AttentionCall, AttentionPlan};
use cubecl::std::tensor::TensorHandle;
use cubek_std::plan::Fastest;

/// The measured choice of a call of `positions` query positions, as the selector's tests stub it:
/// `fastest(decode)` at one position, `fastest(prefill)` at many.
pub(crate) fn stub_measured(positions: QueryPositions) -> AttentionPlan {
    let measured = match positions {
        QueryPositions::One => "decode",
        QueryPositions::Many => "prefill",
    };
    Box::new(Fastest::new(
        measured,
        |_: &AttentionCall| -> TensorHandle { unreachable!("the selector's tests launch no plan") },
    ))
}
