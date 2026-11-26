//! The measured plan the selector's tests hand it: a node named for the race it measures, which no
//! test launches.

use super::MatmulAutotune;
use crate::kernels::plan::{MatmulCall, MatmulPlan};
use cubecl::std::tensor::TensorHandle;
use cubek_std::plan::Fastest;

/// The measured plan of `autotune`, as the selector's tests stub it: `fastest(few-rows)`,
/// `fastest(many-rows)` or `fastest(block-scaled)`.
pub(crate) fn stub_measured(autotune: MatmulAutotune) -> MatmulPlan {
    Box::new(Fastest::new(
        autotune.name(),
        |_: &MatmulCall| -> TensorHandle { unreachable!("the selector's tests launch no plan") },
    ))
}
