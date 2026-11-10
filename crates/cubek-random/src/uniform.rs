use crate::base::random;
use crate::kernels::RandomStrategy;
use cubecl::prelude::*;
use cubek_microkernel::random::Uniform;

/// `out` filled with values spread evenly over `[lower_bound, upper_bound)`, drawn off `seed`.
pub fn random_uniform(
    client: &Client,
    lower_bound: f32,
    upper_bound: f32,
    out: TensorBinding,
    dtype: ElemType,
    seed: u64,
) -> Result<(), LaunchError> {
    random_uniform_with_strategy(
        client,
        lower_bound,
        upper_bound,
        out,
        dtype,
        RandomStrategy::Selected,
        seed,
    )
}

pub(crate) fn random_uniform_with_strategy(
    client: &Client,
    lower_bound: f32,
    upper_bound: f32,
    out: TensorBinding,
    dtype: ElemType,
    strategy: RandomStrategy,
    seed: u64,
) -> Result<(), LaunchError> {
    random(
        client,
        Uniform {
            lower_bound,
            upper_bound,
        },
        out,
        dtype,
        strategy,
        seed,
    )
}
