use crate::base::random;
use crate::kernels::RandomStrategy;
use cubecl::prelude::*;
use cubek_microkernel::random::Bernoulli;

/// `out` filled with `1` at probability `probability` and `0` otherwise, drawn off `seed`.
pub fn random_bernoulli(
    client: &Client,
    probability: f32,
    out: TensorBinding,
    dtype: ElemType,
    seed: u64,
) -> Result<(), LaunchError> {
    random_bernoulli_with_strategy(
        client,
        probability,
        out,
        dtype,
        RandomStrategy::Selected,
        seed,
    )
}

pub(crate) fn random_bernoulli_with_strategy(
    client: &Client,
    probability: f32,
    out: TensorBinding,
    dtype: ElemType,
    strategy: RandomStrategy,
    seed: u64,
) -> Result<(), LaunchError> {
    random(
        client,
        Bernoulli { probability },
        out,
        dtype,
        strategy,
        seed,
    )
}
