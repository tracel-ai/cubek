use crate::base::random;
use crate::kernels::RandomStrategy;
use cubecl::prelude::*;
use cubecl::tile::launch::Decline;
use cubek_microkernel::random::Bernoulli;

/// `out` filled with `1` at probability `probability` and `0` otherwise, drawn off `seed`.
///
/// # Errors
///
/// As [`random_normal`](crate::random_normal).
pub fn random_bernoulli(
    client: &Client,
    probability: f32,
    out: TensorBinding,
    dtype: ElemType,
    seed: u64,
) -> Result<(), Decline> {
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
) -> Result<(), Decline> {
    random(
        client,
        Bernoulli { probability },
        out,
        dtype,
        strategy,
        seed,
    )
}
