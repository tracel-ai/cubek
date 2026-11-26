use crate::base::random;
use crate::kernels::RandomStrategy;
use cubecl::prelude::*;
use cubecl::tile::launch::Decline;
use cubek_microkernel::random::Normal;

/// `out` filled with values of mean `mean` and standard deviation `std`, drawn off `seed`.
///
/// # Errors
///
/// Why no random kernel runs on this device, and [`Decline::Broken`] where a kernel's binding
/// finds false what its routine prepared.
pub fn random_normal(
    client: &Client,
    mean: f32,
    std: f32,
    out: TensorBinding,
    dtype: ElemType,
    seed: u64,
) -> Result<(), Decline> {
    random_normal_with_strategy(
        client,
        mean,
        std,
        out,
        dtype,
        RandomStrategy::Selected,
        seed,
    )
}

pub(crate) fn random_normal_with_strategy(
    client: &Client,
    mean: f32,
    std: f32,
    out: TensorBinding,
    dtype: ElemType,
    strategy: RandomStrategy,
    seed: u64,
) -> Result<(), Decline> {
    random(client, Normal { mean, std }, out, dtype, strategy, seed)
}
