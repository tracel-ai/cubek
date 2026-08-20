use cubecl::{CubeType, Runtime, prelude::*, std::tensor::ViewMut};

use crate::{OutputSlots, PrngState, PrngStrategy, RandomFamily};

use super::{PrngArgs, PrngRuntime, random, to_unit_interval_closed_open};

#[derive(CubeLaunch, CubeType)]
pub(crate) struct Bernoulli {
    pub(crate) probability: f32,
}

#[derive(Debug)]
pub(crate) struct BernoulliFamily;

impl RandomFamily for BernoulliFamily {
    type Runtime = Bernoulli;
}

#[cube]
impl PrngRuntime for Bernoulli {
    fn draw<E: Numeric, N: Size>(
        args: &Bernoulli,
        state: &mut PrngState<N>,
        slots: &OutputSlots,
        nth: usize,
        output: &mut ViewMut<'_, Vector<E, N>, usize>,
    ) {
        let probability = Vector::new(args.probability);
        let uniform = to_unit_interval_closed_open(state.next());

        output.write_checked(
            slots.at(nth),
            Vector::cast_from(uniform.less_than(&probability)),
        );
    }
}

impl PrngArgs for Bernoulli {
    type Args = Self;

    const VECTORS_PER_DRAW: usize = 1;

    fn args<R: Runtime>(self) -> BernoulliLaunch<R> {
        BernoulliLaunch::new(self.probability)
    }
}

/// Pseudo-random generator with bernoulli distribution
pub fn random_bernoulli<R: Runtime>(
    client: &ComputeClient<R>,
    probability: f32,
    out: TensorBinding<R>,
    dtype: ElemType,
) -> Result<(), LaunchError> {
    random::<BernoulliFamily, R>(
        client,
        Bernoulli { probability },
        out,
        dtype,
        PrngStrategy::Inferred,
    )
}
