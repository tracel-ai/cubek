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

#[derive(CubeType)]
pub(crate) struct BernoulliParams<N: Size> {
    probability: Vector<f32, N>,
}

#[cube]
impl PrngRuntime for Bernoulli {
    type Params<N: Size> = BernoulliParams<N>;

    fn params<N: Size>(args: &Bernoulli) -> BernoulliParams<N> {
        BernoulliParams::<N> {
            probability: Vector::new(args.probability),
        }
    }

    fn draw<E: Numeric, N: Size>(
        params: &BernoulliParams<N>,
        state: &mut PrngState<N>,
        slots: &OutputSlots,
        nth: usize,
        output: &mut ViewMut<'_, Vector<E, N>, usize>,
    ) {
        let uniform = to_unit_interval_closed_open(state.next());

        slots.write(
            output,
            nth,
            Vector::cast_from(uniform.less_than(&params.probability)),
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
