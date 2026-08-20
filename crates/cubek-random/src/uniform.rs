use cubecl::{prelude::*, std::tensor::ViewMut};

use crate::{
    OutputSlots, PrngBlueprint, PrngState, PrngStrategy, RandomFamily, to_unit_interval_closed_open,
};

use super::{PrngArgs, PrngRuntime, random};

#[derive(CubeLaunch, CubeType)]
pub(crate) struct Uniform {
    lower_bound: f32,
    upper_bound: f32,
}

#[derive(Debug)]
pub(crate) struct UniformFamily;

impl RandomFamily for UniformFamily {
    type Runtime = Uniform;
}

/// The affine map from the unit interval onto `[lower_bound, upper_bound)`.
#[derive(CubeType)]
pub(crate) struct UniformParams<N: Size> {
    scale: Vector<f32, N>,
    offset: Vector<f32, N>,
}

#[cube]
impl PrngRuntime for Uniform {
    type Params<N: Size> = UniformParams<N>;

    fn params<N: Size>(args: &Uniform) -> UniformParams<N> {
        UniformParams::<N> {
            scale: Vector::new(args.upper_bound - args.lower_bound),
            offset: Vector::new(args.lower_bound),
        }
    }

    fn draw<E: Numeric, N: Size>(
        params: &UniformParams<N>,
        state: &mut PrngState<N>,
        slots: &OutputSlots,
        nth: usize,
        output: &mut ViewMut<'_, Vector<E, N>, usize>,
        #[comptime] _blueprint: PrngBlueprint,
    ) {
        let uniform = fma(
            to_unit_interval_closed_open(state.next()),
            params.scale,
            params.offset,
        );

        slots.write(output, nth, Vector::cast_from(uniform));
    }
}

impl PrngArgs for Uniform {
    type Args = Self;

    const VECTORS_PER_DRAW: usize = 1;

    fn args<R: Runtime>(self) -> UniformLaunch<R> {
        UniformLaunch::new(self.lower_bound, self.upper_bound)
    }
}

/// Pseudo-random generator with uniform distribution
pub fn random_uniform<R: Runtime>(
    client: &ComputeClient<R>,
    lower_bound: f32,
    upper_bound: f32,
    out: TensorBinding<R>,
    dtype: ElemType,
) -> Result<(), LaunchError> {
    random_uniform_with_strategy(
        client,
        lower_bound,
        upper_bound,
        out,
        dtype,
        PrngStrategy::Inferred,
    )
}

pub(crate) fn random_uniform_with_strategy<R: Runtime>(
    client: &ComputeClient<R>,
    lower_bound: f32,
    upper_bound: f32,
    out: TensorBinding<R>,
    dtype: ElemType,
    strategy: PrngStrategy,
) -> Result<(), LaunchError> {
    random::<UniformFamily, R>(
        client,
        Uniform {
            lower_bound,
            upper_bound,
        },
        out,
        dtype,
        strategy,
    )
}
