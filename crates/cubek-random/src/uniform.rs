use cubecl::{prelude::*, std::tensor::ViewMut};

use crate::{OutputSlots, PrngState, PrngStrategy, RandomFamily, to_unit_interval_closed_open};

use super::{PrngArgs, PrngRuntime, random};

#[derive(CubeLaunch, CubeType)]
pub(crate) struct Uniform {
    pub(crate) lower_bound: f32,
    pub(crate) upper_bound: f32,
}

#[derive(Debug)]
pub(crate) struct UniformFamily;

impl RandomFamily for UniformFamily {
    type Runtime = Uniform;
}

#[cube]
impl PrngRuntime for Uniform {
    fn draw<E: Numeric, N: Size>(
        args: &Uniform,
        state: &mut PrngState<N>,
        slots: &OutputSlots,
        nth: usize,
        output: &mut ViewMut<'_, Vector<E, N>, usize>,
    ) {
        let scale = Vector::new(args.upper_bound - args.lower_bound);
        let offset = Vector::new(args.lower_bound);

        let uniform = to_unit_interval_closed_open(state.next()) * scale + offset;

        output.write_checked(slots.at(nth), Vector::cast_from(uniform));
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
    random::<UniformFamily, R>(
        client,
        Uniform {
            lower_bound,
            upper_bound,
        },
        out,
        dtype,
        PrngStrategy::Inferred,
    )
}
