use cubecl::{prelude::*, std::tensor::ViewMut};
use std::f32::consts::PI;

use super::{PrngArgs, PrngRuntime, random};

use crate::{OutputSlots, PrngState, PrngStrategy, RandomFamily, to_unit_interval_open};

#[derive(CubeLaunch, CubeType)]
pub(crate) struct Normal {
    pub(crate) mean: f32,
    pub(crate) std: f32,
}

#[derive(Debug)]
pub(crate) struct NormalFamily;

impl RandomFamily for NormalFamily {
    type Runtime = Normal;
}

#[cube]
impl PrngRuntime for Normal {
    fn draw<E: Numeric, N: Size>(
        args: &Normal,
        state: &mut PrngState<N>,
        slots: &OutputSlots,
        nth: usize,
        output: &mut ViewMut<'_, Vector<E, N>, usize>,
    ) {
        let mean = Vector::new(args.mean);

        let unit_0 = to_unit_interval_open(state.next());
        let unit_1 = to_unit_interval_open(state.next());

        // Box-Muller transform
        let coeff = (unit_0.ln() * Vector::new(-2.0f32)).sqrt() * Vector::new(args.std);
        let trigo_arg = Vector::new(2.0f32 * PI) * unit_1;

        let normal_0 = trigo_arg.cos() * coeff + mean;
        let normal_1 = trigo_arg.sin() * coeff + mean;

        output.write_checked(slots.at(2 * nth), Vector::cast_from(normal_0));
        output.write_checked(slots.at(2 * nth + 1), Vector::cast_from(normal_1));
    }
}

impl PrngArgs for Normal {
    type Args = Self;

    const VECTORS_PER_DRAW: usize = 2;

    fn args<R: Runtime>(self) -> NormalLaunch<R> {
        NormalLaunch::new(self.mean, self.std)
    }
}

/// Pseudo-random generator with normal distribution
pub fn random_normal<R: Runtime>(
    client: &ComputeClient<R>,
    mean: f32,
    std: f32,
    out: TensorBinding<R>,
    dtype: ElemType,
) -> Result<(), LaunchError> {
    random::<NormalFamily, R>(
        client,
        Normal { mean, std },
        out,
        dtype,
        PrngStrategy::Inferred,
    )
}
