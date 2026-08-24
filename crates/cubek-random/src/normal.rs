use cubecl::{prelude::*, std::tensor::ViewMut};
use std::f32::consts::PI;

use super::{PrngArgs, PrngRuntime, random};

use crate::{
    OutputSlots, PrngBlueprint, PrngState, PrngStrategy, RandomFamily, polynomial,
    to_unit_interval_open,
};

#[derive(CubeLaunch, CubeType)]
pub(crate) struct Normal {
    mean: f32,
    std: f32,
}

#[derive(Debug)]
pub(crate) struct NormalFamily;

impl RandomFamily for NormalFamily {
    type Runtime = Normal;
}

#[derive(CubeType)]
pub(crate) struct NormalParams<N: Size> {
    mean: Vector<f32, N>,
    std: Vector<f32, N>,
}

#[cube]
impl PrngRuntime for Normal {
    type Params<N: Size> = NormalParams<N>;

    fn params<N: Size>(args: &Normal) -> NormalParams<N> {
        NormalParams::<N> {
            mean: Vector::new(args.mean),
            std: Vector::new(args.std),
        }
    }

    fn draw<E: Numeric, N: Size>(
        params: &NormalParams<N>,
        state: &mut PrngState<N>,
        slots: &OutputSlots,
        nth: usize,
        output: &mut ViewMut<'_, Vector<E, N>, usize>,
        #[comptime] blueprint: PrngBlueprint,
    ) {
        // Both arms need the open interval: `ln(0)` is `-inf` in hardware and a large
        // finite value in the polynomial arm, so a closed draw would disagree.
        let unit_0 = to_unit_interval_open(state.next());
        let unit_1 = to_unit_interval_open(state.next());

        // A CPU has no vector `ln`, `cos`, or `sin`, only one libm call per lane.
        let (log, cosine, sine) = match comptime!(blueprint) {
            PrngBlueprint::Interleaved => {
                let angle = Vector::new(2.0f32 * PI) * unit_1;
                (unit_0.ln(), angle.cos(), angle.sin())
            }
            PrngBlueprint::Blocked => {
                let (cosine, sine) = polynomial::cos_sin_turns(unit_1);
                (polynomial::ln(unit_0), cosine, sine)
            }
        };

        // Box-Muller transform
        let coeff = (log * Vector::new(-2.0f32)).sqrt() * params.std;

        let normal_0 = fma(cosine, coeff, params.mean);
        let normal_1 = fma(sine, coeff, params.mean);

        slots.write(output, 2 * nth, Vector::cast_from(normal_0));
        slots.write(output, 2 * nth + 1, Vector::cast_from(normal_1));
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
    random_normal_with_strategy(client, mean, std, out, dtype, PrngStrategy::Inferred)
}

pub(crate) fn random_normal_with_strategy<R: Runtime>(
    client: &ComputeClient<R>,
    mean: f32,
    std: f32,
    out: TensorBinding<R>,
    dtype: ElemType,
    strategy: PrngStrategy,
) -> Result<(), LaunchError> {
    random::<NormalFamily, R>(client, Normal { mean, std }, out, dtype, strategy)
}
