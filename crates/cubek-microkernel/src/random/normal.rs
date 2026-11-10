use crate::random::{
    Distribution, DistributionArgs, DrawnLines, PrngState, Transcendentals, polynomial,
    to_unit_interval_open,
};
use core::f32::consts::PI;
use cubecl::prelude::*;
use cubecl::tile::Tile;

/// Values of mean `mean` and standard deviation `std`, two lines a draw by the Box-Muller
/// transform.
#[derive(CubeLaunch, CubeType)]
pub struct Normal {
    pub mean: f32,
    pub std: f32,
}

#[derive(CubeType)]
pub struct NormalParams<N: Size> {
    mean: Vector<f32, N>,
    std: Vector<f32, N>,
}

#[cube]
impl Distribution for Normal {
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
        out: &mut Tile,
        lines: &DrawnLines,
        nth: usize,
        #[comptime] transcendentals: Transcendentals,
    ) {
        // Both arms need the open interval: `ln(0)` is `-inf` in hardware and a large
        // finite value in the polynomial arm, so a closed draw would disagree.
        let unit_0 = to_unit_interval_open(state.advance());
        let unit_1 = to_unit_interval_open(state.advance());

        let (log, cosine, sine) = match transcendentals {
            Transcendentals::Instructions => {
                let angle = Vector::new(2.0f32 * PI) * unit_1;
                (unit_0.ln(), angle.cos(), angle.sin())
            }
            Transcendentals::Polynomial => {
                let (cosine, sine) = polynomial::cos_sin_turns(unit_1);
                (polynomial::ln(unit_0), cosine, sine)
            }
        };

        // Box-Muller transform
        let coeff = (log * Vector::new(-2.0f32)).sqrt() * params.std;

        let normal_0 = fma(cosine, coeff, params.mean);
        let normal_1 = fma(sine, coeff, params.mean);

        lines.write(out, 2 * nth, Vector::<E, N>::cast_from(normal_0));
        lines.write(out, 2 * nth + 1, Vector::<E, N>::cast_from(normal_1));
    }
}

impl DistributionArgs for Normal {
    type Args = Self;

    const LINES_PER_DRAW: usize = 2;

    fn args(self) -> NormalLaunch {
        NormalLaunch::new(self.mean, self.std)
    }
}
