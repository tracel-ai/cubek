use crate::random::{
    Distribution, DistributionArgs, DrawnLines, PrngState, Transcendentals,
    to_unit_interval_closed_open,
};
use cubecl::prelude::*;
use cubecl::tile::Tile;

/// `1` with probability `probability`, `0` otherwise.
#[derive(CubeLaunch, CubeType)]
pub struct Bernoulli {
    pub probability: f32,
}

#[derive(CubeType)]
pub struct BernoulliParams<N: Size> {
    probability: Vector<f32, N>,
}

#[cube]
impl Distribution for Bernoulli {
    type Params<N: Size> = BernoulliParams<N>;

    fn params<N: Size>(args: &Bernoulli) -> BernoulliParams<N> {
        BernoulliParams::<N> {
            probability: Vector::new(args.probability),
        }
    }

    fn draw<E: Numeric, N: Size>(
        params: &BernoulliParams<N>,
        state: &mut PrngState<N>,
        out: &mut Tile,
        lines: &DrawnLines,
        nth: usize,
        #[comptime] _transcendentals: Transcendentals,
    ) {
        let uniform = to_unit_interval_closed_open(state.advance());

        lines.write(
            out,
            nth,
            Vector::<E, N>::cast_from(uniform.less_than(&params.probability)),
        );
    }
}

impl DistributionArgs for Bernoulli {
    type Args = Self;

    const LINES_PER_DRAW: usize = 1;

    fn args(self) -> BernoulliLaunch {
        BernoulliLaunch::new(self.probability)
    }
}
