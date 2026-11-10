use crate::random::{
    Distribution, DistributionArgs, DrawnLines, PrngState, Transcendentals,
    to_unit_interval_closed_open,
};
use cubecl::prelude::*;
use cubecl::tile::Tile;

/// Values spread evenly over `[lower_bound, upper_bound)`.
#[derive(CubeLaunch, CubeType)]
pub struct Uniform {
    pub lower_bound: f32,
    pub upper_bound: f32,
}

#[derive(CubeType)]
pub struct UniformParams<N: Size> {
    scale: Vector<f32, N>,
    offset: Vector<f32, N>,
}

#[cube]
impl Distribution for Uniform {
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
        out: &mut Tile,
        lines: &DrawnLines,
        nth: usize,
        #[comptime] _transcendentals: Transcendentals,
    ) {
        let uniform = fma(
            to_unit_interval_closed_open(state.advance()),
            params.scale,
            params.offset,
        );

        lines.write(out, nth, Vector::<E, N>::cast_from(uniform));
    }
}

impl DistributionArgs for Uniform {
    type Args = Self;

    const LINES_PER_DRAW: usize = 1;

    fn args(self) -> UniformLaunch {
        UniformLaunch::new(self.lower_bound, self.upper_bound)
    }
}
