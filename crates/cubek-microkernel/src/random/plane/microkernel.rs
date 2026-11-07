use super::base::PlaneRandomMicrokernel;
use crate::random::{Distribution, DrawnLines, PrngState, Seeds};
use cubecl::tile::Tile;
use cubecl::{self as cubecl, cube, prelude::*};

/// One plane's run: `out`, the plane's window of the output, drawn from `D` with the lanes
/// interleaved, each lane off a generator seeded for its position in the launch.
#[cube]
pub fn plane_random_microkernel<D: Distribution, E: Numeric, N: Size>(
    out: &mut Tile,
    args: &D::Args,
    seeds: &Seeds,
    #[comptime] microkernel: PlaneRandomMicrokernel,
) {
    let mut state = PrngState::<N>::seeded(ABSOLUTE_POS, seeds);
    let params = D::params::<N>(args);
    let draws = microkernel.values_per_lane / N::value() / comptime!(D::LINES_PER_DRAW);
    let lines = DrawnLines::new(UNIT_POS_X as usize, CUBE_DIM_X as usize);

    #[unroll(draws <= 8)]
    for nth in 0..draws {
        D::draw::<E, N>(
            &params,
            &mut state,
            out,
            &lines,
            nth,
            microkernel.transcendentals,
        );
    }
}
