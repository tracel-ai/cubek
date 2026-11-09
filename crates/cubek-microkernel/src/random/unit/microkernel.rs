use super::base::UnitRandomMicrokernel;
use crate::random::{Distribution, DrawnLines, PrngState, Seeds};
use cubecl::tile::Tile;
use cubecl::{self as cubecl, cube, prelude::*};

/// One unit's run: every line of `out`, the unit's window of the output, drawn from `D` in turn
/// off a generator seeded for the unit's position in the launch.
#[cube]
pub fn unit_random_microkernel<D: Distribution, E: Numeric, N: Size>(
    out: &mut Tile,
    args: &D::Args,
    seeds: &Seeds,
    #[comptime] microkernel: UnitRandomMicrokernel,
) {
    let mut state = PrngState::<N>::seeded(ABSOLUTE_POS, seeds);
    let params = D::params::<N>(args);
    let draws = out.flat::<E, N>().shape() / comptime!(D::LINES_PER_DRAW);
    let lines = DrawnLines::new(0, 1);
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
