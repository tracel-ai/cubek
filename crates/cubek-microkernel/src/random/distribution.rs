use crate::random::PrngState;
use cubecl::prelude::*;
use cubecl::tile::Tile;
use cubek_std::Hardware;

/// A distribution's host side: its parameters as one launch argument, and how many output lines
/// one draw writes.
pub trait DistributionArgs: Send + Sync + 'static {
    type Args: LaunchArg;

    /// Output lines one draw writes: the Box-Muller transform emits a pair, the other
    /// distributions one each.
    const LINES_PER_DRAW: usize;

    fn args(self) -> <Self::Args as LaunchArg>::RuntimeArg;
}

/// How a draw evaluates the transcendentals a distribution needs: the device's instructions, or
/// polynomials where the device has no vector math to issue them as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Transcendentals {
    /// `ln`, `cos` and `sin` as the device's instructions: a GPU's.
    Instructions,
    /// [`polynomial`](crate::random::polynomial)s: a CPU's, whose JIT scalarizes every
    /// transcendental into a libm call per component.
    Polynomial,
}

impl Transcendentals {
    /// The way `hw` evaluates transcendentals cheaply: a CPU runtime's JIT has no vector math
    /// behind them.
    pub fn of(hw: &Hardware) -> Self {
        match hw.properties.hardware.num_cpu_cores {
            Some(_) => Self::Polynomial,
            None => Self::Instructions,
        }
    }
}

/// A distribution a unit draws from: its parameters, splat to the line once, and how one draw
/// turns the generator's next values into output lines.
#[cube]
pub trait Distribution: DistributionArgs {
    /// The distribution's parameters, broadcast to the line.
    type Params<N: Size>: CubeType;

    /// Broadcast the parameters once, so a unit's draws share one set of splats.
    fn params<N: Size>(args: &Self::Args) -> Self::Params<N>;

    /// The `nth` draw of `state`, written to the lines of `out` that `lines` names.
    fn draw<E: Numeric, N: Size>(
        params: &Self::Params<N>,
        state: &mut PrngState<N>,
        out: &mut Tile,
        lines: &DrawnLines,
        nth: usize,
        #[comptime] transcendentals: Transcendentals,
    );
}

/// The lines of a tile a unit's draws land on, as its first one and the distance to the next: a
/// unit's own run steps by one, a lane interleaved with its plane by the plane's width.
#[derive(CubeType)]
pub struct DrawnLines {
    first: usize,
    stride: usize,
}

#[cube]
impl DrawnLines {
    pub fn new(first: usize, stride: usize) -> DrawnLines {
        DrawnLines { first, stride }
    }

    /// Write `value` to the `nth` of these lines of `out`, masked where it lies past the tile.
    pub fn write<E: Numeric, N: Size>(&self, out: &mut Tile, nth: usize, value: Vector<E, N>) {
        let mut lines = out.flat_mut::<E, N>();
        lines.write(self.first + nth * self.stride, value);
    }
}
