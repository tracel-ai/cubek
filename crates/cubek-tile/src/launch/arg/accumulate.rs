//! An output several instances accumulate into, as a single launch argument.

use cubecl::prelude::*;

use crate::*;

/// An output several instances accumulate into, as a single launch argument: [`TileArg`]'s twin
/// for a destination whose writes add, not replace. `Atomic<E>` carries no served width, so
/// [`tile`](Self::tile) states it; the drain adds each line's scalars one atomic at a time.
///
/// **The buffer arrives holding the monoid's identity.** A cell here belongs to several instances
/// and none of them may seed it, so the seeding happens once at the launch. Nothing can check it:
/// the destination cannot read.
#[derive(CubeType, CubeLaunch)]
pub struct AccumulateArg<'a, E: Numeric> {
    pub tensor: &'a Tensor<Atomic<E>>,
    #[cube(comptime)]
    pub spec: TileSpec,
}

#[cube]
impl<'a, E: Numeric> AccumulateArg<'a, E> {
    /// Serve the output as a [`Tile`] that accumulates into it at width `V`. [`TileArg::tile`]'s
    /// twin, and the same call: the kernel's one `space` projected onto this operand's `spec`
    /// axes.
    pub fn tile<V: Size>(&self, #[comptime] space: Partitioning) -> Tile<E> {
        Tile::<E>::of_atomic_accumulate::<V>(
            self.tensor,
            comptime!(space.space().clone()),
            comptime!(self.spec.clone()),
        )
        .under(comptime!(space.levels().to_vec()))
    }
}
