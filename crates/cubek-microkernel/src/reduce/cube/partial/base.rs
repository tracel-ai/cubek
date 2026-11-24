use super::microkernel::cube_partial_reduce_microkernel;
use crate::reduce::ReduceAccumulator;
use crate::reduce::cube::CubeStagedReduceMicrokernel;
use cubecl::tile::Tile;
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// A part of a row reduced by one cube into one of the row's partials: the cube reduces its window
/// as [`CubeStagedReduceMicrokernel`] does one line of rows wide, and its first unit stores the
/// carrier ([`UnitPartialStoreMicrokernel`](crate::reduce::write::UnitPartialStoreMicrokernel)).
///
/// It is part of the hypercube microkernel, whose cubes split a row, and has no launch of its own.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubePartialReduceMicrokernel {
    /// The microkernel the cube reduces its part with.
    #[cube(comptime)]
    cube: CubeStagedReduceMicrokernel,
}

impl CubePartialReduceMicrokernel {
    /// A cube reducing its part with `cube`.
    pub fn new(cube: CubeStagedReduceMicrokernel) -> Self {
        Self { cube }
    }

    pub fn cube(&self) -> CubeStagedReduceMicrokernel {
        self.cube
    }
}

#[cube]
impl CubePartialReduceMicrokernel {
    /// The lines of `input`, the line at `l` read as line `first_line + l` of its row, reduced into
    /// partial `part` of `partials` and `partial_indices` (`cube_partial_reduce_microkernel`).
    ///
    /// # Panics
    ///
    /// Where `partial_indices` is there and its carrier keeps no index, or the other way around.
    pub fn reduce<In: Numeric, Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
        &self,
        input: &Tile,
        partials: &mut Tile,
        partial_indices: &ComptimeOption<Tile>,
        part: usize,
        first_line: usize,
    ) {
        cube_partial_reduce_microkernel::<In, Acc, N, C>(
            input,
            partials,
            partial_indices,
            part,
            first_line,
            comptime!(CubePartialReduceMicrokernel { cube: self.cube }),
        )
    }
}
