use crate::Hardware;
use cubecl::{CubeCount, CubeDim};
use cubecl::tile::launch::{Refusal, spawnable};

/// The half of a routine's `prepare` the blueprint does not hold as a field:
/// the cube the levels imply on this device, the grid the extents imply, and
/// the operand widths as the launch binds them.
///
/// `cube_dim` is read off the blueprint's levels and the device's plane width,
/// and cubecl folds it into the compilation key itself; `cube_count` changes
/// per call and compiles nothing. `V` is the widths the blueprint decided
/// (the widths it binds at), `()` for the families that bind every operand
/// scalar or read no width.
///
/// Not `Copy`/`Eq`: `CubeCount` has a `Dynamic` variant carrying a handle.
#[derive(Clone, Debug)]
pub struct LaunchSettings<V = ()> {
    pub cube_dim: CubeDim,
    pub cube_count: CubeCount,
    pub vectors: V,
}

impl<V> LaunchSettings<V> {
    /// Units this launch spawns per cube.
    pub fn units_per_cube(&self) -> u32 {
        self.cube_dim.num_elems()
    }

    /// Reject a cube or a grid the device cannot spawn ([`spawnable`]). Call this from `prepare`
    /// rather than trusting the sizing arithmetic.
    ///
    /// # Errors
    ///
    /// The cap the cube or the grid passes.
    pub fn validate(self, hw: &Hardware) -> Result<Self, Refusal> {
        spawnable(&hw.properties.hardware, self.cube_dim, &self.cube_count)?;
        Ok(self)
    }
}
