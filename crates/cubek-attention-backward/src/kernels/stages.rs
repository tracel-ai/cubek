//! What a walk stages in shared memory, and the grid that holds it.

use cubecl::tile::launch::Refusal;
use cubecl::tile::{Partitioning, StageStorage};
use cubek_std::Hardware;
use cubek_std::microkernel::{StageRing, StagedOperand};
use cubek_std::routine::{CubeResources, Grid};

/// The stages one walk holds in shared memory: `stage` held as `ring` says, filled by
/// `filling_planes` planes, `copies` times a cube, each copy with `window_bytes` beside it and
/// `body_bytes` declared in the walk's body, held once a body the ring expands
/// ([`StageRing::bodies`]).
///
/// A plane that walks its own stages fills them alone and its cube holds one copy a plane; a cube
/// that walks them above its planes fills them with all of them and holds one.
pub(crate) struct WalkStages {
    pub stage: [StagedOperand; 2],
    pub ring: StageRing,
    pub filling_planes: usize,
    pub copies: usize,
    pub window_bytes: usize,
    pub body_bytes: usize,
}

impl WalkStages {
    /// The grid of the walk `partitioning` places, and the ring as this device holds its stages
    /// ([`StageRing::resolve`]).
    ///
    /// # Errors
    ///
    /// A device that commits to no plane width, or a cube the grid declines.
    pub(crate) fn grid(
        &self,
        hw: &Hardware,
        partitioning: &Partitioning,
    ) -> Result<(Grid, StageRing), Refusal> {
        let plane_width = hw.plane_dim()?;
        let ring = self
            .ring
            .resolve(hw, &self.stage, self.filling_planes * plane_width as usize);
        let stage_bytes = self
            .stage
            .iter()
            .map(|operand| operand.bytes(hw, &StageStorage::Strided))
            .sum();
        let cube = CubeResources {
            plane_width,
            shared_memory: self.copies
                * (ring.bytes(stage_bytes) + self.window_bytes + ring.bodies() * self.body_bytes),
        };
        Ok((Grid::new(hw, partitioning, cube)?, ring))
    }
}
