use crate::matmul::{MatmulAxes, MatmulSpace, MatmulStage, MatmulWorkItem};
use cubecl::tile::Space;
use cubecl::tile::launch::Refusal;

/// The box a cube microkernel is handed: what it stages and walks of its routine's placement, and
/// what of that placement its refusals read. The placement itself is the routine's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulCubeBox {
    /// One batch of the output's `m × n` box, and the whole of `k`, or the cube's run of it where
    /// several cubes share the box.
    pub cube: Space,
    /// The box one step of its walk through `K` stages.
    pub stage: MatmulStage,
    /// The innermost box: what one instruction, or one unit's block, covers.
    pub leaf: Space,
    /// The space the box is stated over.
    pub space: MatmulSpace,
    /// Whether the boxes overhang the output: a box the output's edge cuts short, which a write
    /// that cannot be masked would reach past.
    pub overhangs_output: bool,
    /// Whether the problem has batch axes, a cube handed one batch along each.
    pub batched: bool,
}

impl MatmulCubeBox {
    /// What a cube microkernel declines an item on other axes with: it places and stages in
    /// matmul's own `M`, `N` and `K`.
    pub const OTHER_AXES: Refusal = Refusal::Declined {
        why: "a cube microkernel works in matmul's own M, N and K",
    };

    /// Whether `item` lies along matmul's own axes, which a cube's box is stated in.
    ///
    /// # Errors
    ///
    /// [`OTHER_AXES`](Self::OTHER_AXES), for an item on other axes.
    pub fn admits(item: &MatmulWorkItem) -> Result<(), Refusal> {
        match item.axes == MatmulAxes::MNK {
            true => Ok(()),
            false => Err(Self::OTHER_AXES),
        }
    }
}
