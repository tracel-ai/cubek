use crate::matmul::{MatmulDimensions, MatmulDimensionsKind, MatmulWorkItem};
use cubecl::tile::launch::Decline;
use cubecl::tile::{Dimension, Space};

/// The space a cube microkernel is handed: the part of the problem one cube covers, and what of
/// its routine's placement its declines read. The placement itself is the routine's; how the cube
/// walks its space is the microkernel's own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulCubeSpace {
    /// One batch of the output's `m × n` box, and the whole of `k`, or the cube's run of it where
    /// several cubes share the box.
    pub space: Space,
    /// The innermost box: what one instruction, or one unit's block, covers.
    pub leaf: Space,
    /// The dimensions the problem, and so this space, is stated over.
    pub dimensions_kind: MatmulDimensionsKind,
    /// Whether the boxes overhang the output: a box the output's edge cuts short, which a write
    /// that cannot be masked would reach past.
    pub overhangs_output: bool,
    /// Whether the problem has batch dimensions, a cube handed one batch along each.
    pub batched: bool,
}

impl MatmulCubeSpace {
    /// Its extent along `dimension`.
    pub fn extent(&self, dimension: Dimension) -> usize {
        self.space.extent(dimension)
    }

    /// What a cube microkernel declines an item on other dimensions with: it places and stages in
    /// matmul's own `M`, `N` and `K`.
    pub(crate) const OTHER_DIMENSIONS: Decline = Decline::Unserved {
        why: "a cube microkernel works in matmul's own M, N and K",
    };

    /// Whether `item` lies along matmul's own dimensions, which a cube's box is stated in.
    ///
    /// # Errors
    ///
    /// `OTHER_DIMENSIONS`, for an item on other dimensions.
    pub fn admits(item: &MatmulWorkItem) -> Result<(), Decline> {
        match item.dimensions == MatmulDimensions::MNK {
            true => Ok(()),
            false => Err(Self::OTHER_DIMENSIONS),
        }
    }
}
