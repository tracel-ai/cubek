use crate::matmul::MatmulAxes;
use cubecl::tile::{Axis, Space};
use cubek_std::matmul_axes::{K, KB, M, MI, N, NI};

/// The space a product is stated over, which a placement's walk, its stage and its output's axes
/// are read in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulSpace {
    /// Matmul's own `M`, `N` and `K`, and the batches beside them.
    Plain,
    /// A packed product's: each axis as the blocks its rhs's scales make of it and the position
    /// inside one ([`MatmulBlocks`](crate::matmul::MatmulBlocks)), every extent
    /// compiled in.
    BlockSplit,
}

impl MatmulSpace {
    /// The space `space` is: block-split where it carries `K`'s blocks.
    pub fn of(space: &Space) -> Self {
        match space.contains(KB) {
            true => Self::BlockSplit,
            false => Self::Plain,
        }
    }

    /// The axes a stage of a product over this space lies along: matmul's own, or those inside a
    /// block.
    pub fn stage_axes(self) -> MatmulAxes {
        match self {
            Self::Plain => MatmulAxes::MNK,
            Self::BlockSplit => MatmulAxes::INSIDE_BLOCKS,
        }
    }

    /// The axis a cube's walk steps through `K` along: `K`, or its blocks.
    pub fn walked(self) -> Axis {
        match self {
            Self::Plain => K,
            Self::BlockSplit => KB,
        }
    }

    /// The axis the product's rows lie along inside a block: `M`, or the rows inside its one
    /// block.
    pub fn rows_axis(self) -> Axis {
        match self {
            Self::Plain => M,
            Self::BlockSplit => MI,
        }
    }

    /// The axes the output's rows and columns are cut short along at the problem's edge.
    pub fn output_axes(self) -> [Axis; 2] {
        match self {
            Self::Plain => [M, N],
            Self::BlockSplit => [MI, NI],
        }
    }
}
