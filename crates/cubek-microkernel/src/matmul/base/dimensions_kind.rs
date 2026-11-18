use crate::matmul::MatmulDimensions;
use cubecl::tile::{Dimension, Space};
use cubek_std::matmul_dimensions::{K, KB, M, MI, N, NI};

/// The dimensions a product is stated over, which a placement's walk, its stage and its output's
/// dimensions are read in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulDimensionsKind {
    /// Matmul's own `M`, `N` and `K`, and the batches beside them.
    Plain,
    /// A packed product's: each dimension as the blocks its rhs's scales make of it and the
    /// position inside one ([`MatmulBlocks`](crate::matmul::MatmulBlocks)), every extent compiled
    /// in.
    BlockSplit,
}

impl MatmulDimensionsKind {
    /// The space `space` is: block-split where it carries `K`'s blocks.
    pub fn of(space: &Space) -> Self {
        match space.contains(KB) {
            true => Self::BlockSplit,
            false => Self::Plain,
        }
    }

    /// The dimensions a stage of a product over this space lies along: matmul's own, or those
    /// inside a block.
    pub fn stage_dimensions(self) -> MatmulDimensions {
        match self {
            Self::Plain => MatmulDimensions::MNK,
            Self::BlockSplit => MatmulDimensions::INSIDE_BLOCKS,
        }
    }

    /// The dimension a cube's walk steps through `K` along: `K`, or its blocks.
    pub fn walked(self) -> Dimension {
        match self {
            Self::Plain => K,
            Self::BlockSplit => KB,
        }
    }

    /// The dimension the product's rows lie along inside a block: `M`, or the rows inside its one
    /// block.
    pub fn rows_dimension(self) -> Dimension {
        match self {
            Self::Plain => M,
            Self::BlockSplit => MI,
        }
    }

    /// The dimensions the output's rows and columns are cut short along at the problem's edge.
    pub fn output_dimensions(self) -> [Dimension; 2] {
        match self {
            Self::Plain => [M, N],
            Self::BlockSplit => [MI, NI],
        }
    }
}
