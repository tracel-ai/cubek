use cubek_std::matmul_axes::{K, KI, M, MI, N, NI};
use cubek_tile::Axis;

/// The axes the product's rows, columns and contraction lie along in the caller's space: matmul's
/// own [`MNK`](Self::MNK), or another's, so a microkernel tiles attention's score product
/// (`QP · S` over `D`) as it tiles a matmul.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatmulAxes {
    pub m: Axis,
    pub n: Axis,
    pub k: Axis,
}

impl MatmulAxes {
    pub const MNK: Self = Self { m: M, n: N, k: K };

    /// The axes inside a block the product's rows, columns and contraction lie along in a packed
    /// rhs's block-split space ([`MatmulBlocks`](super::MatmulBlocks)): what a microkernel of the
    /// float product tiles, handed a packed product's blocks.
    pub const INSIDE_BLOCKS: Self = Self {
        m: MI,
        n: NI,
        k: KI,
    };

    /// The caller's axis for `dim`.
    pub fn of(self, dim: MatmulDim) -> Axis {
        match dim {
            MatmulDim::M => self.m,
            MatmulDim::N => self.n,
            MatmulDim::K => self.k,
        }
    }

    /// The dimension of the product `axis` is: none for an axis that is not one of these.
    pub fn dim(self, axis: Axis) -> Option<MatmulDim> {
        MatmulDim::ALL.into_iter().find(|&dim| self.of(dim) == axis)
    }

    /// `extents` stated by dimension, restated along the caller's axes.
    pub fn name<const L: usize>(self, extents: [(MatmulDim, usize); L]) -> [(Axis, usize); L] {
        extents.map(|(dim, extent)| (self.of(dim), extent))
    }
}

/// One of a product's three dimensions: its rows (`M`), its columns (`N`) or its contraction
/// (`K`), named apart from the axis a caller lays it along ([`MatmulAxes`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulDim {
    M,
    N,
    K,
}

impl MatmulDim {
    pub const ALL: [Self; 3] = [Self::M, Self::N, Self::K];
}
