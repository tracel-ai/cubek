use core::num::NonZeroUsize;
use cubek_std::matmul_axes::{M, N};
use cubek_tile::Axis;

/// How a cube's planes are laid out over its box of the output: `along_m` planes along `m` by
/// `along_n` along `n`, each plane owning its own share.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatmulPlaneLayout {
    pub along_m: NonZeroUsize,
    pub along_n: NonZeroUsize,
}

impl MatmulPlaneLayout {
    /// `along_m × along_n` planes.
    pub const fn new(along_m: usize, along_n: usize) -> Self {
        Self {
            along_m: NonZeroUsize::new(along_m).expect("a layout holds at least one plane along m"),
            along_n: NonZeroUsize::new(along_n).expect("a layout holds at least one plane along n"),
        }
    }

    /// The planes one cube holds.
    pub fn count(&self) -> usize {
        self.along_m.get() * self.along_n.get()
    }

    /// The counts a partitioning's plane level distributes.
    pub fn counts(&self) -> [(Axis, usize); 2] {
        [(M, self.along_m.get()), (N, self.along_n.get())]
    }
}

impl core::fmt::Display for MatmulPlaneLayout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "planes={}x{}", self.along_m, self.along_n)
    }
}
