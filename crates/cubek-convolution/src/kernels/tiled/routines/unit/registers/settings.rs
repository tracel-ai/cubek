use core::num::NonZeroUsize;
use cubecl::tile::space::CubeOrder;
use cubek_matmul::kernels::cube_order_name;
use cubek_microkernel::matmul::cube::MatmulPlaneLayout;

/// The block of the output a unit owns in registers, stated apart from the problem: the block of
/// each form is derived from the problem when a launch is prepared, so a setting suits every form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnitBlock {
    /// The largest block the device's registers hold for the form the problem's input gives it
    /// ([`BlockForm::largest_block`](super::BlockForm::largest_block)).
    Largest,
    /// One cell.
    OneCell,
}

impl core::fmt::Display for UnitBlock {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Largest => write!(f, "largest"),
            Self::OneCell => write!(f, "one-cell"),
        }
    }
}

/// How a unit's block is chosen and how the planes that own the blocks are laid out. The block's
/// depth is the device's ([`BlockDepth`](super::BlockDepth)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitRegistersConvSettings {
    pub block: UnitBlock,
    /// How the planes one cube groups are laid out over the output.
    pub planes: MatmulPlaneLayout,
    /// Blocks along the columns one plane's units take in turns, however many units it has.
    pub blocks_per_plane: NonZeroUsize,
    /// The order the cubes take their boxes of planes in, where a cube holds a box of them.
    pub cube_order: CubeOrder,
}

impl core::fmt::Display for UnitRegistersConvSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "unit-registers-conv block={} {} blocks_per_plane={}{}",
            self.block,
            self.planes,
            self.blocks_per_plane,
            cube_order_name(self.cube_order)
        )
    }
}
