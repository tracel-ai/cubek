use cubecl::tile::{Launcher, TileTensorLaunch};

/// The arguments of one launch, its operands bound under its launcher. Its `Debug` names the type
/// only: the tile DSL's `Launcher` has none.
pub struct ConvBinding {
    pub launcher: Launcher,
    pub input: TileTensorLaunch,
    pub rhs: TileTensorLaunch,
    pub out: TileTensorLaunch,
    pub bias: Option<TileTensorLaunch>,
}

impl core::fmt::Debug for ConvBinding {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ConvBinding").finish_non_exhaustive()
    }
}
