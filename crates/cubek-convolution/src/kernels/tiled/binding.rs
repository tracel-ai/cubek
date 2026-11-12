use cubecl::tile::{Launcher, TileTensorLaunch};

/// The arguments of one launch, its operands bound under its launcher. Not `Debug`: the tile DSL's
/// `Launcher` is not.
pub struct ConvBinding {
    pub launcher: Launcher,
    pub input: TileTensorLaunch,
    pub rhs: TileTensorLaunch,
    pub out: TileTensorLaunch,
}
