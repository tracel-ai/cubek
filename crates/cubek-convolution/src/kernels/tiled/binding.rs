use cubecl::tile::{Launcher, TileArgLaunch};

/// The arguments of one launch, its operands bound under its launcher. Not `Debug`: the tile DSL's
/// `Launcher` is not.
pub struct ConvBinding {
    pub launcher: Launcher,
    pub input: TileArgLaunch,
    pub rhs: TileArgLaunch,
    pub out: TileArgLaunch,
}
