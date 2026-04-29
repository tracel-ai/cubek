use cubecl;
use cubecl::{prelude::*, std::tensor::layout::Coords2d};

#[cube]
/// Minimal mask abstraction used by row-wise tile operations.
/// Returns `true` when the element at `local_pos` should be treated as masked
/// (i.e. driven to -inf by `Tile::scale_and_mask`).
pub trait Mask: CubeType {
    fn should_mask(&self, local_pos: Coords2d) -> bool;
}
