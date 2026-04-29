// `RowWise`, `LocalTile`, `LocalTileLayout`, `InnerLayout`, `UnitTile`, `UnitTileLayout`
// now live in cubek-std. This module retains the orphan-rule-bound trait impls
// (`FragmentMask`, `SoftmaxLayout`) tying those types to cubek-attention's softmax traits,
// plus a re-export so existing `crate::components::tile::pipeline::X` paths keep resolving.

mod local_tile;
mod rowwise;
mod unit_tile;

pub use cubek_std::tile::{
    InnerLayout, LocalTile, LocalTileLayout, RowWise, UnitTile, UnitTileLayout,
};
