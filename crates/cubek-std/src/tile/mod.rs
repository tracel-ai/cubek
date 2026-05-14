//! Tile module.

mod base;
mod event;
mod mask;
mod ops;
mod scheduler;
mod scope;
mod variants;

pub use base::*;
pub use event::*;
pub use mask::*;
pub use ops::*;
pub use scheduler::*;
pub use scope::*;
pub use variants::*;

// Stage-memory data types reachable via `cubek_std::tile::` for back-compat
// with callers that pre-date the `tile/variants/stage/{memory,layout}.rs` →
// `stage/stage_memory/{memory,layout}.rs` relocation. Prefer the new
// `cubek_std::stage::` paths in fresh code.
pub use crate::stage::{
    ColMajorTilingOrder, ContiguousTilingLayout, NoTilingLayout, OrderedTilingOrder,
    RowMajorTilingOrder, StridedStageMemory, StridedTilingLayout, TilingLayout, TilingLayoutEnum,
    TilingOrder, TilingOrderEnum, TilingValidation, TmaTilingLayout, TmaTilingOrder,
};
