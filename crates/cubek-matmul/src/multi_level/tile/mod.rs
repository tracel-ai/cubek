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

// Back-compat re-exports; prefer `crate::multi_level::stage::` in new code.
pub use crate::multi_level::stage::{
    ColMajorTilingOrder, ContiguousTilingLayout, NoTilingLayout, OrderedTilingOrder,
    RowMajorTilingOrder, StridedStageMemory, StridedTilingLayout, TilingLayout, TilingLayoutEnum,
    TilingOrder, TilingOrderEnum, TilingValidation, TmaTilingLayout, TmaTilingOrder,
};
