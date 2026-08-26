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

// Stage vocabulary a tile reader needs, re-exported so a tile-side caller does not
// reach into `stage` for it.
pub use crate::multi_level::stage::{
    ColMajorTilingOrder, ContiguousTilingLayout, NoTilingLayout, OrderedTilingOrder,
    RowMajorTilingOrder, StridedStageMemory, StridedTilingLayout, TilingLayout, TilingLayoutEnum,
    TilingOrder, TilingOrderEnum, TilingValidation, TmaTilingLayout, TmaTilingOrder,
};
