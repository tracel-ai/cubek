//! Re-export shim. The tiling-layout trait + impls now live in
//! `cubek_std::tile::variants::stage`; this file keeps the
//! `cubek_matmul::components::stage::memory::layout` path working for
//! existing callers. Deleted in PR 6 once consumers update their imports.

pub use cubek_std::tile::{
    ColMajorTilingOrder, ContiguousTilingLayout, NoTilingLayout, OrderedTilingOrder,
    RowMajorTilingOrder, StridedTilingLayout, TilingLayout, TilingLayoutConfig, TilingLayoutEnum,
    TilingOrder, TilingOrderEnum, TilingValidation, TmaTilingLayout, TmaTilingOrder,
};
