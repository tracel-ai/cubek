//! Re-export shim. The stage event listener now lives in
//! `cubek_std::tile::variants::stage::event`; this file keeps the existing
//! `cubek_matmul::components::stage` path working for callers. Deleted in
//! PR 6 once consumers update their imports.

pub use cubek_std::tile::{NoEvent, StageEvent, StageEventListener};
