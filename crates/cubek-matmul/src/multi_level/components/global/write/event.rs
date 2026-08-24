//! Re-export of the generic write-event types from cubek-std. The
//! types themselves now live in [`cubek_std::tile`] (alongside
//! [`StageEvent`](cubek_std::tile::StageEvent)) since they describe a
//! tile-domain protocol rather than a matmul-specific one.

pub use cubek_std::tile::{WriteEvent, WriteEventExpand, WriteEventListener};
