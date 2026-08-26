//! Re-export of the generic write-event types. They live in
//! [`crate::multi_level::tile`] (alongside
//! [`StageEvent`](crate::multi_level::tile::StageEvent)) since they describe a
//! tile-domain protocol rather than a matmul-specific one.

pub use crate::multi_level::tile::{WriteEvent, WriteEventExpand, WriteEventListener};
