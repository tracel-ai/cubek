//! Routine selection for the FlashAttention backward kernels.
//!
//! Placeholder for tiled routine families. The per-row kernels in `launch`
//! are dispatched directly without a routine abstraction; a tiled variant
//! would be selected here.
