//! FlashAttention forward pass: definitions, launch entry points, routine
//! selection, and the CPU reference + benchmark catalogue.

pub mod definition;
pub mod launch;
pub mod routines;

pub use launch::flash_attention_forward_lse;
