//! The walk: the recurrence a layer of a linear recurrence family carries per value head, over its
//! positions `t`:
//!
//! ```text
//! S_t = γ_t·S_{t-1} + k_t ⊗ β_t·w_t        o_t = S_tᵀ·q_t
//! ```
//!
//! `w_t = v_t − (γ_t·S_{t-1})ᵀ·k_t` under the delta rule and `w_t = v_t` under the additive rule,
//! which also adds `D·v_t` to `o_t`. `S` is `[key, value]`, `γ_t = exp(decay_t)` and `β_t` (the
//! strength) are one scalar per head and position, `D` (the skip) one scalar per head, and `q`, `k`
//! are read from the key head the value head is grouped under. Column `j` of the state evolves
//! from `k`, `q` and `v_t[j]` alone, so the work item is a run of a head's
//! value columns, all of their key rows, walked through every position; owners of different runs
//! share nothing. An owner holds its columns' state on-chip for the whole walk: the state is read
//! once and written once, however long the walk.

pub mod cube;
mod item;
mod operand;
mod rule;

pub use item::*;
pub use operand::*;
pub use rule::*;
