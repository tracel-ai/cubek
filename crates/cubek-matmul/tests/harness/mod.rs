//! Launcher plumbing and problem builders shared by both architectures. Carries no
//! test of its own: `run` takes a closure, so it is architecture-agnostic.

mod broadcast;
mod launcher;
mod problems;

pub(crate) use broadcast::*;
pub(crate) use launcher::*;
pub(crate) use problems::*;

pub(crate) use cubek_matmul::eval::cpu_reference::assert_result;
