//! `Strategy::Auto`: the root's own dispatch, which belongs to neither family.
//! A directory module because a bare `tests/*.rs` becomes its own test binary,
//! and could not then reach `crate::harness`.

mod base;
