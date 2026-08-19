//! The hardware contraction leaves: a CMMA/WMMA fragment and a manual MMA fragment, each
//! executing its own instruction. Impl blocks only; the dispatch that reaches one of them is the
//! verb's, in `ops/matmul/lower.rs`.

mod cmma;
mod manual;
