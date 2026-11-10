//! The RMS norm: its routines and their autotune ([`kernels`]), and its halves for deferring it
//! into neighbouring products ([`deferred`]).

pub mod deferred;
pub mod kernels;
