#![allow(missing_docs)]

pub mod basic;
#[cfg(feature = "extended")]
pub mod extended;
#[cfg(feature = "full")]
pub mod full;

pub mod launcher_strategy;
mod reference;
