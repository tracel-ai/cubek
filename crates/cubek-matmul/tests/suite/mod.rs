#![allow(missing_docs)]

pub mod layered;
pub mod naive;

mod cpu_reference;

pub use cpu_reference::assert_result;
