#[cfg(feature = "extended")]
pub mod extended;
pub mod normal;

pub(crate) mod launcher;

mod reference;

pub(crate) use reference::assert_result;
