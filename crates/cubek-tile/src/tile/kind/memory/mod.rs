//! The addressable kind ([`Memory`], gmem and smem): what one is ([`base`]), how it is built
//! from a launched operand ([`global`]) or derived as a shared-memory stage ([`stage`]), how it
//! is touched ([`access`]), the window it looks through ([`window`]), the tensor-map source
//! ([`tma`]) and the views a leaf reads it through ([`view`]).

mod access;
mod base;
mod factor;
mod global;
mod stage;
mod tma;
mod view;
mod window;

pub use base::*;
pub(crate) use factor::*;
pub use global::*;
pub(crate) use stage::*;
pub use tma::*;
pub use view::*;
pub use window::*;
