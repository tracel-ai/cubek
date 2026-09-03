//! One operand's data in the kernel: the [`Tile`] itself ([`base`]), the backing stores it
//! dispatches on (one file each), the views a leaf reads it through ([`view`]), and what a
//! quantized store carries ([`quant`]). The launch surface (specs, deliveries, builder)
//! lives in `physical/`; a kernel's first line is [`Tile::of`] on a plain tensor.

mod atomic;
mod base;
mod cmma;
mod mem;
mod mma;
mod packing;
mod plane;
mod procedural;
mod quant;
mod register;
mod tma;
mod view;

pub use base::*;
pub use cmma::*;
pub use mem::*;
pub use mma::*;
pub use packing::*;
pub use plane::*;
pub use procedural::*;
pub use quant::*;
pub use register::*;
pub use tma::*;
pub use view::*;
