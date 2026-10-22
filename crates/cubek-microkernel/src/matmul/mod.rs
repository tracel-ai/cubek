//! The matmul microkernels, in [`dense`]: the one product, whichever of its inputs are packed. It
//! holds its microkernels in a folder per scope, named by algorithm, beside what only they share.
//! Beside them, what every matmul-shaped microkernel reads, attention's included: the
//! [`MatmulWorkItem`] and its axes and layouts, the [`MatmulMicrokernel`] trait, the operands it
//! names and how a packed one is stored ([`stored`], [`tile_order`]), the [`MatmulPartitioning`] every product
//! places its work with, stated over a packed product's block-split space too ([`MatmulSpace`]),
//! the [`MatmulStep`], the least box every load of a product fills whole, and the [`staged_walk`]
//! a cube takes through `K`, holding its stages in the [`staging`](crate::staging) ring.

mod axes;
mod buffers;
pub mod dense;
mod layout;
mod matmul_microkernel;
mod operand;
mod partitioning;
pub mod staged_walk;
mod step;
pub mod stored;
pub mod tile_order;
mod work_item;

pub use axes::*;
pub use buffers::*;
pub use layout::*;
pub use matmul_microkernel::*;
pub use operand::*;
pub use partitioning::*;
pub use step::*;
pub use work_item::*;
