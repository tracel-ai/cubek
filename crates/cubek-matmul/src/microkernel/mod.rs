//! The matmul microkernels, a folder per problem they solve: [`dense`], the float product, and
//! [`q_rhs`], the product against a quantized weight. Each holds its microkernels in a folder
//! per scope, named by algorithm, beside what only they share. Beside them, what every
//! matmul-shaped microkernel reads, attention's included: the [`MatmulWorkItem`] and its axes and
//! layouts, the [`MatmulMicrokernel`] trait, the operands it names, the [`MatmulPartitioning`]
//! every product places its work with, stated over a packed product's block-split space too
//! ([`MatmulSpace`]), the [`MatmulStep`], the least box every load of a product fills whole, the
//! [`staged_walk`] a cube takes through `K`, and the [`staging`] ring every family's staged walk
//! holds its stages in.
//!
//! A microkernel is what one owner of a work item runs, and its scope is the owner: a unit, a
//! plane, a cube. Nothing here is launched: a routine is a whole kernel written around one
//! microkernel, and the launch is its own. The framework every family's microkernels share is
//! cubek-std's: the [`Microkernel`](cubek_std::microkernel::Microkernel) trait, the estimate, a
//! cube's planes, and the routine ([`cubek_std::routine`]).

mod axes;
mod buffers;
pub mod dense;
mod layout;
mod matmul_microkernel;
mod operand;
mod partitioning;
pub mod q_rhs;
pub mod staged_walk;
pub mod staging;
mod step;
mod work_item;

pub use axes::*;
pub use buffers::*;
pub use layout::*;
pub use matmul_microkernel::*;
pub use operand::*;
pub use partitioning::*;
pub use step::*;
pub use work_item::*;
