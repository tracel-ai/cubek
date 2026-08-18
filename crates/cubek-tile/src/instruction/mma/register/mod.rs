//! The register-resident leaf: a software outer-product GEMM microkernel over memory tiles.
//!
//! - [`base`]: the entry point ([`mma_register_memory`]) and shared accumulator helpers.
//! - [`block`]: the K walk into an `mr × nr` register block ([`contract_block`]), shared by the
//!   memory-backed leaf and the promoted [`RegisterData`](crate::RegisterData).
//! - [`direct`]: the 2-D microkernel ([`mma_register_direct`]) for single-contracted-axis cases.
//! - [`gather`]: the N-D microkernel ([`mma_register_gather`]) for multi-contracted or gathered cases.

mod base;
mod block;
mod direct;
mod gather;

pub(crate) use base::mma_register_memory;
pub(crate) use block::contract_block;
