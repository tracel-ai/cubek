#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "kernels")]
pub mod dequantize;

#[cfg(feature = "kernels")]
pub mod quantize;

#[cfg(feature = "kernels")]
pub mod layout;

pub use cubecl_common::quant::scheme;

#[cfg(feature = "kernels")]
pub(crate) mod utils {
    use crate::scheme::{QuantLevel, QuantScheme};

    pub(crate) fn check_block_size_compat(scheme: &QuantScheme, div: usize) {
        // Validate block size compatibility
        if let QuantScheme {
            level: QuantLevel::Block(block_size),
            ..
        } = scheme
        {
            let block_size = *block_size.as_slice().last().unwrap() as usize;
            assert!(
                block_size.is_multiple_of(div),
                "Block size must be divisible by {div}, got block_size={block_size}"
            );
        }
    }

    pub(crate) fn quant_size_bytes(scheme: &QuantScheme) -> usize {
        debug_assert!(
            scheme.size_bits_stored() % 8 == 0,
            "size_bits_stored must be divisible by 8, got {} -> {}",
            scheme.size_bits_stored(),
            scheme.size_bits_stored() / 8
        );
        scheme.size_bits_stored() / 8
    }
}
