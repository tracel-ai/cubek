#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "kernels")]
pub mod dequantize;

#[cfg(feature = "kernels")]
pub mod dequantize_tiled;

#[cfg(feature = "kernels")]
pub mod quantize;

#[cfg(feature = "kernels")]
mod tile_tensor;
#[cfg(feature = "kernels")]
pub use tile_tensor::{quantized_arg, quantized_arg_shifted};

#[cfg(feature = "kernels")]
pub mod layout;

#[cfg(feature = "kernels")]
pub mod scale;

pub use cubecl_common::quant::scheme;

#[cfg(feature = "kernels")]
pub(crate) mod utils {
    use crate::scheme::{BlockExtent, QuantScheme, QuantStore};
    use cubecl::ir::{ElemType, UIntKind};

    pub(crate) fn check_block_size_compat(scheme: &QuantScheme, div: usize) {
        // Validate block size compatibility
        if let BlockExtent::Values(block_size) = scheme.finest().size.extents[0] {
            let block_size = block_size.get() as usize;
            assert!(
                block_size.is_multiple_of(div),
                "Block size must be divisible by {div}, got block_size={block_size}"
            );
        }
    }

    pub(crate) fn packed_storage_elem(scheme: &QuantScheme) -> ElemType {
        match scheme.store {
            QuantStore::PackedU32(_) => ElemType::UInt(UIntKind::U32),
            store => panic!("Unsupported packed storage {store:?}"),
        }
    }
}
