#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "kernels")]
pub mod dequantize;

#[cfg(feature = "kernels")]
pub mod dequantize_tiled;

#[cfg(feature = "kernels")]
pub mod quantize;

#[cfg(feature = "kernels")]
pub mod layout;

#[cfg(feature = "kernels")]
pub mod scale;

pub use cubecl_common::quant::scheme;

#[cfg(feature = "kernels")]
pub(crate) mod utils {
    use crate::scheme::{QuantScheme, QuantStore, QuantValue};
    use cubecl::features::TypeUsage;
    use cubecl::ir::{ElemType, UIntKind};
    use cubecl::prelude::*;

    /// Panic when the scheme stores natively but the device cannot convert the storage
    /// element (`i8` under the hood for the 8-bit values and fp8).
    pub(crate) fn check_i8_supported<R: Runtime>(client: &ComputeClient<R>, scheme: &QuantScheme) {
        match scheme {
            QuantScheme {
                value: QuantValue::Q8F | QuantValue::Q8S | QuantValue::E4M3 | QuantValue::E5M2,
                store: QuantStore::Native,
                ..
            }
            | QuantScheme {
                value: QuantValue::E2M1,
                store: QuantStore::PackedNative(_),
                ..
            } if !i8::supported_uses(client).contains(TypeUsage::Conversion) => {
                panic!(
                    "{:?} is not supported for native quantization",
                    scheme.value
                );
            }
            _ => {}
        }
    }

    pub(crate) fn check_block_size_compat(scheme: &QuantScheme, div: usize) {
        // Validate block size compatibility
        if let Some(block_size) = scheme.block_size() {
            let block_size = *block_size.as_slice().last().unwrap() as usize;
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
