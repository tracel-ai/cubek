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
pub(crate) mod per_tensor;

pub use cubecl_common::quant::scheme;

#[cfg(feature = "kernels")]
pub(crate) mod utils {
    use crate::scheme::{QuantScheme, QuantStore};
    use cubecl::ir::{ElemType, UIntKind};

    pub(crate) fn check_block_size_compat(scheme: &QuantScheme, div: usize) {
        // Validate block size compatibility
        if let Some(block_size) = scheme.level.block_size() {
            let block_size = *block_size.as_slice().last().unwrap() as usize;
            assert!(
                block_size.is_multiple_of(div),
                "Block size must be divisible by {div}, got block_size={block_size}"
            );
        }
    }

    /// The element type of the per-tensor scale, defaulting to `f32` for the levels that have
    /// none, where the kernel never builds the view and the type goes unused.
    pub(crate) fn global_dtype(scheme: &QuantScheme) -> ElemType {
        scheme
            .level
            .global_param()
            .map(ElemType::from_quant_param)
            .unwrap_or(ElemType::Float(cubecl::ir::FloatKind::F32))
    }

    /// The scheme is what decides whether a per-tensor scale exists; a binding that disagrees with
    /// it silently drops the scale or applies one that should not be there.
    pub(crate) fn check_global_bindings(scheme: &QuantScheme, provided: bool) {
        let expected = scheme.level.global_param().is_some();
        assert_eq!(
            provided,
            expected,
            "global binding was {}, but {:?} {} a per-tensor scale",
            if provided { "provided" } else { "omitted" },
            scheme.level,
            if expected {
                "requires"
            } else {
                "does not take"
            }
        );
    }

    pub(crate) fn packed_storage_elem(scheme: &QuantScheme) -> ElemType {
        match scheme.store {
            QuantStore::PackedU32(_) => ElemType::UInt(UIntKind::U32),
            store => panic!("Unsupported packed storage {store:?}"),
        }
    }
}
