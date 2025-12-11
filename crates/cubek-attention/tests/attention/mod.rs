pub(crate) mod launcher;

mod reference;
mod utils;

pub(crate) use reference::assert_result;
pub(crate) use utils::tiling_scheme_ops;

mod unit {
    use cubek_attention::{Selection, Strategy, kernels::SharedAttentionSettings};
    fn strategy(selection: Selection<SharedAttentionSettings>) -> Strategy {
        Strategy::Unit(selection)
    }

    const TILE_SIZE: cubek_attention::components::AttentionTileSize =
        cubek_attention::components::AttentionTileSize {
            seq_q: 4,
            seq_kv: 4,
            head_dim: 4,
            val_dim: 4,
        };

    const STAGE_Q_BASE: u32 = 32;

    mod f16_ty {
        use super::*;
        use cubecl::frontend::CubePrimitive;

        fn global_dtypes() -> AttentionStorageTypes {
            AttentionStorageTypes::from_single_dtype(half::f16::as_type_native_unchecked())
        }

        include!("tests.rs");
    }

    mod f32_ty {
        use super::*;
        use cubecl::frontend::CubePrimitive;

        fn global_dtypes() -> AttentionStorageTypes {
            AttentionStorageTypes::from_single_dtype(f32::as_type_native_unchecked())
        }

        include!("tests.rs");
    }
}

mod blackbox_accelerated {
    use cubek_attention::{Selection, Strategy, kernels::SharedAttentionSettings};
    fn strategy(selection: Selection<SharedAttentionSettings>) -> Strategy {
        Strategy::BlackboxAccelerated(selection)
    }

    #[cfg(target_os = "macos")]
    const TILE_SIZE: cubek_attention::components::AttentionTileSize =
        cubek_attention::components::AttentionTileSize {
            seq_q: 8,
            seq_kv: 8,
            head_dim: 8,
            val_dim: 8,
        };
    #[cfg(not(target_os = "macos"))]
    const TILE_SIZE: cubek_attention::components::AttentionTileSize =
        cubek_attention::components::AttentionTileSize {
            seq_q: 16,
            seq_kv: 16,
            head_dim: 16,
            val_dim: 16,
        };

    const STAGE_Q_BASE: u32 = 1;

    mod f16_ty {
        use super::*;
        use cubecl::frontend::CubePrimitive;

        fn global_dtypes() -> AttentionStorageTypes {
            AttentionStorageTypes::from_single_dtype(half::f16::as_type_native_unchecked())
        }

        include!("tests.rs");
    }

    mod f32_ty {
        use super::*;
        use cubecl::frontend::CubePrimitive;

        fn global_dtypes() -> AttentionStorageTypes {
            AttentionStorageTypes::from_single_dtype(f32::as_type_native_unchecked())
        }

        include!("tests.rs");
    }
}
