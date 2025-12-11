pub(crate) mod launcher;

mod reference;
mod utils;

pub(crate) use reference::assert_result;
pub(crate) use utils::tiling_scheme_ops;

mod unit {
    use cubek_attention::launch::{
        AttentionBlueprint, AttentionTileSize, RoutineStrategy, Strategy,
    };
    fn strategy(blueprint: AttentionBlueprint) -> Strategy {
        Strategy::Unit(RoutineStrategy::Forced(blueprint))
    }

    fn tile_size() -> AttentionTileSize {
        AttentionTileSize {
            seq_q: 4,
            seq_kv: 4,
            head_dim: 4,
            val_dim: 4,
        }
    }

    fn minimal_seq_q_stage() -> u32 {
        32
    }

    mod f16_ty {
        use super::*;
        use cubecl::frontend::CubePrimitive;
        use cubek_attention::launch::AttentionGlobalTypes;

        fn global_dtypes() -> AttentionGlobalTypes {
            AttentionGlobalTypes::from_single_dtype(half::f16::as_type_native_unchecked())
        }

        include!("tests.rs");
    }

    mod f32_ty {
        use super::*;
        use cubecl::frontend::CubePrimitive;
        use cubek_attention::launch::AttentionGlobalTypes;

        fn global_dtypes() -> AttentionGlobalTypes {
            AttentionGlobalTypes::from_single_dtype(f32::as_type_native_unchecked())
        }

        include!("tests.rs");
    }
}

mod blackbox_accelerated {
    use cubek_attention::launch::{
        AttentionBlueprint, AttentionTileSize, RoutineStrategy, Strategy,
    };

    fn strategy(blueprint: AttentionBlueprint) -> Strategy {
        Strategy::BlackboxAccelerated(RoutineStrategy::Forced(blueprint))
    }

    fn tile_size() -> AttentionTileSize {
        #[cfg(target_os = "macos")]
        {
            use cubek_attention::launch::AttentionTileSize;

            AttentionTileSize {
                seq_q: 8,
                seq_kv: 8,
                head_dim: 8,
                val_dim: 8,
            }
        }

        #[cfg(not(target_os = "macos"))]
        AttentionTileSize {
            seq_q: 8,
            seq_kv: 8,
            head_dim: 8,
            val_dim: 8,
        }
    }

    fn minimal_seq_q_stage() -> u32 {
        1
    }

    mod f16_ty {
        use super::*;
        use cubecl::frontend::CubePrimitive;
        use cubek_attention::launch::AttentionGlobalTypes;

        fn global_dtypes() -> AttentionGlobalTypes {
            AttentionGlobalTypes::from_single_dtype(half::f16::as_type_native_unchecked())
        }

        include!("tests.rs");
    }

    mod f32_ty {
        use super::*;
        use cubecl::frontend::CubePrimitive;
        use cubek_attention::launch::AttentionGlobalTypes;

        fn global_dtypes() -> AttentionGlobalTypes {
            AttentionGlobalTypes::from_single_dtype(f32::as_type_native_unchecked())
        }

        include!("tests.rs");
    }
}
