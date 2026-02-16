pub(crate) mod launcher;

mod reference;
mod utils;

use cubecl::frontend::CubePrimitive;
use cubecl::{Runtime, client::ComputeClient, ir::StorageType};
pub(crate) use reference::assert_result;
pub(crate) use utils::tiling_scheme_ops;

mod unit {
    use cubecl::{Runtime, client::ComputeClient};
    use cubek_attention::{
        definition::{
            AttentionBlueprint, AttentionGlobalTypes, AttentionLineSizes, AttentionTileSize,
        },
        launch::{BlueprintStrategy, Strategy},
    };
    fn forced_strategy(blueprint: AttentionBlueprint) -> Strategy {
        Strategy::Unit(BlueprintStrategy::Forced(blueprint))
    }
    fn inferred_strategy() -> Strategy {
        Strategy::Unit(BlueprintStrategy::Inferred(()))
    }

    fn minimal_seq_q_stage() -> u32 {
        32
    }

    fn tile_size<R: Runtime>(
        client: &ComputeClient<R>,
        global_types: AttentionGlobalTypes,
    ) -> AttentionTileSize {
        let line_sizes = AttentionLineSizes::new_max(client, &global_types);
        AttentionTileSize {
            seq_q: line_sizes.query as u32,
            seq_kv: line_sizes.key as u32,
            head_dim: line_sizes.query as u32,
            val_dim: line_sizes.value as u32,
        }
    }

    mod f16_ty {
        use crate::attention::mask_dtype;

        use super::*;
        use cubecl::frontend::CubePrimitive;
        use cubek_attention::definition::AttentionGlobalTypes;

        fn global_dtypes<R: Runtime>(client: &ComputeClient<R>) -> AttentionGlobalTypes {
            AttentionGlobalTypes::from_single_float_dtype(
                half::f16::as_type_native_unchecked(),
                mask_dtype(client),
            )
        }

        include!("blueprint_tests.rs");
        include!("selector_tests.rs");
    }

    mod f32_ty {
        use crate::attention::mask_dtype;

        use super::*;
        use cubecl::frontend::CubePrimitive;
        use cubek_attention::definition::AttentionGlobalTypes;

        fn global_dtypes<R: Runtime>(client: &ComputeClient<R>) -> AttentionGlobalTypes {
            AttentionGlobalTypes::from_single_float_dtype(
                f32::as_type_native_unchecked(),
                mask_dtype(client),
            )
        }

        include!("blueprint_tests.rs");
        include!("selector_tests.rs");
    }
}

mod blackbox_accelerated {
    use cubecl::{Runtime, client::ComputeClient};
    use cubek_attention::{
        definition::{AttentionBlueprint, AttentionGlobalTypes, AttentionTileSize},
        launch::{BlueprintStrategy, Strategy},
    };

    fn forced_strategy(blueprint: AttentionBlueprint) -> Strategy {
        Strategy::BlackboxAccelerated(BlueprintStrategy::Forced(blueprint))
    }
    fn inferred_strategy() -> Strategy {
        Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(()))
    }

    fn tile_size<R: Runtime>(
        _client: &ComputeClient<R>,
        _global_types: AttentionGlobalTypes,
    ) -> AttentionTileSize {
        #[cfg(target_os = "macos")]
        {
            use cubek_attention::definition::AttentionTileSize;

            AttentionTileSize {
                seq_q: 8,
                seq_kv: 8,
                head_dim: 8,
                val_dim: 8,
            }
        }

        #[cfg(not(target_os = "macos"))]
        AttentionTileSize {
            seq_q: 16,
            seq_kv: 16,
            head_dim: 16,
            val_dim: 16,
        }
    }

    fn minimal_seq_q_stage() -> u32 {
        1
    }

    mod f16_ty {
        use crate::attention::mask_dtype;

        use super::*;
        use cubecl::frontend::CubePrimitive;
        use cubek_attention::definition::AttentionGlobalTypes;

        fn global_dtypes<R: Runtime>(client: &ComputeClient<R>) -> AttentionGlobalTypes {
            AttentionGlobalTypes::from_single_float_dtype(
                half::f16::as_type_native_unchecked(),
                mask_dtype(client),
            )
        }

        include!("blueprint_tests.rs");
        include!("selector_tests.rs");
    }

    mod f32_ty {
        use crate::attention::mask_dtype;

        use super::*;
        use cubecl::frontend::CubePrimitive;
        use cubek_attention::definition::AttentionGlobalTypes;

        fn global_dtypes<R: Runtime>(client: &ComputeClient<R>) -> AttentionGlobalTypes {
            AttentionGlobalTypes::from_single_float_dtype(
                f32::as_type_native_unchecked(),
                mask_dtype(client),
            )
        }

        include!("blueprint_tests.rs");
        include!("selector_tests.rs");
    }
}
fn mask_dtype<R: Runtime>(client: &ComputeClient<R>) -> StorageType {
    let props = client.properties();
    let u8_ty = u8::as_type_native_unchecked();
    let u32_ty = u32::as_type_native_unchecked();

    if props.supports_type(u8_ty) {
        u8_ty
    } else if props.supports_type(u32_ty) {
        u32_ty
    } else {
        panic!("Client does not support u8 or u32 native types");
    }
}
