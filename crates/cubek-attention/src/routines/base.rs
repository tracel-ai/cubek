use std::fmt::Debug;

use cubecl::client::ComputeClient;
use cubecl::{CubeDim, Runtime, tensor_line_size_parallel};
use cubek_std::test_utils::contiguous_strides;

use crate::components::tile::TileAttentionFamily;
use crate::components::{
    batch::BatchAttentionFamily, global::GlobalAttentionFamily, stage::StageAttentionFamily,
};
use crate::launch::{
    AttentionBlueprint, AttentionDefinition, AttentionElems, AttentionIdent, AttentionLineSizes,
    AttentionSetupError, CubeCountPlan, RoutineStrategy,
};

pub trait Routine: Debug + Clone {
    type TileAttention: TileAttentionFamily;
    type StageAttention: StageAttentionFamily;
    type GlobalAttention: GlobalAttentionFamily;
    type BatchAttention: BatchAttentionFamily;

    type Strategy;

    fn prepare(
        definition: &AttentionDefinition,
        device_settings: &DeviceSettings,
        strategy: RoutineStrategy<Self>,
    ) -> Result<LaunchInfo, AttentionSetupError>;
}

pub struct LaunchInfo {
    pub blueprint: AttentionBlueprint,
    pub dtypes: AttentionElems,
    pub cube_dim: CubeDim,
    pub cube_count_plan: CubeCountPlan,
}

pub struct DeviceSettings {
    pub plane_dim: u32,
    pub line_sizes: AttentionLineSizes,
}

impl DeviceSettings {
    pub fn new<R: Runtime>(client: &ComputeClient<R>, definition: &AttentionDefinition) -> Self {
        let find_line_size = |shape: &[usize; 4], dtype_size: usize| -> u8 {
            let supported_line_sizes = client.io_optimized_line_sizes_unchecked(dtype_size);

            tensor_line_size_parallel(
                supported_line_sizes,
                shape,
                &contiguous_strides(shape, false),
                shape.len() - 1,
            )
        };

        let line_sizes = AttentionLineSizes {
            query: find_line_size(
                &definition.dims.shape(AttentionIdent::Query),
                definition.global_dtypes.query.size(),
            ),
            key: find_line_size(
                &definition.dims.shape(AttentionIdent::Key),
                definition.global_dtypes.key.size(),
            ),
            value: find_line_size(
                &definition.dims.shape(AttentionIdent::Value),
                definition.global_dtypes.value.size(),
            ),
            // lined mask not always supported at the moment
            mask: 1,
            out: find_line_size(
                &definition.dims.shape(AttentionIdent::Query),
                definition.global_dtypes.out.size(),
            ),
        };

        DeviceSettings {
            plane_dim: client.properties().hardware.plane_size_max,
            line_sizes,
        }
    }
}
