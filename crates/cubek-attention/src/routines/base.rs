use std::fmt::Debug;

use cubecl::client::ComputeClient;
use cubecl::{CubeDim, Runtime};

use crate::components::tile::TileAttentionFamily;
use crate::components::{
    batch::BatchAttentionFamily, global::GlobalAttentionFamily, stage::StageAttentionFamily,
};
use crate::launch::{
    AttentionBlueprint, AttentionDefinition, AttentionElems, AttentionLineSizes,
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
        DeviceSettings {
            plane_dim: client.properties().hardware.plane_size_max,
            line_sizes: AttentionLineSizes::new_max(client, definition),
        }
    }
}
