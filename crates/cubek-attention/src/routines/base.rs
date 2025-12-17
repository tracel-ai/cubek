use std::fmt::Debug;

use cubecl::client::ComputeClient;
use cubecl::{CubeDim, Runtime};

use crate::components::tile::TileAttentionFamily;
use crate::components::{
    batch::BatchAttentionFamily, global::GlobalAttentionFamily, stage::StageAttentionFamily,
};
use crate::definition::{
    AttentionElems, AttentionLineSizes, AttentionProblem, AttentionSetupError, CubeCountPlan,
};
use crate::launch::BlueprintStrategy;

pub trait Routine: Debug + Clone {
    type TileAttention: TileAttentionFamily;
    type StageAttention: StageAttentionFamily;
    type GlobalAttention: GlobalAttentionFamily;
    type BatchAttention: BatchAttentionFamily<Blueprint = Self::Blueprint>;

    type Strategy;
    type Blueprint;

    fn prepare(
        problem: &AttentionProblem,
        device_settings: &DeviceSettings,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<LaunchInfo<Self::Blueprint>, AttentionSetupError>;
}

pub struct LaunchInfo<B> {
    pub blueprint: B,
    pub dtypes: AttentionElems,
    pub cube_dim: CubeDim,
    pub cube_count_plan: CubeCountPlan,
}

pub struct DeviceSettings {
    pub plane_dim: u32,
    pub line_sizes: AttentionLineSizes,
}

impl DeviceSettings {
    pub fn new<R: Runtime>(client: &ComputeClient<R>, problem: &AttentionProblem) -> Self {
        DeviceSettings {
            plane_dim: client.properties().hardware.plane_size_max,
            line_sizes: AttentionLineSizes::new_max(client, problem),
        }
    }
}
