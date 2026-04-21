use std::fmt::Debug;

use cubecl::{
    {CubeDim, Runtime},
    {client::ComputeClient, ir::AddressType},
};

use crate::components::tile::TileAttentionFamily;
use crate::components::{
    batch::BatchAttentionFamily, global::GlobalAttentionFamily, stage::StageAttentionFamily,
};
use crate::definition::{
    AttentionElems, AttentionProblem, AttentionSetupError, AttentionVectorSizes, CubeCountPlan,
};
use crate::launch::BlueprintStrategy;

pub trait Routine: Debug + Clone {
    type TileAttention: TileAttentionFamily;
    type StageAttention: StageAttentionFamily;
    type GlobalAttention: GlobalAttentionFamily;
    type BatchAttention: BatchAttentionFamily<Blueprint = Self::Blueprint>;

    type Strategy;
    type Blueprint: Clone;

    fn prepare<R: Runtime>(
        problem: &AttentionProblem,
        device_settings: &DeviceSettings<R>,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<LaunchInfo<Self::Blueprint>, AttentionSetupError>;
}

pub struct LaunchInfo<B> {
    pub blueprint: B,
    pub dtypes: AttentionElems,
    pub cube_dim: CubeDim,
    pub cube_count_plan: CubeCountPlan,
    pub address_type: AddressType,
}

pub struct DeviceSettings<R: Runtime> {
    pub plane_dim: u32,
    pub max_cube_count: (u32, u32, u32),
    pub vector_sizes: AttentionVectorSizes,
    pub client: ComputeClient<R>,
}

impl<R: Runtime> core::fmt::Debug for DeviceSettings<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceSettings")
            .field("plane_dim", &self.plane_dim)
            .field("max_cube_count", &self.max_cube_count)
            .field("vector_sizes", &self.vector_sizes)
            .finish()
    }
}

impl<R: Runtime> DeviceSettings<R> {
    pub fn new(client: &ComputeClient<R>, problem: &AttentionProblem) -> Self {
        DeviceSettings {
            plane_dim: client.properties().hardware.plane_size_max,
            max_cube_count: client.properties().hardware.max_cube_count,
            vector_sizes: AttentionVectorSizes::new_max_for_problem(client, problem),
            client: client.clone(),
        }
    }
}
