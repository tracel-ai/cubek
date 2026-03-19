use cubecl::prelude::*;
use cubecl::{self, ir::DeviceProperties};
use cubek_matmul::components::{
    global::{WriteEventListener, WriteTiling, read::sync_full_cyclic::SyncFullCyclicLoading},
    stage::{ContiguousTilingLayout, RowMajorTilingOrder, StageFamily},
};
use cubek_std::stage::StageMemoryConfig;
use std::{fmt::Debug, hash::Hash};

use crate::components::tile::TileAttentionConfig;
use crate::definition::{
    AttentionElems, AttentionPartitionSize, AttentionPrecision, AttentionStageSize,
    AttentionTileSize,
};
use crate::{components::global::GlobalAttentionConfig, definition::attention_types::*};
use crate::{
    components::{global::simple::MaskReader, stage::AttentionPartitioner},
    definition::AttentionSetupError,
};
use crate::{
    components::{
        global::simple::QueryReader,
        stage::{plane::PlanePartitionStageConfig, unit::UnitPartitionStageConfig},
    },
    definition::AttentionBlueprint,
};
use cubecl::std::tensor::layout::Coords2d;

pub type AttentionTilingLayout = ContiguousTilingLayout<RowMajorTilingOrder>;
pub type AttentionLoadingStrategy = SyncFullCyclicLoading<RowMajorTilingOrder>;

/// A family of TileAttention implementations that operate with any [precision](AttentionPrecision).
pub trait StageAttentionFamily: Send + Sync + 'static {
    /// The specific TileAttention implementation associated with this family.
    type Attention<AP: AttentionPrecision>: StageAttention<
            AP,
            Config = Self::Config,
            KeyStage = <Self::KeyStage as StageFamily>::Stage<
                KS<AP>,
                KSS<AP>,
                AttentionTilingLayout,
            >,
            ValueStage = <Self::ValueStage as StageFamily>::Stage<
                VS<AP>,
                VSS<AP>,
                AttentionTilingLayout,
            >,
            OutStage = <Self::OutStage as StageFamily<ReadWrite>>::Stage<
                OS<AP>,
                OSS<AP>,
                WriteTiling,
            >,
        >;

    /// The configuration type associated with this Attention family.
    type Config: StageAttentionConfig;

    type KeyStage: StageFamily;
    type ValueStage: StageFamily;
    type OutStage: StageFamily<ReadWrite>;

    /// Constructs the configuration based on the algorithm's blueprint.
    ///
    /// This function may return an error if the configuration cannot be supported.
    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &AttentionBlueprint,
        dtypes: &AttentionElems,
    ) -> Result<Self::Config, AttentionSetupError>;
}

#[cube]
pub trait StageAttention<AP: AttentionPrecision>: 'static + Send + Sync {
    type KeyStage: CubeType;
    type ValueStage: CubeType;
    type OutStage: CubeType;

    /// The configuration type associated with this Attention.
    type Config: StageAttentionConfig;
    type Partitioner: AttentionPartitioner;

    type QueryPartition: CubeType;
    type KeyPartition: CubeType;
    type ValuePartition: CubeType;
    type SoftmaxPartition: CubeType;
    type OutputPartition: CubeType;
    type MaskPartition: CubeType;
    type RunningState: CubeType;

    fn init_state(#[comptime] config: Self::Config) -> Sequence<Self::RunningState>;

    fn execute(
        query: &Self::QueryPartition,
        key_stage: &Self::KeyStage,
        value_stage: &Self::ValueStage,
        key_partition: &mut Self::KeyPartition,
        value_partition: &mut Self::ValuePartition,
        mask_reader: &MaskReader<AP>,
        mask_partition: &mut Self::MaskPartition,
        softmax_partition: &mut Self::SoftmaxPartition,
        output: &mut Self::OutputPartition,
        prev_state: &mut Sequence<Self::RunningState>,
        #[comptime] config: Self::Config,
    );

    fn rescale(
        acc: &mut Self::OutputPartition,
        state: Sequence<Self::RunningState>,
        #[comptime] config: Self::Config,
    );

    fn write<W: WriteEventListener, G: GlobalAttentionConfig>(
        acc: &Self::OutputPartition,
        stage: &mut Self::OutStage,
        writer: &mut W,
        #[comptime] config: Self::Config,
    );

    fn init_query(#[comptime] config: Self::Config) -> Self::QueryPartition;
    fn init_key(#[comptime] config: Self::Config) -> Self::KeyPartition;
    fn init_value(#[comptime] config: Self::Config) -> Self::ValuePartition;
    fn init_mask(
        out_of_bounds: ComptimeOption<Coords2d>,
        #[comptime] config: Self::Config,
    ) -> Self::MaskPartition;
    fn init_softmax(#[comptime] config: Self::Config) -> Self::SoftmaxPartition;
    fn init_output(#[comptime] config: Self::Config) -> Self::OutputPartition;

    fn read_query(
        reader: &QueryReader<AP>,
        registers: &mut Self::QueryPartition,
        #[comptime] config: Self::Config,
    );
}

/// Configuration for the Tile Attention level
pub trait StageAttentionConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    type TileConfig: TileAttentionConfig;

    fn tile_config(&self) -> Self::TileConfig;
    fn tile_size(&self) -> AttentionTileSize;

    fn elements_in_partition_seq_q(&self) -> u32;
    fn elements_in_partition_seq_kv(&self) -> u32;
    fn elements_in_stage_seq_q(&self) -> u32;

    fn plane_dim(&self) -> u32;
    fn num_planes(&self) -> u32;

    fn key_smem_config(&self) -> StageMemoryConfig;
    fn value_smem_config(&self) -> StageMemoryConfig;
    fn out_smem_config(&self) -> StageMemoryConfig;
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum PartitionAttentionConfig<TC: TileAttentionConfig> {
    Unit(UnitPartitionStageConfig<TC>),
    Plane(PlanePartitionStageConfig<TC>),
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SharedPartitionAttentionConfig<TC: TileAttentionConfig> {
    pub tile_config: TC,
    pub partition_size: AttentionPartitionSize,
    pub stage_size: AttentionStageSize,
    pub num_planes: u32,
    pub key_smem_config: StageMemoryConfig,
    pub value_smem_config: StageMemoryConfig,
    pub out_smem_config: StageMemoryConfig,
}

impl<TC: TileAttentionConfig> PartitionAttentionConfig<TC> {
    pub fn shared(&self) -> SharedPartitionAttentionConfig<TC> {
        match self {
            PartitionAttentionConfig::Unit(unit_partition_stage_config) => {
                unit_partition_stage_config.shared
            }
            PartitionAttentionConfig::Plane(plane_partition_stage_config) => {
                plane_partition_stage_config.shared
            }
        }
    }
}

impl<TC: TileAttentionConfig> StageAttentionConfig for PartitionAttentionConfig<TC> {
    type TileConfig = TC;

    fn tile_config(&self) -> Self::TileConfig {
        self.shared().tile_config
    }

    fn num_planes(&self) -> u32 {
        self.shared().num_planes
    }

    fn plane_dim(&self) -> u32 {
        self.tile_config().plane_dim()
    }

    fn key_smem_config(&self) -> StageMemoryConfig {
        self.shared().key_smem_config
    }

    fn value_smem_config(&self) -> StageMemoryConfig {
        self.shared().value_smem_config
    }

    fn out_smem_config(&self) -> StageMemoryConfig {
        self.shared().out_smem_config
    }

    fn tile_size(&self) -> AttentionTileSize {
        self.tile_config().tile_size()
    }

    fn elements_in_partition_seq_q(&self) -> u32 {
        self.shared().partition_size.seq_q * self.tile_size().seq_q
    }

    fn elements_in_partition_seq_kv(&self) -> u32 {
        self.shared().partition_size.seq_kv * self.tile_size().seq_kv
    }

    fn elements_in_stage_seq_q(&self) -> u32 {
        self.shared().stage_size.seq_q * self.elements_in_partition_seq_q()
    }
}
