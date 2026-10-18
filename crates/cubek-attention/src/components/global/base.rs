use cubecl::{
    prelude::*,
    {self, ir::DeviceProperties},
};

use crate::{
    components::global::simple::AttentionWriter,
    forward::definition::{
        AttentionBlueprint, AttentionElems, AttentionPrecision, AttentionSetupError,
        attention_types::*,
    },
};
use cubecl::std::tensor::r#virtual::VirtualTensor;

use crate::components::{
    global::{GlobalHeads, simple::QueryReader},
    stage::StageAttentionConfig,
};
use std::{fmt::Debug, hash::Hash};

/// A family of [GlobalAttention] implementations that operate with any [precision](AttentionPrecision).
pub trait GlobalAttentionFamily: Send + Sync + 'static {
    /// The specific [GlobalAttention] implementation associated with this family.
    type Attention<AP: AttentionPrecision>: GlobalAttention<AP, Config = Self::Config>;

    /// The configuration type associated with this Attention family.
    type Config: GlobalAttentionConfig;

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
pub trait GlobalAttention<AP: AttentionPrecision>: 'static {
    /// Writes to Out at the same offset it loaded Query
    type Writer<'a>: AttentionWriter<'a, OS<AP>, OSS<AP>, OG<AP>, OGS<AP>>;

    /// Loads to SMEM as is (transposed later)
    type KeyReader: CubeType;
    /// Loads to SMEM as is
    type ValueReader: CubeType;
    /// Loads to SMEM as is
    type MaskReader: CubeType;

    /// The configuration type associated with this Attention.
    type Config: GlobalAttentionConfig;

    fn execute(
        query_reader: QueryReader<AP>,
        key_reader: Self::KeyReader,
        value_reader: Self::ValueReader,
        mask_reader: Self::MaskReader,
        writer: Self::Writer<'_>,
        seq_q: u32,
        seq_kv: u32,
        #[comptime] config: Self::Config,
    );

    // Every reader takes the same resolved `GlobalHeads` and picks the head it
    // indexes by: K and V use the kv head, Q / Mask / Out use the query head.

    fn init_query_reader(
        heads: GlobalHeads,
        stage_q_offset: u32,
        query: VirtualTensor<QG<AP>, QGS<AP>>,
        #[comptime] config: Self::Config,
    ) -> QueryReader<'static, AP>;

    fn init_key_reader(
        heads: GlobalHeads,
        key: VirtualTensor<KG<AP>, KGS<AP>>,
        #[comptime] config: Self::Config,
    ) -> Self::KeyReader;

    fn init_value_reader(
        heads: GlobalHeads,
        value: VirtualTensor<VG<AP>, VGS<AP>>,
        #[comptime] config: Self::Config,
    ) -> Self::ValueReader;

    fn init_mask_reader(
        heads: GlobalHeads,
        stage_q_offset: u32,
        mask: ComptimeOption<VirtualTensor<MSK<AP>, MSKS<AP>>>,
        #[comptime] config: Self::Config,
    ) -> Self::MaskReader;

    fn init_writer(
        heads: GlobalHeads,
        stage_q_offset: u32,
        out: VirtualTensor<OG<AP>, OGS<AP>, ReadWrite>,
        #[comptime] config: Self::Config,
    ) -> Self::Writer<'static>;
}

/// Configuration for the Global Attention level
pub trait GlobalAttentionConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    type StageConfig: StageAttentionConfig;

    fn stage_config(&self) -> Self::StageConfig;
    fn cube_dim(&self) -> CubeDim;
}
