use cubecl;
use cubecl::{ir::DeviceProperties, prelude::*};
use cubek_matmul::components::{CubeDimResource, tile::Tile};
use cubek_std::InvalidConfigError;

use crate::definition::attention_types::{ACC, KSS, KVT, OSS, QGS, QT, SM, SML, VSS};
use crate::definition::{
    AttentionBlueprint, AttentionElems, AttentionPrecision, AttentionSetupError,
};
use crate::{
    components::tile::TileAttentionConfig, components::tile::matmul::InnerMatmul,
    components::tile::output::AttentionOutput, components::tile::softmax::Softmax,
};

/// Logits below this are considered masked (effectively -inf)
/// Value chosen to fit within f16 range (~-65,504 max)
pub(crate) const LOGIT_MASKED: f32 = -6e4;

/// Any value smaller than this is considered numerically zero
/// (used for fully-masked rows or tiny contributions)
/// Value chosen to be above f16 smallest normal (~6.1e-5)
pub(crate) const FULLY_MASKED_ROW_THRESHOLD: f32 = 1e-4;

#[cube]
pub trait TileAttention<AP: AttentionPrecision>: Send + Sync + 'static {
    type Config: TileAttentionConfig<
            ScoreMatmulConfig = <Self::ScoreMatmul as InnerMatmul<
                QT<AP>,
                QGS<AP>,
                KVT<AP>,
                KSS<AP>,
                SM<AP>,
                Const<0>,
            >>::Config,
            ValueMatmulConfig = <Self::ValueMatmul as InnerMatmul<
                SML<AP>,
                Const<0>,
                KVT<AP>,
                VSS<AP>,
                ACC<AP>,
                OSS<AP>,
            >>::Config,
            AttentionOutputConfig = <Self::Output as AttentionOutput<ACC<AP>, OSS<AP>>>::Config,
        >;
    type ScoreMatmul: InnerMatmul<QT<AP>, QGS<AP>, KVT<AP>, KSS<AP>, SM<AP>, Const<0>>;
    type Softmax: Softmax<
            SM<AP>,
            ScoreTile = Tile<SM<AP>, Const<0>, ReadWrite>,
            SoftmaxedTile = Tile<SML<AP>, Const<0>, ReadWrite>,
            ScaleColumn = <Self::Output as AttentionOutput<ACC<AP>, OSS<AP>>>::ScaleColumn,
            RunningState = <Self::Output as AttentionOutput<ACC<AP>, OSS<AP>>>::RunningState,
            Config = <Self::Config as TileAttentionConfig>::SoftmaxConfig,
        >;
    type ValueMatmul: InnerMatmul<SML<AP>, Const<0>, KVT<AP>, VSS<AP>, ACC<AP>, OSS<AP>>;
    type Output: AttentionOutput<ACC<AP>, OSS<AP>>;
}

pub trait TileAttentionFamily: Send + Sync + 'static {
    /// The specific TileMatmul implementation associated with this family.
    type TileAttention<AP: AttentionPrecision>: TileAttention<AP, Config = Self::Config>;

    /// The configuration type associated with this matmul family.
    type Config: TileAttentionConfig;

    /// Returns whether this tile matmul requires specialized hardware accelerators (e.g., tensor cores).
    fn requires_accelerator() -> bool;

    /// Returns the compute resources required to run this tile matmul.
    fn computation_resources() -> Result<CubeDimResource, InvalidConfigError>;

    /// Constructs the configuration based on the algorithm's blueprint.
    ///
    /// This function may return an error if the configuration cannot be supported.
    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &AttentionBlueprint,
        dtypes: &AttentionElems,
    ) -> Result<Self::Config, AttentionSetupError>;
}
