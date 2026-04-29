use cubecl;
use cubecl::{ir::DeviceProperties, prelude::*};
use cubek_std::{
    CubeDimResource, InvalidConfigError,
    tile::{Plane, Tile},
};

use crate::definition::attention_types::{ACC, OSS, SM, SML};
use crate::definition::{
    AttentionBlueprint, AttentionElems, AttentionPrecision, AttentionSetupError,
};
use crate::{
    components::tile::TileAttentionConfig, components::tile::output::AttentionOutput,
    components::tile::softmax::Softmax,
};

#[cube]
pub trait TileAttention<AP: AttentionPrecision>: Send + Sync + 'static {
    type Config: TileAttentionConfig<
            AttentionOutputConfig = <Self::Output as AttentionOutput<ACC<AP>, OSS<AP>>>::Config,
        >;
    type Softmax: Softmax<
            SM<AP>,
            ScoreTile = Tile<SM<AP>, Const<0>, Plane, ReadWrite>,
            SoftmaxedTile = Tile<SML<AP>, Const<0>, Plane, ReadWrite>,
            ScaleColumn = <Self::Output as AttentionOutput<ACC<AP>, OSS<AP>>>::ScaleColumn,
            RunningState = <Self::Output as AttentionOutput<ACC<AP>, OSS<AP>>>::RunningState,
            Config = <Self::Config as TileAttentionConfig>::SoftmaxConfig,
        >;
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
