use cubecl::prelude::*;
use half::{bf16, f16};

use crate::{
    definition::{AccumulatorPrecision, AttentionGlobalTypes},
    launch::{AttentionArgs, TensorArgs},
};

/// Attention spec defining each element types used in the computation as well as
/// how the arguments are passed to the kernel.
pub trait AttentionSpec: Send + Sync + Clone + 'static {
    type Precision: AttentionPrecision;
    /// How the input and output tensors are passed as arguments.
    type Args: AttentionArgs;
}

impl<AP: AttentionPrecision, Args: AttentionArgs> AttentionSpec for (AP, Args) {
    type Precision = AP;
    type Args = Args;
}

// A simple default for TensorArgs
impl<AP: AttentionPrecision> AttentionSpec for AP {
    type Precision = AP;
    type Args = TensorArgs;
}

pub trait QueryPrecision: Send + Sync + Copy + 'static {
    type Global: Float;
    type GlobalSize: Size;
    type Tile: Float;
}

pub trait StagedMatrixPrecision: Send + Sync + Copy + 'static {
    type Global: Float;
    type GlobalSize: Size;
    type Stage: Float;
    type StageSize: Size;
}

pub trait AttentionPrecision: Send + Sync + Copy + 'static {
    type Query: QueryPrecision;
    type Key: StagedMatrixPrecision;
    type Value: StagedMatrixPrecision;
    type KVTile: Float;
    type SoftmaxAcc: Float;
    type SoftmaxLhs: Float;
    type Accumulator: Float;
    type Mask: Numeric;
    type MaskSize: Size;
    type Out: StagedMatrixPrecision;
}

impl QueryPrecision for f16 {
    type Global = f16;
    type GlobalSize = Const<0>;
    type Tile = f16;
}

impl QueryPrecision for bf16 {
    type Global = bf16;
    type GlobalSize = Const<0>;
    type Tile = bf16;
}

impl QueryPrecision for flex32 {
    type Global = f32;
    type GlobalSize = Const<0>;
    type Tile = f16;
}

impl QueryPrecision for f32 {
    type Global = f32;
    type GlobalSize = Const<0>;
    type Tile = f32;
}

impl QueryPrecision for f64 {
    type Global = f64;
    type GlobalSize = Const<0>;
    type Tile = f32;
}

impl<G: Float, GS: Size, T: Float> QueryPrecision for (G, GS, T) {
    type Global = G;
    type GlobalSize = GS;
    type Tile = T;
}

impl StagedMatrixPrecision for f16 {
    type Global = f16;
    type GlobalSize = Const<0>;
    type Stage = f16;
    type StageSize = Const<0>;
}

impl StagedMatrixPrecision for bf16 {
    type Global = bf16;
    type GlobalSize = Const<0>;
    type Stage = bf16;
    type StageSize = Const<0>;
}

impl StagedMatrixPrecision for flex32 {
    type Global = f32;
    type GlobalSize = Const<0>;
    type Stage = f16;
    type StageSize = Const<0>;
}

impl StagedMatrixPrecision for f32 {
    type Global = f32;
    type GlobalSize = Const<0>;
    type Stage = f32;
    type StageSize = Const<0>;
}

impl StagedMatrixPrecision for f64 {
    type Global = f64;
    type GlobalSize = Const<0>;
    type Stage = f32;
    type StageSize = Const<0>;
}

impl<G: Float, GS: Size, S: Float, SS: Size> StagedMatrixPrecision for (G, GS, S, SS) {
    type Global = G;
    type GlobalSize = GS;
    type Stage = S;
    type StageSize = SS;
}

impl AttentionPrecision for f16 {
    type Query = f16;
    type Key = f16;
    type Value = f16;
    type KVTile = f16;
    type SoftmaxLhs = f16;
    #[cfg(target_os = "macos")]
    type SoftmaxAcc = f16;
    #[cfg(target_os = "macos")]
    type Accumulator = f16;
    #[cfg(not(target_os = "macos"))]
    type SoftmaxAcc = f32;
    #[cfg(not(target_os = "macos"))]
    type Accumulator = f32;
    type Mask = u8;
    type MaskSize = Const<0>;
    type Out = f16;
}

impl AttentionPrecision for flex32 {
    type Query = flex32;
    type Key = flex32;
    type Value = flex32;
    type KVTile = f16;
    type SoftmaxLhs = f16;
    #[cfg(target_os = "macos")]
    type SoftmaxAcc = f16;
    #[cfg(target_os = "macos")]
    type Accumulator = f16;
    #[cfg(not(target_os = "macos"))]
    type SoftmaxAcc = f32;
    #[cfg(not(target_os = "macos"))]
    type Accumulator = f32;
    type Mask = u8;
    type MaskSize = Const<0>;
    type Out = f32;
}

impl AttentionPrecision for bf16 {
    type Query = bf16;
    type Key = bf16;
    type Value = bf16;
    type KVTile = bf16;
    type SoftmaxLhs = bf16;
    #[cfg(target_os = "macos")]
    type SoftmaxAcc = bf16;
    #[cfg(target_os = "macos")]
    type Accumulator = bf16;
    #[cfg(not(target_os = "macos"))]
    type SoftmaxAcc = f32;
    #[cfg(not(target_os = "macos"))]
    type Accumulator = f32;
    type Mask = u8;
    type MaskSize = Const<0>;
    type Out = bf16;
}

impl AttentionPrecision for f32 {
    type Query = f32;
    type Key = f32;
    type Value = f32;
    type KVTile = f32;
    type SoftmaxAcc = f32;
    type SoftmaxLhs = f32;
    type Accumulator = f32;
    type Mask = u8;
    type MaskSize = Const<0>;
    type Out = f32;
}

impl AttentionPrecision for f64 {
    type Query = f64;
    type Key = f64;
    type Value = f64;
    type KVTile = f32;
    type SoftmaxAcc = f32;
    type SoftmaxLhs = f32;
    type Accumulator = f32;
    type Mask = u8;
    type MaskSize = Const<0>;
    type Out = f64;
}

impl<
    QG: Float,
    QGS: Size,
    QT: Float,
    KG: Float,
    KGS: Size,
    KS: Float,
    KSS: Size,
    VG: Float,
    VGS: Size,
    VS: Float,
    VSS: Size,
    KVT: Float,
    SM: Float,
    SML: Float,
    ACC: Float,
    MSK: Numeric,
    MSKS: Size,
    OG: Float,
    OGS: Size,
    OS: Float,
    OSS: Size,
> AttentionPrecision
    for (
        (QG, QGS, QT),
        (KG, KGS, KS, KSS),
        (VG, VGS, VS, VSS),
        KVT,
        SM,
        SML,
        ACC,
        MSK,
        MSKS,
        (OG, OGS, OS, OSS),
    )
{
    type Query = (QG, QGS, QT);
    type Key = (KG, KGS, KS, KSS);
    type Value = (VG, VGS, VS, VSS);
    type KVTile = KVT;
    type SoftmaxAcc = SM;
    type SoftmaxLhs = SML;
    type Accumulator = ACC;
    type Mask = MSK;
    type MaskSize = MSKS;
    type Out = (OG, OGS, OS, OSS);
}

/// Input argument
pub type InputArg<AA> = <AA as AttentionArgs>::Input<
    (
        NumericExpand<0>, // QG
        SizeExpand<1>,    // QGS
    ),
    (
        NumericExpand<2>, // KG
        SizeExpand<3>,    // KGS
    ),
    (
        NumericExpand<4>, // VG
        SizeExpand<5>,    // VGS
    ),
    (
        NumericExpand<6>, // MSK
        SizeExpand<7>,    // MSKS
    ),
>;

/// Output argument
pub type OutputArg<AA> = <AA as AttentionArgs>::Output<(NumericExpand<8>, SizeExpand<9>)>; // OG, OGS

/// Input runtime argument
pub type InputRuntimeArg<AA, R> = <InputArg<AA> as LaunchArg>::RuntimeArg<R>;

/// Output runtime argument
pub type OutputRuntimeArg<AA, R> = <OutputArg<AA> as LaunchArg>::RuntimeArg<R>;

pub mod attention_types {
    use crate::definition::{
        AttentionPrecision, AttentionSpec, QueryPrecision, StagedMatrixPrecision,
    };

    pub type QG<AS> =
        <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Query as QueryPrecision>::Global;
    pub type QGS<AS> =
        <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Query as QueryPrecision>::GlobalSize;
    pub type QT<AS> =
        <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Query as QueryPrecision>::Tile;
    pub type KG<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Key as StagedMatrixPrecision>::Global;
    pub type KGS<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Key as StagedMatrixPrecision>::GlobalSize;
    pub type KS<AS> =
        <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Key as StagedMatrixPrecision>::Stage;
    pub type KSS<AS> =
        <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Key as StagedMatrixPrecision>::StageSize;
    pub type VG<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Value as StagedMatrixPrecision>::Global;
    pub type VGS<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Value as StagedMatrixPrecision>::GlobalSize;
    pub type VS<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Value as StagedMatrixPrecision>::Stage;
    pub type VSS<AS> =
    <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Value as StagedMatrixPrecision>::StageSize;

    pub type KVT<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::KVTile;
    pub type SM<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::SoftmaxAcc;
    pub type SML<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::SoftmaxLhs;

    pub type ACC<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::Accumulator;
    pub type MSK<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::Mask;
    pub type MSKS<AS> = <<AS as AttentionSpec>::Precision as AttentionPrecision>::MaskSize;

    pub type OG<AS> = <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Out as StagedMatrixPrecision>::Global;
    pub type OGS<AS> = <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Out as StagedMatrixPrecision>::GlobalSize;
    pub type OS<AS> = <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Out as StagedMatrixPrecision>::Stage;
    pub type OSS<AS> = <<<AS as AttentionSpec>::Precision as AttentionPrecision>::Out as StagedMatrixPrecision>::StageSize;
}

pub type Args<MS> = <MS as AttentionSpec>::Args;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct AttentionElems {
    pub query_global: StorageType,
    pub query_tile: StorageType,
    pub key_global: StorageType,
    pub key_stage: StorageType,
    pub value_global: StorageType,
    pub value_stage: StorageType,
    pub key_value_tile: StorageType,
    pub softmax_acc: StorageType,
    pub softmax_lhs: StorageType,
    pub accumulator: StorageType,
    pub mask: StorageType,
    pub out_global: StorageType,
    pub out_stage: StorageType,
}

impl AttentionElems {
    pub fn from_global_types(
        global_dtypes: &AttentionGlobalTypes,
        tile_type: StorageType,
        accumulator_precision: &AccumulatorPrecision,
    ) -> AttentionElems {
        let accumulator = match accumulator_precision {
            AccumulatorPrecision::Strict(storage_type) => *storage_type,
            AccumulatorPrecision::Loose => AccumulatorPrecision::default_accumulator_type(),
        };

        Self {
            query_global: global_dtypes.query,
            query_tile: tile_type,
            key_global: global_dtypes.key,
            key_stage: tile_type,
            value_global: global_dtypes.value,
            value_stage: tile_type,
            key_value_tile: tile_type,
            softmax_acc: accumulator,
            softmax_lhs: tile_type,
            accumulator,
            mask: global_dtypes.mask,
            out_global: global_dtypes.out,
            out_stage: global_dtypes.out,
        }
    }
}

impl From<&AttentionElems> for [StorageType; 5] {
    fn from(elems: &AttentionElems) -> Self {
        [
            elems.query_global,
            elems.key_global,
            elems.value_global,
            elems.mask,
            elems.out_global,
        ]
    }
}
