use cubecl::{
    features::MmaConfig,
    ir::{DeviceProperties, StorageType},
    prelude::*,
};
use cubek_std::{
    InvalidConfigError, MatrixLayout, TileSize,
    tile::{Strided, TileKind},
};

use crate::{
    components::{
        resource::CubeDimResource,
        tile::{TileConfig, Tilex},
    },
    definition::{MatmulElems, MatmulSetupError, MatmulVectorSizes, TilingBlueprint},
};

/// A family of [TileMatmul] implementations that operate with any precision.
pub trait TileMatmulFamily: Send + Sync + 'static {
    /// Config for this matmul
    type Config: TileConfig;

    /// The specific [TileMatmul] implementation associated with this family.
    type Matmul<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size>: TileMatmul<L, VL, R, VR, A, VA, Config = Self::Config>;

    /// Returns whether this tile matmul requires specialized hardware accelerators (e.g., tensor cores).
    fn requires_accelerator() -> bool;

    /// Whether this matmul family is able to cast on load/store from the stage.
    fn can_cast_stage_element() -> bool;

    /// Returns whether this tile matmul may benefit from swizzling.
    /// Used to determine the selection, since swizzling may require different stage sizes.
    fn should_swizzle<R: Runtime>(client: &ComputeClient<R>) -> bool;

    /// Returns the compute resources required to run this matmul.
    fn cubedim_resource() -> Result<CubeDimResource, InvalidConfigError>;

    /// Constructs the configuration based on the matmul problem, selection, and vector sizes.
    ///
    /// This function may return an error if the configuration cannot be supported on the current runtime.
    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError>;

    /// Returns whether a tile configuration is supported
    fn is_supported<R: Runtime>(_client: &ComputeClient<R>, _config: MmaConfig) -> bool {
        !Self::requires_accelerator()
    }

    /// Returns all sizes supported for these types, if any
    fn supported_sizes<R: Runtime>(
        _client: &ComputeClient<R>,
        _lhs_ty: StorageType,
        _rhs_ty: StorageType,
        _acc_ty: StorageType,
    ) -> Vec<TileSize> {
        Vec::new()
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError>;
}

pub trait TileIO: CubeType + Send + Sync + 'static {
    /// Tile for the lhs and rhs data
    type In: TileKind;
    /// Tile for the accumulator data
    type Acc: TileKind;
    /// Tile for the output data
    type Out: TileKind<ReadWrite>;
}

#[derive(CubeType)]
pub struct StandardTileIO {}

impl TileIO for StandardTileIO {
    type In = Strided;
    type Acc = Option<Strided>;
    type Out = Strided;
}

pub trait Operands: CubeType + Send + Sync + 'static {
    type Lhs: CubeType;
    type Rhs: CubeType;
    type Acc: CubeType;
}

/// Provides matrix multiplication operations at the tile level.
///
/// At the tile level,
///  - Dimensions M, N and K are fixed to an integer, and the
///    matrix multiplication works only for size (M, K) · (K, N) = (M, N).
///
/// Assumptions:
///  - Inputs must always be valid. If the actual matrix multiplication
///    should be done on smaller sizes than M, N and K, padding with zeros must be done beforehand.
///  - Enough units are present to perform the whole computation
#[cube]
pub trait TileMatmul<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size>:
    'static + Send + Sync
{
    /// Config for this matmul
    type Config: TileConfig;
    type Scope;

    /// Executes the matrix multiplication of Lhs and Rhs, adding the result to the accumulator
    fn execute(
        lhs: &Tilex<L, VL, Self::Scope, ReadWrite>,
        rhs: &Tilex<R, VR, Self::Scope, ReadWrite>,
        acc: &mut Tilex<A, VA, Self::Scope, ReadWrite>,
        #[comptime] config: Self::Config,
    );

    /// Create the container for Lhs
    ///
    /// # Safety
    ///
    /// This may point towards uninitialized memory.
    /// Make sure to call [load_lhs](TileMatmul::load_lhs) prior to [execute](TileMatmul::execute).
    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tilex<L, VL, Self::Scope, ReadWrite>;

    /// Load the container of Lhs from tile data
    fn load_lhs<E: Numeric, ES: Size>(
        tile: &Tilex<E, ES, Self::Scope, ReadOnly>,
        lhs: &mut Tilex<L, VL, Self::Scope, ReadWrite>,
        #[comptime] config: Self::Config,
    );

    /// Create the container for Rhs
    ///
    /// # Safety
    ///
    /// This may point towards uninitialized memory.
    /// Make sure to call [load_rhs](TileMatmul::load_rhs) prior to [execute](TileMatmul::execute).
    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tilex<R, VR, Self::Scope, ReadWrite>;

    /// Load the container of Rhs from tile data
    fn load_rhs<E: Numeric, ES: Size>(
        tile: &Tilex<E, ES, Self::Scope, ReadOnly>,
        rhs: &mut Tilex<R, VR, Self::Scope, ReadWrite>,
        #[comptime] config: Self::Config,
    );

    /// Allocate the container to receive the execution output.
    ///
    /// # Safety
    ///
    /// The output container must be initialized to some value (typically 0),
    /// because the execution adds to the already present value.
    /// Make sure to call [load_acc](TileMatmul::load_acc) prior to [execute](TileMatmul::execute).
    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tilex<A, VA, Self::Scope, ReadWrite>;

    /// Load the container of Acc from tile data
    fn load_acc<E: Numeric, ES: Size>(
        tile: &Tilex<E, ES, Self::Scope, ReadOnly>,
        acc: &mut Tilex<A, VA, Self::Scope, ReadWrite>,
        #[comptime] config: Self::Config,
    );

    /// Write the content of the output container to the given slice
    fn write_results<E: Numeric, ES: Size>(
        tile: &mut Tilex<E, ES, Self::Scope, ReadWrite>,
        out: &mut Tilex<A, VA, Self::Scope, ReadWrite>,
        #[comptime] config: Self::Config,
    );
}
