use cubecl::{ir::DeviceProperties, prelude::*};

use crate::components::global::multi_stage::EventLoadingMode;
use crate::components::global::read::ReaderMode;
use crate::components::global::{
    GlobalWriterConfig, InputLoadFlow, LoadFlows, PlaneFlowConfig, SpecializedLoadingSides,
};
use crate::components::stage::{StageConfig, StageMemoryConfig};
use crate::components::{global::memory::GlobalMemoryConfig, stage::NumStages};
use crate::definition::StageIdent;
use crate::definition::TilingBlueprint;
use crate::definition::{AccG, MatmulSetupError};
use crate::definition::{LhsG, MatmulElems, MatmulLineSizes, RhsG};
use crate::definition::{MatmulPrecision, MatmulProblem};
use crate::{components::CubeDimResource, launch::RuntimeConfig};
use cubecl::std::{
    CubeOption,
    tensor::{View, layout::Coords2d},
};
use std::fmt::Debug;
use std::hash::Hash;

/// A family of [matmuls](GlobalMatmul) working with any [precision](MatmulPrecision).
pub trait GlobalMatmulFamily<RC: RuntimeConfig>: Send + Sync + 'static {
    /// The specific [GlobalMatmul] implementation associated with this family.
    type Matmul<MP: MatmulPrecision>: GlobalMatmul<RC, MP, Config = Self::Config>;

    /// The configuration type associated with this matmul family.
    type Config: GlobalConfig;

    /// Constructs the configuration based on the matmul problem, selection, and line sizes.
    ///
    /// This function may return an error if the configuration cannot be supported on the current runtime.
    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError>;

    fn num_stages() -> NumStages;

    /// Returns the compute resources required to run this matmul.
    fn cubedim_resource(
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        line_sizes: &MatmulLineSizes,
    ) -> Result<CubeDimResource, MatmulSetupError>;

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        line_sizes: &MatmulLineSizes,
    ) -> Result<(), MatmulSetupError>;
}

#[cube]
/// Provides matrix multiplication operations at the global level.
///
/// At the global level,
///  - Inputs are views over global memory, meaning access is given to
///    only parts of the global memory inputs at once.
///  - All planes within a Cube are used to solve the problem
///  - Dimensions M and N are fixed to an integer, but K is arbitrary large.
///    The matrix multiplication works only for size (M, _) · (_, N) = (M, N).
///    M and N should match the underlying Stage matmul's M and N.
///
/// # Assumptions
/// - Line sizes of the inputs evenly divide the dimension they are aligned with.
///
/// # Safety
///
/// It is not assumed that the matmul's dimensions match its inputs dimensions perfectly.
/// It is therefore important that Readers and Writers perform checks to avoid out-of-bounds
/// before reading data.
pub trait GlobalMatmul<RC: RuntimeConfig, MP: MatmulPrecision>: 'static + Send + Sync {
    type Config: GlobalConfig;

    /// Global reader for matrix A (Lhs)
    type LhsGlobalReader: CubeType;
    /// Global reader for matrix B (Rhs)
    type RhsGlobalReader: CubeType;
    /// Global reader for matrix C (Accumulator/Bias)
    type AccGlobalReader: CubeType;
    /// Writer to store the output stage into global memory
    type GlobalWriter: CubeType;

    /// The accumulator type for the tile matmul
    type Accumulators: CubeType;

    /// Performs the matrix multiplication over data loaded by the
    /// Lhs and Rhs readers, over the range given for K, and stores with
    /// using the output writer.
    ///
    /// To compute the whole range of k values, use k_range=(0, K) where
    /// K is the K dimension of Lhs and Rhs.
    fn execute(
        lhs_reader: Self::LhsGlobalReader,
        rhs_reader: Self::RhsGlobalReader,
        acc_reader: Self::AccGlobalReader,
        writer: Self::GlobalWriter,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    );

    /// Initialize the global reader for Lhs, starting at row m and column k
    fn init_lhs_global_reader(
        lhs: View<Line<LhsG<MP>>, Coords2d>,
        runtime_config: RC,
        #[comptime] config: Self::Config,
    ) -> Self::LhsGlobalReader;

    /// Initialize the global reader for Rhs, starting at row k and column n
    fn init_rhs_global_reader(
        rhs: View<Line<RhsG<MP>>, Coords2d>,
        runtime_config: RC,
        #[comptime] config: Self::Config,
    ) -> Self::RhsGlobalReader;

    /// Initialize the global reader for Rhs, starting at row k and column n
    fn init_acc_global_reader(
        acc: CubeOption<View<Line<AccG<MP>>, Coords2d>>,
        runtime_config: RC,
        #[comptime] config: Self::Config,
    ) -> Self::AccGlobalReader;

    /// Initialize the accumulator without data
    fn init_accumulators(#[comptime] config: Self::Config) -> Self::Accumulators;

    /// Initialize the global writer at row m and column n
    fn init_global_writer(
        out: View<Line<AccG<MP>>, Coords2d, ReadWrite>,
        #[comptime] config: Self::Config,
    ) -> Self::GlobalWriter;
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SharedGlobalMatmulConfig<S: StageConfig> {
    pub stage_config: S,
    pub num_planes: u32,
    pub lhs_reader_config: GlobalReaderConfig,
    pub rhs_reader_config: GlobalReaderConfig,
    pub acc_reader_config: GlobalReaderConfig,
    pub writer_config: GlobalWriterConfig,
    pub must_sync_plane_after_execution: bool,
}

impl<S: StageConfig> SharedGlobalMatmulConfig<S> {
    pub fn check_k_bounds(&self) -> bool {
        let from_lhs = self.lhs_reader_config.gmem_config.check_col_bounds;
        let from_rhs = self.rhs_reader_config.gmem_config.check_row_bounds;
        assert!(from_lhs == from_rhs);
        from_lhs
    }

    pub fn plane_dim(&self) -> u32 {
        self.stage_config.plane_dim()
    }

    pub fn plane_flow_config(&self) -> PlaneFlowConfig {
        self.stage_config.plane_flow_config()
    }

    pub fn specialized_loading_sides(&self) -> SpecializedLoadingSides {
        LoadFlows {
            lhs: self.lhs_reader_config.input_load_flow,
            rhs: self.rhs_reader_config.input_load_flow,
        }
        .into()
    }
}

impl<S: StageConfig> GlobalConfig for SharedGlobalMatmulConfig<S> {
    type StageConfig = S;

    fn stage_config(&self) -> Self::StageConfig {
        self.stage_config
    }

    fn lhs_reader_config(&self) -> GlobalReaderConfig {
        self.lhs_reader_config
    }

    fn rhs_reader_config(&self) -> GlobalReaderConfig {
        self.rhs_reader_config
    }

    fn cube_dim(&self) -> CubeDim {
        CubeDim::new_2d(self.plane_dim(), self.num_planes)
    }

    fn global_line_sizes(&self) -> MatmulLineSizes {
        MatmulLineSizes {
            lhs: self.lhs_reader_config.gmem_config.line_size,
            rhs: self.rhs_reader_config.gmem_config.line_size,
            out: self.writer_config.gmem_config.line_size,
        }
    }

    fn writer_config(&self) -> GlobalWriterConfig {
        self.writer_config
    }

    fn must_sync_plane_after_execution(&self) -> bool {
        self.must_sync_plane_after_execution
    }
}

/// Configuration for the [global matmul](GlobalMatmul) level.
pub trait GlobalConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    type StageConfig: StageConfig;

    /// Convert itself to the underlying stage matmul config
    fn stage_config(&self) -> Self::StageConfig;
    fn lhs_reader_config(&self) -> GlobalReaderConfig;
    fn rhs_reader_config(&self) -> GlobalReaderConfig;
    fn writer_config(&self) -> GlobalWriterConfig;
    fn cube_dim(&self) -> CubeDim;
    fn global_line_sizes(&self) -> MatmulLineSizes;
    fn must_sync_plane_after_execution(&self) -> bool;
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct GlobalReaderConfig {
    pub gmem_config: GlobalMemoryConfig,
    pub smem_config: StageMemoryConfig,
    pub precompute_job: bool,
    pub plane_dim: u32,
    pub reader_mode: ReaderMode,
    pub event_loading_mode: EventLoadingMode,
    pub input_load_flow: InputLoadFlow,
    pub plane_flow_config: PlaneFlowConfig,

    // ideally remove because doesn't apply to any kind of problem
    pub stage_ident: StageIdent,
}

impl GlobalReaderConfig {
    pub fn loading_planes_count(&self) -> u32 {
        self.smem_config.num_planes
    }

    pub fn loading_units_count(&self) -> u32 {
        self.plane_dim * self.loading_planes_count()
    }
}
