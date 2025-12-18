use cubecl::{CubeDim, Runtime, client::ComputeClient, flex32, prelude::CubePrimitive, tf32};

use crate::{
    components::{
        CubeDimResource,
        global::{LoadSpecializationConfig, memory::GlobalLayoutConfig, read::ReaderMode},
        stage::{PartitionBuffering, SwizzleMode},
    },
    definition::{
        CubeCountPlan, MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSetupError, MatrixLayout,
        TilingScheme, hypercube::HypercubeBlueprint,
    },
    routines::DeviceSettings,
};
use std::{fmt::Debug, hash::Hash};

pub trait Blueprint: Debug + Clone + Eq + PartialEq + Hash {
    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig;
    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig;
    fn out_global_layout_config(&self) -> GlobalLayoutConfig;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TilingBlueprint {
    pub plane_dim: u32,
    pub tiling_scheme: TilingScheme,
    pub swizzle_modes: SwizzleModes,
    pub partition_buffering: PartitionBuffering,
    pub loading_precompute_strategy: LoadingPrecomputeStrategy,
    pub reader_mode: ReaderMode,
    pub load_specialization_config: LoadSpecializationConfig,
    pub hypercube_blueprint: HypercubeBlueprint,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
    pub line_sizes: MatmulLineSizes,
    pub check_m_bounds: bool,
    pub check_n_bounds: bool,
    pub check_k_bounds: bool,
    // TODO should eventually be removed because it's duplication
    pub dtypes: MatmulElems,
}

impl Blueprint for TilingBlueprint {
    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig {
        todo!()
    }

    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig {
        todo!()
    }

    fn out_global_layout_config(&self) -> GlobalLayoutConfig {
        todo!()
    }
}

/// Modifies the given matmul element types based on the kind of accelerator the kernel is run on.
pub fn adjust_dtypes<R: Runtime>(
    client: &ComputeClient<R>,
    dtypes: &mut MatmulElems,
    requires_accelerator: bool,
) {
    let f32_dtype = f32::as_type_native_unchecked();
    let flex_dtype = flex32::as_type_native_unchecked();
    let tf32_dtype = tf32::as_type_native_unchecked();
    let f16_dtype = half::f16::as_type_native_unchecked();

    if requires_accelerator {
        if *dtypes.lhs_global == f32_dtype
            && *dtypes.rhs_global == f32_dtype
            && client.properties().supports_type(tf32_dtype)
        {
            dtypes.lhs_stage.dtype = tf32_dtype;
            dtypes.rhs_stage.dtype = tf32_dtype;
            dtypes.lhs_register.dtype = tf32_dtype;
            dtypes.rhs_register.dtype = tf32_dtype;
        } else if *dtypes.lhs_global == flex_dtype
            && *dtypes.rhs_global == flex_dtype
            && client.properties().supports_type(f16_dtype)
        {
            dtypes.lhs_stage.dtype = f16_dtype;
            dtypes.rhs_stage.dtype = f16_dtype;
            dtypes.lhs_register.dtype = f16_dtype;
            dtypes.rhs_register.dtype = f16_dtype;
        }
    }
}

#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SwizzleModes {
    pub lhs: SwizzleMode,
    pub rhs: SwizzleMode,
    pub acc: SwizzleMode,
    pub out: SwizzleMode,
}

impl SwizzleModes {
    pub fn has_swizzle(&self) -> bool {
        self.lhs != SwizzleMode::None
            || self.rhs != SwizzleMode::None
            || self.acc != SwizzleMode::None
            || self.out != SwizzleMode::None
    }
}

impl TilingBlueprint {
    pub fn builder(tiling_scheme: TilingScheme, plane_dim: u32) -> TilingBlueprintBuilder {
        let hypercube_config = HypercubeBlueprint::builder(&tiling_scheme).build();
        TilingBlueprintBuilder::new()
            .tiling_scheme(tiling_scheme)
            .hypercube_config(hypercube_config)
            .plane_dim(plane_dim)
    }

    //     let plane_dim = device_settings.plane_dim;
    // let num_planes =
    //     Self::BatchMatmul::computation_resources()?.num_planes(plane_dim)?;
    // let cube_dim = CubeDim::new_2d(plane_dim, num_planes);
    // let cube_count_plan =

    pub fn cube_launch_info<R: Runtime>(
        &self,
        cubedim_resource: CubeDimResource,
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
    ) -> Result<(CubeDim, CubeCountPlan), MatmulSetupError> {
        let plane_dim = device_settings.plane_dim;
        let cube_dim = cubedim_resource.to_cube_dim(plane_dim)?;
        let cube_count_plan = CubeCountPlan::from_blueprint(
            &self.hypercube_blueprint,
            problem,
            &device_settings.max_cube_count,
        );

        Ok((cube_dim, cube_count_plan))
    }
}

pub struct TilingBlueprintBuilder {
    plane_dim: Option<u32>,
    pub tiling_scheme: Option<TilingScheme>,
    shared_swizzle: SwizzleModes,
    hypercube_selection: Option<HypercubeBlueprint>,
    partition_buffering: PartitionBuffering,
    loading_precompute_strategy: LoadingPrecomputeStrategy,
    reader_mode: ReaderMode,
    load_specialization_config: LoadSpecializationConfig,
}

impl TilingBlueprintBuilder {
    fn new() -> Self {
        Self {
            plane_dim: None,
            tiling_scheme: None,
            shared_swizzle: Default::default(),
            hypercube_selection: None,
            partition_buffering: PartitionBuffering::default(),
            loading_precompute_strategy: LoadingPrecomputeStrategy::default(),
            reader_mode: ReaderMode::default(),
            load_specialization_config: LoadSpecializationConfig::default(),
        }
    }

    pub fn plane_dim(mut self, plane_dim: u32) -> Self {
        self.plane_dim = Some(plane_dim);
        self
    }

    pub fn tiling_scheme(mut self, tiling_scheme: TilingScheme) -> Self {
        self.tiling_scheme = Some(tiling_scheme);
        self
    }

    pub fn shared_swizzle(mut self, swizzle: SwizzleModes) -> Self {
        self.shared_swizzle = swizzle;
        self
    }

    pub fn hypercube_config(mut self, hypercube_config: HypercubeBlueprint) -> Self {
        self.hypercube_selection = Some(hypercube_config);
        self
    }

    pub fn partition_buffering(mut self, partition_buffering: PartitionBuffering) -> Self {
        self.partition_buffering = partition_buffering;
        self
    }

    pub fn loading_precompute_strategy(
        mut self,
        loading_precompute_strategy: LoadingPrecomputeStrategy,
    ) -> Self {
        self.loading_precompute_strategy = loading_precompute_strategy;
        self
    }

    pub fn reader_mode(mut self, reader_mode: ReaderMode) -> Self {
        self.reader_mode = reader_mode;
        self
    }

    pub fn load_specialization_config(
        mut self,
        load_specialization_config: LoadSpecializationConfig,
    ) -> Self {
        self.load_specialization_config = load_specialization_config;
        self
    }

    pub fn build(self) -> TilingBlueprint {
        TilingBlueprint {
            plane_dim: self.plane_dim.unwrap(),
            tiling_scheme: self.tiling_scheme.unwrap(),
            swizzle_modes: self.shared_swizzle,
            hypercube_blueprint: self.hypercube_selection.unwrap(),
            partition_buffering: self.partition_buffering,
            loading_precompute_strategy: self.loading_precompute_strategy,
            reader_mode: self.reader_mode,
            load_specialization_config: self.load_specialization_config,
            lhs_layout: todo!(),
            rhs_layout: todo!(),
            line_sizes: todo!(),
            check_m_bounds: todo!(),
            check_n_bounds: todo!(),
            check_k_bounds: todo!(),
            dtypes: todo!()
        }
        // let check_k_bounds = !(problem.k as u32).is_multiple_of(stage_shape_k);
        // let check_m_bounds = !(problem.m as u32).is_multiple_of(stage_shape_m);
        // let check_n_bounds = !(problem.n as u32).is_multiple_of(stage_shape_n);
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub enum MultiRowStrategy {
    /// Always one row per plane
    #[default]
    Never,
    /// Always multiple rows per plane
    Always(u32),
    /// Uses multiple rows if the `m` dimension of the matmul implies at least the minimum number of stages along `m`
    Adaptive { minimum_stage_count: u32 },
}

#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum LoadingPrecomputeStrategy {
    /// Don't precompute anything in loading jobs
    #[default]
    Never,
    /// Precompute values that are shared across tasks
    Always,
}

impl From<LoadingPrecomputeStrategy> for bool {
    fn from(strategy: LoadingPrecomputeStrategy) -> Self {
        match strategy {
            LoadingPrecomputeStrategy::Always => true,
            LoadingPrecomputeStrategy::Never => false,
        }
    }
}
