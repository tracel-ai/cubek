use cubecl::{CubeDim, Runtime, client::ComputeClient, flex32, prelude::CubePrimitive, tf32};
use cubek_std::{
    MatrixLayout, SwizzleModes,
    cube_count::{Count3d, CubeCountPlan, HypercubeBlueprint},
};

use crate::{
    components::{
        CubeDimResource,
        global::{LoadFlows, memory::GlobalLayoutConfig, read::ReaderMode},
        stage::PartitionBuffering,
        tile::TileMatmulKind,
    },
    definition::{MatmulElems, MatmulProblem, MatmulSetupError, TilingScheme},
    routines::DeviceSettings,
};
use std::{fmt::Debug, hash::Hash};

pub trait Blueprint: Debug + Clone + Eq + PartialEq + Hash {
    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig;
    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig;
    fn out_global_layout_config(&self) -> GlobalLayoutConfig;

    // TODO Would be better to not have these methods but
    // otherwise it's hard to launch either as TMA or not
    fn tiling_scheme(&self) -> TilingScheme;
    fn swizzle_modes(&self) -> SwizzleModes;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchMatmulBlueprint {
    // TODO remove
    pub plane_dim: u32,
    pub tile_matmul: TileMatmulKind,
    pub tiling_scheme: TilingScheme,
    pub swizzle_modes: SwizzleModes,
    pub partition_buffering: PartitionBuffering,
    pub loading_precompute_strategy: LoadingPrecomputeStrategy,
    pub reader_mode: ReaderMode,
    pub load_flows: LoadFlows,
    pub hypercube_blueprint: HypercubeBlueprint,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
    pub check_m_bounds: bool,
    pub check_n_bounds: bool,
    pub check_k_bounds: bool,
}

impl Blueprint for BatchMatmulBlueprint {
    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: self.lhs_layout,
            check_row_bounds: self.check_m_bounds,
            check_col_bounds: self.check_k_bounds,
        }
    }

    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: self.rhs_layout,
            check_row_bounds: self.check_k_bounds,
            check_col_bounds: self.check_n_bounds,
        }
    }

    fn out_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: self.check_m_bounds,
            check_col_bounds: self.check_n_bounds,
        }
    }

    fn tiling_scheme(&self) -> TilingScheme {
        self.tiling_scheme
    }

    fn swizzle_modes(&self) -> SwizzleModes {
        self.swizzle_modes
    }
}

/// Modifies the given matmul element types based on the kind of accelerator the kernel is run on.
pub fn adjust_dtypes<R: Runtime>(
    client: &ComputeClient<R>,
    dtypes: &mut MatmulElems,
    requires_accelerator: bool,
) {
    let f32_dtype = f32::as_type_native_unchecked().storage_type();
    let flex_dtype = flex32::as_type_native_unchecked().storage_type();
    let tf32_dtype = tf32::as_type_native_unchecked().storage_type();
    let f16_dtype = half::f16::as_type_native_unchecked().storage_type();

    if requires_accelerator {
        if dtypes.lhs_global == f32_dtype
            && dtypes.rhs_global == f32_dtype
            && client.properties().supports_type(tf32_dtype)
        {
            dtypes.lhs_stage = tf32_dtype;
            dtypes.rhs_stage = tf32_dtype;
            dtypes.lhs_register = tf32_dtype;
            dtypes.rhs_register = tf32_dtype;
        } else if dtypes.lhs_global == flex_dtype
            && dtypes.rhs_global == flex_dtype
            && client.properties().supports_type(f16_dtype)
        {
            dtypes.lhs_stage = f16_dtype;
            dtypes.rhs_stage = f16_dtype;
            dtypes.lhs_register = f16_dtype;
            dtypes.rhs_register = f16_dtype;
        }
    }
}

impl BatchMatmulBlueprint {
    pub fn builder(
        tile_matmul: TileMatmulKind,
        tiling_scheme: TilingScheme,
        plane_dim: u32,
        problem: &MatmulProblem,
    ) -> BatchMatmulBlueprintBuilder {
        let hypercube_blueprint = HypercubeBlueprint::builder().build();

        BatchMatmulBlueprintBuilder {
            plane_dim,
            tile_matmul,
            tiling_scheme,
            hypercube_blueprint,
            m: problem.m as u32,
            n: problem.n as u32,
            k: problem.k as u32,
            lhs_layout: problem.lhs_layout,
            rhs_layout: problem.rhs_layout,
            shared_swizzle: Default::default(),
            stage_buffering: 1,
            partition_buffering: PartitionBuffering::default(),
            loading_precompute_strategy: LoadingPrecomputeStrategy::default(),
            reader_mode: ReaderMode::default(),
            load_specialization_config: LoadFlows::default(),
        }
    }

    pub fn cube_launch_info<R: Runtime>(
        &self,
        cubedim_resource: CubeDimResource,
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
    ) -> Result<(CubeDim, CubeCountPlan), MatmulSetupError> {
        let plane_dim = device_settings.plane_dim;
        let cube_dim = cubedim_resource.to_cube_dim(plane_dim)?;

        // The number of elements per global partition must be non-zero on every axis,
        // otherwise the `div_ceil` below panics with "attempt to divide by zero". A
        // zero here means the tiling scheme is degenerate (e.g. `stage_size == 0`);
        // reject it so autotune skips the candidate instead of crashing.
        let part_m = self.tiling_scheme.elements_per_global_partition_along_m();
        let part_n = self.tiling_scheme.elements_per_global_partition_along_n();
        let part_b = self.tiling_scheme.global_partition_size.batches;
        if part_m == 0 || part_n == 0 || part_b == 0 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Degenerate tiling scheme: elements per global partition is zero \
                 (m={part_m}, n={part_n}, batches={part_b}) for {:?}",
                self.tiling_scheme
            ))));
        }

        let target_cube_count = Count3d {
            x: (problem.m as u32).div_ceil(part_m),
            y: (problem.n as u32).div_ceil(part_n),
            z: (problem.num_batches() as u32).div_ceil(part_b),
        };
        let cube_count_plan = CubeCountPlan::from_blueprint(
            &self.hypercube_blueprint,
            target_cube_count,
            &device_settings.max_cube_count,
        );

        Ok((cube_dim, cube_count_plan))
    }
}

pub struct BatchMatmulBlueprintBuilder {
    plane_dim: u32,
    tile_matmul: TileMatmulKind,
    tiling_scheme: TilingScheme,

    m: u32,
    n: u32,
    k: u32,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,

    stage_buffering: u32,
    hypercube_blueprint: HypercubeBlueprint,

    shared_swizzle: SwizzleModes,
    partition_buffering: PartitionBuffering,
    loading_precompute_strategy: LoadingPrecomputeStrategy,
    reader_mode: ReaderMode,
    load_specialization_config: LoadFlows,
}

impl BatchMatmulBlueprintBuilder {
    pub fn hypercube_blueprint(mut self, hypercube_blueprint: HypercubeBlueprint) -> Self {
        self.hypercube_blueprint = hypercube_blueprint;
        self
    }

    pub fn shared_swizzle(mut self, swizzle: SwizzleModes) -> Self {
        self.shared_swizzle = swizzle;
        self
    }

    pub fn partition_buffering(mut self, partition_buffering: PartitionBuffering) -> Self {
        self.partition_buffering = partition_buffering;
        self
    }

    pub fn stage_buffering(mut self, stage_buffering: u32) -> Self {
        self.stage_buffering = stage_buffering;
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

    pub fn load_specialization_config(mut self, load_specialization_config: LoadFlows) -> Self {
        self.load_specialization_config = load_specialization_config;
        self
    }

    pub fn build(self) -> BatchMatmulBlueprint {
        let k_group = self.stage_buffering;

        let check_m_bounds = !self
            .m
            .is_multiple_of(self.tiling_scheme.elements_per_stage_along_m());
        let check_n_bounds = !self
            .n
            .is_multiple_of(self.tiling_scheme.elements_per_stage_along_n());
        let check_k_bounds = !self
            .k
            .is_multiple_of(self.tiling_scheme.elements_per_stage_along_k() * k_group);

        BatchMatmulBlueprint {
            plane_dim: self.plane_dim,
            tile_matmul: self.tile_matmul,
            tiling_scheme: self.tiling_scheme,
            swizzle_modes: self.shared_swizzle,
            hypercube_blueprint: self.hypercube_blueprint,
            partition_buffering: self.partition_buffering,
            loading_precompute_strategy: self.loading_precompute_strategy,
            reader_mode: self.reader_mode,
            load_flows: self.load_specialization_config,
            lhs_layout: self.lhs_layout,
            rhs_layout: self.rhs_layout,
            check_m_bounds,
            check_n_bounds,
            check_k_bounds,
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definition::MatmulElems;
    use cubecl::{ir::AddressType, zspace::shape};

    /// Regression for #434: multi-stage k-loops consume `stage_buffering` stages
    /// per iteration, so the k-bounds guard must cover the whole group.
    #[test]
    fn check_k_bounds_covers_stage_buffering() {
        let tiling_scheme = TilingScheme::builder()
            .with_tile_size((16, 16, 16).into())
            .with_partition_size((1, 1, 2).into())
            .with_stage_size((1, 1, 1).into())
            .build()
            .unwrap();
        assert_eq!(tiling_scheme.elements_per_stage_along_k(), 32);

        // k % 32 == 0 but k % 64 == 32: one stage fits, a stage pair does not.
        let problem = MatmulProblem::from_parameters(
            32,
            256,
            2848,
            shape![1],
            shape![1],
            MatrixLayout::RowMajor,
            MatrixLayout::RowMajor,
            MatrixLayout::RowMajor,
            None,
            None,
            MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems(),
            AddressType::default(),
        );

        let single =
            BatchMatmulBlueprint::builder(TileMatmulKind::Cmma, tiling_scheme, 32, &problem)
                .build();
        assert!(!single.check_k_bounds);

        let double =
            BatchMatmulBlueprint::builder(TileMatmulKind::Cmma, tiling_scheme, 32, &problem)
                .stage_buffering(2)
                .build();
        assert!(double.check_k_bounds);
    }
}
