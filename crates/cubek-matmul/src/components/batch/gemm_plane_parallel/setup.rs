use cubecl::{
    CubeCount, CubeDim, Runtime,
    client::ComputeClient,
    ir::{AddressType, DeviceProperties},
    server::LaunchError,
};
use cubek_std::{MatrixLayout, cube_count::HypercubeBlueprint};

use crate::{
    components::{
        CubeDimResource,
        batch::{
            BatchMatmulFamily, CheckBounds,
            gemm_plane_parallel::{
                GemmPlaneParallel, GemmPlaneParallelConfig, KAccess, MatmulOperandLayouts, PlanesSplit,
                config::layout_for,
                matmul_entry,
            },
        },
        global::memory::GlobalLayoutConfig,
        stage::NumStages,
    },
    definition::{
        Blueprint, CubeMappingLaunch, MatmulElems, MatmulProblem, MatmulSetupError, MatmulTypes,
        MatmulVectorSizes, SwizzleModes, TilingScheme,
    },
    launch::*,
};

/// Plane-parallel GEMM family. Each plane reduces over `k` for a single
/// `(m, n)` output cell; cubes enumerate the `(m, n)` grid. The same family
/// also handles GEMV problems via [`MatmulKind`]: when one of `m, n` is 1,
/// the kernel collapses to the corresponding GEMV variant.
pub struct GemmPlaneParallelFamily {}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct GemmPlaneParallelBlueprint {
    pub dtypes: MatmulElems,
    pub num_planes: usize,
    // Should equal plane_dim * vector_size
    pub tile_dim: usize,
    pub hypercube_blueprint: HypercubeBlueprint,
    pub kind: MatmulOperandLayouts,
    pub planes_split: PlanesSplit,
    pub check_bounds: CheckBounds,
}

impl Blueprint for GemmPlaneParallelBlueprint {
    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: layout_for(self.kind.lhs, MatrixLayout::RowMajor),
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: layout_for(self.kind.rhs, MatrixLayout::ColMajor),
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn out_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn tiling_scheme(&self) -> TilingScheme {
        panic!("GemmPlaneParallel Blueprint doesn't have a TilingScheme")
    }

    fn swizzle_modes(&self) -> SwizzleModes {
        panic!("GemmPlaneParallel Blueprint doesn't have Swizzle Modes")
    }
}

impl BatchMatmulFamily<()> for GemmPlaneParallelFamily {
    type Matmul<MP: MatmulTypes> = GemmPlaneParallel<MP>;
    type Config = GemmPlaneParallelConfig;
    type Blueprint = GemmPlaneParallelBlueprint;

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &Self::Blueprint,
        _dtypes: &MatmulElems,
        _vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        Ok(GemmPlaneParallelConfig {
            plane_dim: device_props.hardware.plane_size_max,
            num_planes: blueprint.num_planes as u32,
            kind: blueprint.kind,
            planes_split: blueprint.planes_split,
            check_bounds: blueprint.check_bounds,
        })
    }

    fn num_stages() -> NumStages {
        (1, 1).into()
    }

    unsafe fn launch_unchecked<'a, MA: MatmulArgs<Config = ()>, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        address_type: AddressType,
        input: InputRuntimeArg<MA, R>,
        output: OutputRuntimeArg<MA, R>,
        _config: ConfigRuntimeArg<MA, R>,
        cube_mapping: CubeMappingLaunch<R>,
        blueprint: GemmPlaneParallelBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), LaunchError> {
        unsafe {
            matmul_entry::launch_unchecked::<MA, Lhs, LhsSize, Rhs, RhsSize, Acc, AccSize, R>(
                client,
                cube_count,
                cube_dim,
                address_type,
                input,
                output,
                (),
                cube_mapping,
                blueprint,
                [dtypes.lhs_global, dtypes.rhs_global, dtypes.acc_global],
                [vector_sizes.lhs, vector_sizes.rhs, vector_sizes.out],
            )
        };

        Ok(())
    }

    fn cubedim_resource(
        blueprint: &Self::Blueprint,
        _dtypes: &MatmulElems,
        _vector_sizes: &MatmulVectorSizes,
    ) -> Result<CubeDimResource, MatmulSetupError> {
        Ok(CubeDimResource::Planes(blueprint.num_planes as u32))
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        if vector_sizes.lhs != vector_sizes.rhs {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Lhs and Rhs vector sizes must be equal, got lhs:{:?}, rhs:{:?}",
                vector_sizes.lhs, vector_sizes.rhs
            ))));
        }

        if vector_sizes.out != 1 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Out vector size must be 1, got {:?}",
                vector_sizes.out,
            ))));
        }

        let plane_dim = client.properties().hardware.plane_size_max as usize;
        if blueprint.tile_dim != plane_dim * vector_sizes.lhs {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Tile dim must equal plane_dim * vector_size, got {:?} != {:?} * {:?}",
                blueprint.tile_dim, plane_dim, vector_sizes.lhs,
            ))));
        }

        if !problem.k.is_multiple_of(blueprint.tile_dim) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Problem dimension k={:?} must be divisible by tile dim ({:?})",
                problem.k, blueprint.tile_dim,
            ))));
        }

        // Re-derive the kind from the problem so a forced blueprint also gets
        // its layouts checked.
        let derived = MatmulOperandLayouts::from_problem(problem)?;
        if derived != blueprint.kind {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Blueprint kind {:?} disagrees with problem kind {:?}",
                blueprint.kind, derived
            ))));
        }

        // Per-side staged checks: each staged side must have its non-K
        // axis divisible by `tile_dim` and contributes a tile to shared
        // memory. The combined SM footprint is checked once.
        let traversal = blueprint.kind.k_traversal();
        let mut needed_shm = 0;
        if matches!(traversal.lhs, KAccess::Staged) {
            check_staged_divisibility(problem.m, blueprint.tile_dim, "m")?;
            needed_shm += blueprint.tile_dim
                * blueprint.tile_dim
                * dtypes.lhs_global.size()
                * vector_sizes.lhs;
        }
        if matches!(traversal.rhs, KAccess::Staged) {
            check_staged_divisibility(problem.n, blueprint.tile_dim, "n")?;
            needed_shm += blueprint.tile_dim
                * blueprint.tile_dim
                * dtypes.rhs_global.size()
                * vector_sizes.rhs;
        }
        let max_shm = client.properties().hardware.max_shared_memory_size;
        if needed_shm > max_shm {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Staged plane-parallel kernel needs {} bytes of shared memory, max {}",
                needed_shm, max_shm
            ))));
        }

        Ok(())
    }
}

fn check_staged_divisibility(
    non_k_dim: usize,
    tile_dim: usize,
    axis_label: &str,
) -> Result<(), MatmulSetupError> {
    if !non_k_dim.is_multiple_of(tile_dim) {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "Staged plane-parallel kernel needs problem.{} ({}) to be divisible by tile_dim ({})",
            axis_label, non_k_dim, tile_dim,
        ))));
    }
    Ok(())
}
