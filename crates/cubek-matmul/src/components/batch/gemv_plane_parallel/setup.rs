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
            BatchMatmulFamily,
            gemv_plane_parallel::{
                GemvKind, VecMatPlaneParallel, VecMatPlaneParallelConfig, matmul_entry,
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

/// Simple partitioned batch matmul family for any precision
pub struct GemvPlaneParallelFamily {}
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct GemvPlaneParallelBlueprint {
    pub dtypes: MatmulElems,
    pub num_planes: usize,
    // Should equal plane_dim * vector_size
    pub tile_dim: usize,
    pub hypercube_blueprint: HypercubeBlueprint,
    pub kind: GemvKind,
}

impl Blueprint for GemvPlaneParallelBlueprint {
    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::ColMajor,
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
        panic!("VecMatPlaneParallel Blueprint doesn't have a TilingScheme")
    }

    fn swizzle_modes(&self) -> SwizzleModes {
        panic!("VecMatPlaneParallel Blueprint doesn't have Swizzle Modes")
    }
}

impl BatchMatmulFamily<()> for GemvPlaneParallelFamily {
    type Matmul<MP: MatmulTypes> = VecMatPlaneParallel<MP>;
    type Config = VecMatPlaneParallelConfig;
    type Blueprint = GemvPlaneParallelBlueprint;

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &Self::Blueprint,
        _dtypes: &MatmulElems,
        _vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        Ok(VecMatPlaneParallelConfig {
            plane_dim: device_props.hardware.plane_size_max,
            num_planes: blueprint.num_planes as u32,
            plan: blueprint.kind,
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
        blueprint: GemvPlaneParallelBlueprint,
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
                "Problem dimensions k={:?} must be divisible by tile dim ({:?})",
                problem.k, blueprint.tile_dim,
            ))));
        }

        match blueprint.kind {
            GemvKind::VecMatRowMajor => {
                if !problem.n.is_multiple_of(blueprint.tile_dim) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "For VecMatTransposeSwap, problem.n ({:?}) must be divisible by tile_dim ({:?})",
                        problem.n, blueprint.tile_dim,
                    ))));
                }

                if blueprint.tile_dim
                    * blueprint.tile_dim
                    * dtypes.rhs_global.size()
                    * vector_sizes.rhs
                    > client.properties().hardware.max_shared_memory_size
                {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Requesting too much shared memory, requested {:?}, max {:?}",
                        blueprint.tile_dim
                            * blueprint.tile_dim
                            * dtypes.rhs_global.size()
                            * vector_sizes.rhs,
                        client.properties().hardware.max_shared_memory_size
                    ))));
                }
            }
            GemvKind::MatVecColMajor => {
                if !problem.m.is_multiple_of(blueprint.tile_dim) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "For MatVecTransposeSwap, problem.m ({:?}) must be divisible by tile_dim ({:?})",
                        problem.m, blueprint.tile_dim,
                    ))));
                }

                if blueprint.tile_dim
                    * blueprint.tile_dim
                    * dtypes.lhs_global.size()
                    * vector_sizes.lhs
                    > client.properties().hardware.max_shared_memory_size
                {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Requesting too much shared memory, requested {:?}, max {:?}",
                        blueprint.tile_dim
                            * blueprint.tile_dim
                            * dtypes.lhs_global.size()
                            * vector_sizes.lhs,
                        client.properties().hardware.max_shared_memory_size
                    ))));
                }
            }
            _ => {}
        }
        Ok(())
    }
}
