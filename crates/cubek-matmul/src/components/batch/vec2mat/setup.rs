use cubecl::{
    CubeCount, CubeDim, Runtime,
    client::ComputeClient,
    ir::{AddressType, DeviceProperties},
    quant::scheme::QuantLevel,
    server::LaunchError,
};
use cubek_std::{MatrixLayout, cube_count::HypercubeBlueprint};

use crate::{
    components::{
        CubeDimResource,
        batch::{
            BatchMatmulFamily,
            vec2mat::{Vec2Mat, Vec2MatMatmulConfig, matmul_entry},
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
pub struct Vec2MatFamily {}
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Vec2MatBlueprint {
    pub dtypes: MatmulElems,
    pub num_planes: usize,
    // Should equal plane_dim * vector_size
    pub tile_dim: usize,
    pub hypercube_blueprint: HypercubeBlueprint,
}

impl Blueprint for Vec2MatBlueprint {
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
        panic!("Vec2Mat Blueprint doesn't have a TilingScheme")
    }

    fn swizzle_modes(&self) -> SwizzleModes {
        panic!("Vec2Mat Blueprint doesn't have Swizzle Modes")
    }
}

impl BatchMatmulFamily<()> for Vec2MatFamily {
    type Matmul<MP: MatmulTypes> = Vec2Mat<MP>;
    type Config = Vec2MatMatmulConfig;
    type Blueprint = Vec2MatBlueprint;

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &Self::Blueprint,
        _dtypes: &MatmulElems,
        _vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        Ok(Vec2MatMatmulConfig {
            plane_dim: device_props.hardware.plane_size_max,
            num_planes: blueprint.num_planes as u32,
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
        blueprint: Vec2MatBlueprint,
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
        _client: &ComputeClient<R>,
        _blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        _dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        if let Some(scheme) = problem.lhs_scheme
            && let QuantLevel::Block(block_size) = scheme.level
        {
            let vector_size = vector_sizes.lhs * scheme.num_quants();
            let block_size = block_size.to_dim_vec(2.max(block_size.len()));
            let block_width = block_size[block_size.len() - 1] as usize;
            if !block_width.is_multiple_of(vector_size) {
                return Err(MatmulSetupError::InvalidConfig(Box::new(
                    "Block size isn't a multiple of load size on lhs",
                )));
            }
        }

        if let Some(scheme) = problem.rhs_scheme
            && let QuantLevel::Block(block_size) = scheme.level
        {
            let vector_size = vector_sizes.rhs * scheme.num_quants();
            let block_size = block_size.to_dim_vec(2.max(block_size.len()));
            let block_width = block_size[block_size.len() - 2] as usize;
            if !block_width.is_multiple_of(vector_size) {
                return Err(MatmulSetupError::InvalidConfig(Box::new(
                    "Block size isn't a multiple of load size on rhs",
                )));
            }
        }

        Ok(())
    }
}
