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
            gemm_outer_product::{
                GemmOuterProduct, GemmOuterProductConfig, MatmulOperandLayouts, PlanesSplit,
                Variant, config::layout_for, matmul_entry,
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

/// Outer-product GEMM family. CPU-only kernel (errors when launched on a
/// device with `plane_dim > 1`). Supports all four (lhs, rhs) layout
/// combinations: Row-Col uses a dot-product micro-kernel mirroring
/// `gemm_plane_parallel` so the two routines can be benchmarked head-to-head;
/// the other three layouts use broadcast-FMA outer-product micro-kernels
/// vectorized along the natural output axis (N for rhs RowMajor, M for
/// lhs ColMajor).
pub struct GemmOuterProductFamily {}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct GemmOuterProductBlueprint {
    pub dtypes: MatmulElems,
    pub num_planes: usize,
    pub hypercube_blueprint: HypercubeBlueprint,
    pub kind: MatmulOperandLayouts,
    pub planes_split: PlanesSplit,
    pub check_bounds: CheckBounds,
}

impl Blueprint for GemmOuterProductBlueprint {
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
        panic!("GemmOuterProduct Blueprint doesn't have a TilingScheme")
    }

    fn swizzle_modes(&self) -> SwizzleModes {
        panic!("GemmOuterProduct Blueprint doesn't have Swizzle Modes")
    }
}

impl BatchMatmulFamily<()> for GemmOuterProductFamily {
    type Matmul<MP: MatmulTypes> = GemmOuterProduct<MP>;
    type Config = GemmOuterProductConfig;
    type Blueprint = GemmOuterProductBlueprint;

    fn expand_config(
        _device_props: &DeviceProperties,
        blueprint: &Self::Blueprint,
        _dtypes: &MatmulElems,
        _vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        Ok(GemmOuterProductConfig {
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
        blueprint: GemmOuterProductBlueprint,
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
        _dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        let plane_dim = client.properties().hardware.plane_size_max as usize;
        if plane_dim > 1 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "GemmOuterProduct is CPU-only (plane_dim must be 1, got {})",
                plane_dim,
            ))));
        }

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

        let vs = vector_sizes.lhs;
        if !problem.k.is_multiple_of(vs) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Problem dimension k={:?} must be divisible by vector_size ({:?})",
                problem.k, vs,
            ))));
        }

        // Outer-product variants need divisibility on the block axis so
        // each plane handles a full vector_size-wide slice. Row-Col is
        // 1×1 per plane and has no such constraint.
        match blueprint.kind.variant() {
            Variant::Dot => {}
            Variant::OuterNLhsContig | Variant::OuterNLhsStrided => {
                if !problem.n.is_multiple_of(vs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "OuterN variants need n ({}) divisible by vector_size ({})",
                        problem.n, vs,
                    ))));
                }
            }
            Variant::OuterM => {
                if !problem.m.is_multiple_of(vs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "OuterM variant needs m ({}) divisible by vector_size ({})",
                        problem.m, vs,
                    ))));
                }
            }
        }

        let derived = MatmulOperandLayouts::from_problem(problem)?;
        if derived != blueprint.kind {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Blueprint kind {:?} disagrees with problem kind {:?}",
                blueprint.kind, derived
            ))));
        }

        Ok(())
    }
}
