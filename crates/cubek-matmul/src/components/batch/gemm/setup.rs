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
            gemm::{
                Gemm, GemmConfig, MatmulOperandLayouts, PlanesSplit, Variant, config::layout_for,
                matmul::matmul_entry,
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

/// Unified GEMM family. Selects a kernel variant from operand layouts:
/// `Dot` (Row-Col) supports any `plane_dim` (plane-cooperative reduction
/// over K); `OuterM` / `OuterN` are CPU-only (require `plane_dim == 1`).
/// Also handles GEMV when one of `m`, `n` is 1 — the vector side is
/// classified by `OperandLayout::Vector` and uses a layout-appropriate
/// variant.
pub struct GemmFamily {}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct GemmBlueprint {
    pub dtypes: MatmulElems,
    pub num_planes: usize,
    pub hypercube_blueprint: HypercubeBlueprint,
    pub kind: MatmulOperandLayouts,
    pub planes_split: PlanesSplit,
    pub check_bounds: CheckBounds,
}

impl Blueprint for GemmBlueprint {
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
        panic!("Gemm Blueprint doesn't have a TilingScheme")
    }

    fn swizzle_modes(&self) -> SwizzleModes {
        panic!("Gemm Blueprint doesn't have Swizzle Modes")
    }
}

impl BatchMatmulFamily<()> for GemmFamily {
    type Matmul<MP: MatmulTypes> = Gemm<MP>;
    type Config = GemmConfig;
    type Blueprint = GemmBlueprint;

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &Self::Blueprint,
        _dtypes: &MatmulElems,
        _vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        Ok(GemmConfig {
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

    unsafe fn launch_unchecked<MA: MatmulArgs<Config = ()>, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        address_type: AddressType,
        input: InputRuntimeArg<MA, R>,
        output: OutputRuntimeArg<MA, R>,
        _config: ConfigRuntimeArg<MA, R>,
        cube_mapping: CubeMappingLaunch<R>,
        blueprint: GemmBlueprint,
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
        let variant = blueprint.kind.variant();

        // Per-variant constraints. Dot supports plane-cooperative K reduction;
        // OuterM/OuterN are CPU-only because they don't reduce across units.
        match variant {
            Variant::Dot => {
                let tile_dim = plane_dim * vs;
                if !problem.k.is_multiple_of(tile_dim) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Problem dimension k={:?} must be divisible by plane_dim * vector_size ({:?})",
                        problem.k, tile_dim,
                    ))));
                }
            }
            Variant::OuterNLhsContig | Variant::OuterNLhsStrided => {
                if plane_dim > 1 {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "OuterN variants require plane_dim == 1 (CPU-only), got {}",
                        plane_dim,
                    ))));
                }
                if !problem.k.is_multiple_of(vs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Problem dimension k={:?} must be divisible by vector_size ({:?})",
                        problem.k, vs,
                    ))));
                }
                if !problem.n.is_multiple_of(vs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "OuterN variants need n ({}) divisible by vector_size ({})",
                        problem.n, vs,
                    ))));
                }
            }
            Variant::OuterM => {
                if plane_dim > 1 {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "OuterM variant requires plane_dim == 1 (CPU-only), got {}",
                        plane_dim,
                    ))));
                }
                if !problem.k.is_multiple_of(vs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Problem dimension k={:?} must be divisible by vector_size ({:?})",
                        problem.k, vs,
                    ))));
                }
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
