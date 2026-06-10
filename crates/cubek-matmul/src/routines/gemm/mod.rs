pub mod launch;

use std::{
    cmp::{max, min},
    fmt::Display,
};

use cubecl::{CubeCount, CubeDim, Runtime, client::ComputeClient, ir::AddressType};
use cubek_std::cube_count::{CubeCountPlan, CubeCountStrategy, GlobalOrder, HypercubeBlueprint};

use crate::{
    args::{ConfigRuntimeArg, InputRuntimeArg, MatmulArgs, OutputRuntimeArg},
    components::{
        batch::{
            BatchMatmulFamily, CheckBounds,
            gemm::{GemmBlueprint, GemmFamily, MatmulOperandLayouts, PlanesSplit, Variant},
        },
        stage::NumStages,
    },
    definition::{
        CubeMappingLaunch, MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes,
    },
    routines::{
        BatchMatmulRoutine, BlueprintStrategy, DeviceSettings, ExpandInfo, LaunchInfo, Routine,
        batch_validate_blueprint, num_concurrent_planes,
    },
};

pub struct GemmRoutine {}

#[derive(Default, Clone)]
pub struct GemmStrategy {
    pub target_num_planes: Option<usize>,
}

impl Display for GemmStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "_{:?}", self.target_num_planes)
    }
}

/// Returns `(m_units, n_units)` — count of per-plane blocks along each
/// output axis for the chosen variant. Outer-product variants pack
/// `vector_size` cells per block along their natural-vector axis.
fn output_units(problem: &MatmulProblem, variant: Variant, vector_size: usize) -> (usize, usize) {
    match variant {
        Variant::Dot => (problem.m, problem.n),
        Variant::OuterNLhsContig | Variant::OuterNLhsStrided => {
            (problem.m, problem.n / vector_size)
        }
        Variant::OuterM => (problem.m / vector_size, problem.n),
    }
}

impl Routine<()> for GemmRoutine {
    type Strategy = GemmStrategy;
    type Blueprint = GemmBlueprint;
}

impl BatchMatmulRoutine<()> for GemmRoutine {
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    fn launch<MA: MatmulArgs<Config = ()>, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        address_type: AddressType,
        input: InputRuntimeArg<MA, R>,
        output: OutputRuntimeArg<MA, R>,
        config: ConfigRuntimeArg<MA, R>,
        cube_count_input: CubeMappingLaunch<R>,
        blueprint: Self::Blueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        {
            unsafe {
                <GemmFamily>::launch_unchecked::<MA, R>(
                    client,
                    cube_dim,
                    cube_count,
                    address_type,
                    input,
                    output,
                    config,
                    cube_count_input,
                    blueprint,
                    dtypes,
                    vector_sizes,
                )?
            }
            Ok(())
        }
    }

    #[allow(clippy::result_large_err)]
    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        batch_validate_blueprint::<GemmFamily, (), R>(
            client,
            blueprint,
            problem,
            dtypes,
            vector_sizes,
        )
    }

    fn num_stages() -> NumStages {
        GemmFamily::num_stages()
    }

    fn expand_blueprint<R: cubecl::Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<(), Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let dtypes = MatmulElems::from_globals(&problem.global_dtypes);
        let properties = device_settings.client.properties();

        match strategy {
            BlueprintStrategy::Forced(blueprint) => Ok(ExpandInfo {
                blueprint: blueprint.clone(),
                dtypes,
            }),
            BlueprintStrategy::Inferred(strategy) => {
                let target_num_planes = match strategy.target_num_planes {
                    Some(num_planes) => num_planes,
                    None => num_concurrent_planes(&properties.hardware),
                };

                let kind = MatmulOperandLayouts::from_problem(problem)?;
                let variant = kind.variant();
                let planes_split = variant.planes_split();
                let vector_size = device_settings.vector_sizes.lhs;

                let (m_units, n_units) = output_units(problem, variant, vector_size);
                let split_units = match planes_split {
                    PlanesSplit::M => m_units,
                    PlanesSplit::N => n_units,
                };
                let num_planes = max(1, min(target_num_planes, split_units));

                let check_bounds = if split_units.is_multiple_of(num_planes) {
                    CheckBounds::None
                } else {
                    CheckBounds::Terminate
                };

                let blueprint = GemmBlueprint {
                    dtypes: dtypes.clone(),
                    num_planes,
                    hypercube_blueprint: HypercubeBlueprint::builder()
                        .cube_count_strategy(CubeCountStrategy::Flattened)
                        .global_order(GlobalOrder::RowMajor)
                        .build(),
                    kind,
                    planes_split,
                    check_bounds,
                };

                Ok(ExpandInfo { blueprint, dtypes })
            }
        }
    }

    fn prepare<R: cubecl::Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        expand_info: ExpandInfo<Self::Blueprint>,
    ) -> Result<LaunchInfo<Self::Blueprint>, MatmulSetupError> {
        let ExpandInfo { blueprint, dtypes } = expand_info;

        Self::validate_blueprint(
            &device_settings.client,
            &blueprint,
            problem,
            &dtypes,
            &device_settings.vector_sizes,
        )?;

        let cube_dim =
            GemmFamily::cubedim_resource(&blueprint, &dtypes, &device_settings.vector_sizes)?
                .to_cube_dim(device_settings.plane_dim)?;

        let variant = blueprint.kind.variant();
        let vector_size = device_settings.vector_sizes.lhs;
        let (m_units, n_units) = output_units(problem, variant, vector_size);
        let (m_cubes, n_cubes) = match blueprint.planes_split {
            PlanesSplit::M => (
                m_units.div_ceil(blueprint.num_planes) as u32,
                n_units as u32,
            ),
            PlanesSplit::N => (
                m_units as u32,
                n_units.div_ceil(blueprint.num_planes) as u32,
            ),
        };

        let cube_count_plan = CubeCountPlan::from_blueprint(
            &blueprint.hypercube_blueprint,
            (m_cubes, n_cubes, problem.num_batches() as u32).into(),
            &device_settings.max_cube_count,
        );

        Ok(LaunchInfo {
            blueprint,
            dtypes,
            cube_dim,
            cube_count_plan,
            address_type: problem.address_type,
            vector_sizes: device_settings.vector_sizes,
        })
    }
}
