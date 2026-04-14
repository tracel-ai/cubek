use std::{
    cmp::{max, min},
    fmt::Display,
};

use cubek_std::cube_count::{CubeCountPlan, CubeCountStrategy, GlobalOrder, HypercubeBlueprint};

use crate::{
    components::batch::{
        BatchMatmulFamily, CheckBounds,
        gemv_plane_parallel::{GemvKind, GemvPlaneParallelBlueprint, GemvPlaneParallelFamily},
    },
    definition::{MatmulElems, MatmulProblem, MatmulSetupError},
    routines::{
        BlueprintStrategy, DeviceSettings, ExpandInfo, LaunchInfo, Routine, num_concurrent_planes,
    },
};

pub struct GemvPlaneParallelRoutine {}

#[derive(Default, Clone)]
pub struct GemvPlaneParallelStrategy {
    pub target_num_planes: Option<usize>,
}

impl Display for GemvPlaneParallelStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "_{:?}", self.target_num_planes)
    }
}

impl Routine<()> for GemvPlaneParallelRoutine {
    type Strategy = GemvPlaneParallelStrategy;
    type BatchMatmul = GemvPlaneParallelFamily;
    type Blueprint = <Self::BatchMatmul as BatchMatmulFamily<()>>::Blueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily<()>>::Config;

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

                let kind = GemvKind::from_problem(problem)?;
                let tile_dim =
                    device_settings.plane_dim as usize * device_settings.vector_sizes.rhs;
                let num_planes = match kind {
                    GemvKind::MatVecRowMajor | GemvKind::VecMatColMajor => {
                        // For tile swizzle
                        max(1, min(target_num_planes, problem.k / tile_dim))
                    }
                    GemvKind::VecMatRowMajor | GemvKind::MatVecColMajor => {
                        // For within tile
                        max(1, min(target_num_planes, tile_dim))
                    }
                };

                let num_parallel_problems = match kind {
                    GemvKind::VecMatColMajor => problem.n,
                    GemvKind::VecMatRowMajor => problem.n / tile_dim,
                    GemvKind::MatVecRowMajor => problem.m,
                    GemvKind::MatVecColMajor => problem.m / tile_dim,
                };
                let check_bounds = if num_parallel_problems.is_multiple_of(num_planes) {
                    CheckBounds::None
                } else {
                    CheckBounds::Terminate
                };

                let blueprint = GemvPlaneParallelBlueprint {
                    dtypes: dtypes.clone(),
                    num_planes,
                    tile_dim,
                    hypercube_blueprint: HypercubeBlueprint::builder()
                        .cube_count_strategy(CubeCountStrategy::Flattened)
                        .global_order(GlobalOrder::RowMajor)
                        .build(),
                    kind,
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

        let cube_dim = Self::BatchMatmul::cubedim_resource(
            &blueprint,
            &dtypes,
            &device_settings.vector_sizes,
        )?
        .to_cube_dim(device_settings.plane_dim)?;

        let num_parallel_problems = match blueprint.kind {
            GemvKind::VecMatColMajor => problem.n,
            GemvKind::VecMatRowMajor => problem.n / blueprint.tile_dim,
            GemvKind::MatVecRowMajor => problem.m,
            GemvKind::MatVecColMajor => problem.m / blueprint.tile_dim,
        };

        let working_cubes = num_parallel_problems.div_ceil(blueprint.num_planes);

        let cube_count_plan = CubeCountPlan::from_blueprint(
            &blueprint.hypercube_blueprint,
            (1, working_cubes as u32, problem.num_batches() as u32).into(),
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
