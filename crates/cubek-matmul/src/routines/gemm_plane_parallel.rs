use std::{
    cmp::{max, min},
    fmt::Display,
};

use cubek_std::cube_count::{CubeCountPlan, CubeCountStrategy, GlobalOrder, HypercubeBlueprint};

use crate::{
    components::batch::{
        BatchMatmulFamily, CheckBounds,
        gemm_plane_parallel::{GemmPlaneParallelBlueprint, GemmPlaneParallelFamily},
    },
    definition::{MatmulElems, MatmulProblem, MatmulSetupError},
    routines::{
        BlueprintStrategy, DeviceSettings, ExpandInfo, LaunchInfo, Routine, num_concurrent_planes,
    },
};

pub struct GemmPlaneParallelRoutine {}

#[derive(Default, Clone)]
pub struct GemmPlaneParallelStrategy {
    pub target_num_planes: Option<usize>,
}

impl Display for GemmPlaneParallelStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "_{:?}", self.target_num_planes)
    }
}

impl Routine<()> for GemmPlaneParallelRoutine {
    type Strategy = GemmPlaneParallelStrategy;
    type BatchMatmul = GemmPlaneParallelFamily;
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

                let tile_dim =
                    device_settings.plane_dim as usize * device_settings.vector_sizes.rhs;

                // Each plane owns one n-column, so cap parallelism by n.
                let num_planes = max(1, min(target_num_planes, problem.n));

                let check_bounds = if problem.n.is_multiple_of(num_planes) {
                    CheckBounds::None
                } else {
                    CheckBounds::Terminate
                };

                let blueprint = GemmPlaneParallelBlueprint {
                    dtypes: dtypes.clone(),
                    num_planes,
                    tile_dim,
                    hypercube_blueprint: HypercubeBlueprint::builder()
                        .cube_count_strategy(CubeCountStrategy::Flattened)
                        .global_order(GlobalOrder::RowMajor)
                        .build(),
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

        // Cubes enumerate (m, n_groups) where each cube owns one m row and a
        // block of `num_planes` n-columns.
        let m_cubes = problem.m as u32;
        let n_cubes = problem.n.div_ceil(blueprint.num_planes) as u32;

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
