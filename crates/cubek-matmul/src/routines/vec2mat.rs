use std::{
    cmp::{max, min},
    fmt::Display,
};

use cubek_std::cube_count::{
    CubeCountPlan, CubeCountStrategy, GlobalOrder, GlobalOrderStrategy, HypercubeBlueprint,
};

use crate::{
    components::batch::{
        BatchMatmulFamily,
        vec2mat::{Vec2MatBlueprint, Vec2MatFamily},
    },
    definition::{MatmulElems, MatmulProblem, MatmulSetupError},
    routines::{BlueprintStrategy, DeviceSettings, ExpandInfo, LaunchInfo, Routine},
};

pub struct Vec2MatRoutine {}

#[derive(Default, Clone)]
pub struct Vec2MatStrategy {
    pub target_num_planes: usize,
}

impl Display for Vec2MatStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "_{}", self.target_num_planes)
    }
}

impl Routine<()> for Vec2MatRoutine {
    type Strategy = Vec2MatStrategy;
    type BatchMatmul = Vec2MatFamily;
    type Blueprint = <Self::BatchMatmul as BatchMatmulFamily<()>>::Blueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily<()>>::Config;

    fn expand_blueprint<R: cubecl::Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<(), Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let dtypes = MatmulElems::from_globals(&problem.global_dtypes);

        match strategy {
            BlueprintStrategy::Forced(blueprint) => Ok(ExpandInfo {
                blueprint: blueprint.clone(),
                dtypes,
            }),
            BlueprintStrategy::Inferred(strategy) => {
                let tile_dim =
                    device_settings.plane_dim as usize * device_settings.vector_sizes.rhs;
                let max_planes_for_swizzle = problem.k / tile_dim;
                let num_planes = max(1, min(strategy.target_num_planes, max_planes_for_swizzle));

                let blueprint = Vec2MatBlueprint {
                    dtypes: dtypes.clone(),
                    num_planes,
                    tile_dim,
                    hypercube_blueprint: HypercubeBlueprint::builder()
                        .cube_count_strategy(CubeCountStrategy::Flattened)
                        .global_order(
                            GlobalOrderStrategy::Fixed(GlobalOrder::RowMajor),
                            1,
                            (problem.n / (tile_dim * num_planes)) as u32,
                        )
                        .build(),
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

        let working_planes = problem.n.div_ceil(blueprint.tile_dim);
        let working_cubes = working_planes.div_ceil(blueprint.num_planes);

        let cube_count_plan = CubeCountPlan::from_blueprint(
            &blueprint.hypercube_blueprint,
            (1, working_cubes as u32, problem.num_batches() as u32).into(),
            &device_settings.max_cube_count,
        );
        println!("{:?}", device_settings.vector_sizes);

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
