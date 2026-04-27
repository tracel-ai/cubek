use std::{
    cmp::{max, min},
    fmt::Display,
};

use cubek_std::cube_count::{CubeCountPlan, CubeCountStrategy, GlobalOrder, HypercubeBlueprint};

use crate::{
    components::batch::{
        BatchMatmulFamily, CheckBounds,
        gemv_unit_perpendicular::{
            VecMatUnitPerpendicularBlueprint, VecMatUnitPerpendicularFamily,
        },
    },
    definition::{MatmulElems, MatmulProblem, MatmulSetupError},
    routines::{
        BlueprintStrategy, DeviceSettings, ExpandInfo, LaunchInfo, Routine, num_concurrent_planes,
    },
};

pub struct GemvUnitPerpendicularRoutine {}

#[derive(Default, Clone)]
pub struct GemvUnitPerpendicularStrategy {
    pub target_num_planes: Option<usize>,
}

impl Display for GemvUnitPerpendicularStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "_{:?}", self.target_num_planes)
    }
}

impl Routine<()> for GemvUnitPerpendicularRoutine {
    type Strategy = GemvUnitPerpendicularStrategy;
    type BatchMatmul = VecMatUnitPerpendicularFamily;
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
                let tile_dim =
                    device_settings.plane_dim as usize * device_settings.vector_sizes.rhs;
                let target_num_planes = match strategy.target_num_planes {
                    Some(num_planes) => num_planes,
                    None => num_concurrent_planes(&properties.hardware),
                };

                let max_planes_for_swizzle = problem.k.div_ceil(tile_dim);
                let num_planes = max(1, min(target_num_planes, max_planes_for_swizzle));

                let working_planes = problem.n.div_ceil(tile_dim);
                let aligned_n = problem.n.is_multiple_of(tile_dim);
                let aligned_k = problem.k.is_multiple_of(tile_dim);
                let check_bounds = if !aligned_n || !aligned_k {
                    // The last tile along n or k has OOB positions. Units must stay
                    // alive for plane_shuffle, so use checked reads/writes instead of
                    // terminating. OOB reads return zero (no contribution to acc).
                    CheckBounds::Checked
                } else if !working_planes.is_multiple_of(num_planes) {
                    // All work is fully valid, but some planes in the last cube are
                    // entirely idle and can be terminated.
                    CheckBounds::Terminate
                } else {
                    CheckBounds::None
                };

                let blueprint = VecMatUnitPerpendicularBlueprint {
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

        let working_planes = problem.n.div_ceil(blueprint.tile_dim);
        let working_cubes = working_planes.div_ceil(blueprint.num_planes);

        let cube_count_plan = CubeCountPlan::from_blueprint(
            &blueprint.hypercube_blueprint,
            (working_cubes as u32, 1, problem.num_batches() as u32).into(),
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
