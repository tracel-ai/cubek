use std::fmt::Display;

use crate::{
    components::batch::{
        BatchMatmulFamily,
        naive::{NaiveBatchMatmulFamily, NaiveBlueprint},
    },
    definition::{
        CubeCountPlan, MatmulAvailabilityError, MatmulElems, MatmulProblem, MatmulSetupError,
    },
    routines::{BlueprintStrategy, DeviceSettings, ExpandInfo, LaunchInfo, Routine},
};

pub struct NaiveRoutine {}

#[derive(Default, Clone)]
pub struct NaiveStrategy {}

impl Display for NaiveStrategy {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl From<()> for NaiveStrategy {
    fn from(_value: ()) -> Self {
        Self {}
    }
}

impl Routine<()> for NaiveRoutine {
    type Strategy = NaiveStrategy;
    type BatchMatmul = NaiveBatchMatmulFamily;
    type Blueprint = <Self::BatchMatmul as BatchMatmulFamily<()>>::Blueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily<()>>::Config;

    fn expand_blueprint<R: cubecl::Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        _strategy: &BlueprintStrategy<(), Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let dtypes = MatmulElems::from_globals(&problem.global_dtypes);
        let blueprint = NaiveBlueprint {
            line_size_out: device_settings.line_sizes.out as u32,
            dtypes: dtypes.clone(),
        };
        Ok(ExpandInfo { blueprint, dtypes })
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
            &device_settings.line_sizes,
        )?;

        let cube_dim =
            Self::BatchMatmul::cubedim_resource(&blueprint, &dtypes, &device_settings.line_sizes)?
                .to_cube_dim(device_settings.plane_dim)?;

        Ok(LaunchInfo {
            blueprint,
            dtypes,
            cube_dim,
            cube_count_plan: simple_cube_count(
                &problem.lhs_shape,
                &problem.rhs_shape,
                &problem.out_shape,
                cube_dim.x,
                cube_dim.y,
            )?,
        })
    }
}

#[allow(clippy::result_large_err)]
fn simple_cube_count(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    output_shape: &[usize],
    cube_dim_x: u32,
    cube_dim_y: u32,
) -> Result<CubeCountPlan, MatmulSetupError> {
    let ndims = lhs_shape.len();
    let m = lhs_shape[ndims - 2];
    let n = rhs_shape[ndims - 1];

    let m_cubes = f32::ceil(m as f32 / cube_dim_x as f32) as u32;
    let n_cubes = f32::ceil(n as f32 / cube_dim_y as f32) as u32;
    let mut batch_cubes = 1u32;

    #[allow(clippy::needless_range_loop)]
    for i in 0..ndims - 2 {
        batch_cubes *= output_shape[i] as u32;
    }

    let cube_count_plan = CubeCountPlan::new_from_problem(m_cubes, n_cubes, batch_cubes);
    let max_cube_count = u16::MAX as u32;

    if m_cubes > max_cube_count || n_cubes > max_cube_count || batch_cubes > max_cube_count {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::CubeCountTooBig(cube_count_plan.resolve()),
        ));
    }

    Ok(cube_count_plan)
}
