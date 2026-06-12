pub mod launch;

use std::fmt::Display;

use cubecl::{CubeCount, CubeDim, Runtime, client::ComputeClient, ir::AddressType};
use cubek_std::cube_count::CubeCountPlan;

use crate::{
    args::{ConfigRuntimeArg, InputRuntimeArg, MatmulArgs, OutputRuntimeArg},
    components::{
        batch::{
            BatchMatmulFamily,
            naive::{NaiveBatchMatmulFamily, NaiveBlueprint},
        },
        stage::NumStages,
    },
    definition::{
        CubeMappingLaunch, MatmulAvailabilityError, MatmulElems, MatmulProblem, MatmulSetupError,
        MatmulVectorSizes,
    },
    routines::{
        BatchMatmulRoutine, BlueprintStrategy, DeviceSettings, ExpandInfo, LaunchInfo, Routine,
        batch_validate_blueprint,
    },
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
    type Blueprint = NaiveBlueprint;
}

impl BatchMatmulRoutine<()> for NaiveRoutine {
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
                <NaiveBatchMatmulFamily>::launch_unchecked::<MA, R>(
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
        batch_validate_blueprint::<NaiveBatchMatmulFamily, (), R>(
            client,
            blueprint,
            problem,
            dtypes,
            vector_sizes,
        )
    }

    fn num_stages() -> NumStages {
        NaiveBatchMatmulFamily::num_stages()
    }

    fn expand_blueprint<R: cubecl::Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        _strategy: &BlueprintStrategy<(), Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let dtypes = MatmulElems::from_globals(&problem.global_dtypes);
        let blueprint = NaiveBlueprint {
            vector_size_out: device_settings.vector_sizes.out as u32,
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
            &device_settings.vector_sizes,
        )?;

        let cube_dim = NaiveBatchMatmulFamily::cubedim_resource(
            &blueprint,
            &dtypes,
            &device_settings.vector_sizes,
        )?
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
            address_type: problem.address_type,
            vector_sizes: device_settings.vector_sizes,
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

    let cube_count_plan = CubeCountPlan::new_from_problem((m_cubes, n_cubes, batch_cubes).into());
    let max_cube_count = u16::MAX as u32;

    if m_cubes > max_cube_count || n_cubes > max_cube_count || batch_cubes > max_cube_count {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::CubeCountTooBig(cube_count_plan.resolve()),
        ));
    }

    Ok(cube_count_plan)
}
