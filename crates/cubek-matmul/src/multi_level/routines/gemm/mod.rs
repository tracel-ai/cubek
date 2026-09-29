pub mod launch;

use std::{
    cmp::{max, min},
    fmt::Display,
};

use cubecl::{
    CubeCount, CubeDim,
    client::Client,
    ir::{AddressType, HardwareProperties},
};
use cubek_std::cube_count::{CubeCountPlan, CubeCountStrategy, GlobalOrder, HypercubeBlueprint};

use crate::{
    definition::{MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes},
    multi_level::{
        BatchMatmulRoutine, ExpandInfo, LaunchInfo,
        args::{ConfigRuntimeArg, InputRuntimeArg, MatmulArgs, OutputRuntimeArg},
        batch_validate_blueprint,
        components::{
            batch::{
                BatchMatmulFamily, CheckBounds,
                gemm::{GemmBlueprint, GemmFamily, MatmulOperandLayouts, PlanesSplit, Variant},
            },
            stage::NumStages,
        },
        definition::CubeMappingLaunch,
        num_concurrent_planes,
    },
    routine::{BlueprintStrategy, DeviceSettings, Routine},
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

/// Returns `(m_units, n_units)`: count of per-plane blocks along each
/// output axis for the chosen variant. Outer-product variants pack
/// `vector_size` cells per block along their natural-vector axis.
fn output_units(problem: &MatmulProblem, variant: Variant, vector_size: usize) -> (usize, usize) {
    match variant {
        Variant::Dot => (problem.m, problem.n),
        Variant::OuterN => (problem.m, problem.n / vector_size),
        Variant::OuterM => (problem.m / vector_size, problem.n),
    }
}

/// The output axis a cube's planes step across: the variant's own, unless it leaves some of a CPU's
/// planes, which are its worker threads, without a block while the other axis holds more.
fn split_axis(
    variant: Variant,
    (m_units, n_units): (usize, usize),
    target_num_planes: usize,
    hardware: &HardwareProperties,
) -> PlanesSplit {
    let units = |split| match split {
        PlanesSplit::M => m_units,
        PlanesSplit::N => n_units,
    };
    let preferred = variant.planes_split();
    let crossed = match preferred {
        PlanesSplit::M => PlanesSplit::N,
        PlanesSplit::N => PlanesSplit::M,
    };
    let idles_threads = hardware.num_cpu_cores.is_some() && units(preferred) < target_num_planes;
    if idles_threads && units(crossed) > units(preferred) {
        crossed
    } else {
        preferred
    }
}

impl Routine<()> for GemmRoutine {
    type Strategy = GemmStrategy;
    type Blueprint = GemmBlueprint;
}

impl BatchMatmulRoutine<()> for GemmRoutine {
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    fn launch<MA: MatmulArgs<Config = ()>>(
        client: &Client,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        address_type: AddressType,
        input: InputRuntimeArg<MA>,
        output: OutputRuntimeArg<MA>,
        config: ConfigRuntimeArg<MA>,
        cube_count_input: CubeMappingLaunch,
        blueprint: Self::Blueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        {
            unsafe {
                <GemmFamily>::launch_unchecked::<MA>(
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
    fn validate_blueprint(
        client: &Client,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        batch_validate_blueprint::<GemmFamily, ()>(client, blueprint, problem, dtypes, vector_sizes)
    }

    fn num_stages() -> NumStages {
        GemmFamily::num_stages()
    }

    fn expand_blueprint(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings,
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

                let variant = MatmulOperandLayouts::from_problem(problem)?.variant()?;
                let vector_size = device_settings.vector_sizes.lhs;

                let (m_units, n_units) = output_units(problem, variant, vector_size);
                let planes_split = split_axis(
                    variant,
                    (m_units, n_units),
                    target_num_planes,
                    &properties.hardware,
                );
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
                    variant,
                    planes_split,
                    check_bounds,
                };

                Ok(ExpandInfo { blueprint, dtypes })
            }
        }
    }

    fn prepare(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings,
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

        let variant = blueprint.variant;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> HardwareProperties {
        HardwareProperties {
            load_width: 256,
            vector_register_count: Some(16),
            plane_size_min: 1,
            plane_size_max: 1,
            max_bindings: u32::MAX,
            max_shared_memory_size: 48 * 1024,
            max_cube_count: (u32::MAX, u32::MAX, u32::MAX),
            max_units_per_cube: 16,
            max_cube_dim: (16, 16, 16),
            num_streaming_multiprocessors: None,
            num_cpu_cores: Some(16),
            last_level_cache_size: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            max_vector_size: usize::MAX,
            cube_mma_reserved_shared_memory: 0,
        }
    }

    fn gpu() -> HardwareProperties {
        HardwareProperties {
            vector_register_count: None,
            plane_size_min: 32,
            plane_size_max: 32,
            max_units_per_cube: 1024,
            max_cube_dim: (1024, 1024, 64),
            num_cpu_cores: None,
            ..cpu()
        }
    }

    fn split_axis_on(
        hardware: &HardwareProperties,
        variant: Variant,
        units: (usize, usize),
    ) -> PlanesSplit {
        split_axis(variant, units, num_concurrent_planes(hardware), hardware)
    }

    #[test]
    fn a_cpu_spreads_a_single_row_across_its_columns() {
        assert_eq!(
            split_axis_on(&cpu(), Variant::OuterN, (1, 256)),
            PlanesSplit::N
        );
    }

    #[test]
    fn a_cpu_spreads_a_single_column_across_its_rows() {
        assert_eq!(
            split_axis_on(&cpu(), Variant::Dot, (4096, 1)),
            PlanesSplit::M
        );
        assert_eq!(
            split_axis_on(&cpu(), Variant::OuterM, (256, 1)),
            PlanesSplit::M
        );
    }

    #[test]
    fn a_cpu_keeps_a_variant_axis_with_a_block_for_every_thread() {
        assert_eq!(
            split_axis_on(&cpu(), Variant::OuterN, (64, 1024)),
            PlanesSplit::M
        );
    }

    #[test]
    fn a_gpu_keeps_the_variant_axis() {
        assert_eq!(
            split_axis_on(&gpu(), Variant::Dot, (4096, 1)),
            PlanesSplit::N
        );
        assert_eq!(
            split_axis_on(&gpu(), Variant::OuterN, (1, 256)),
            PlanesSplit::M
        );
    }
}
