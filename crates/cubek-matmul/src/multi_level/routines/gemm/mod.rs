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
        num_concurrent_planes, outer_product_accumulators,
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
/// output axis for the chosen variant. A block is `accumulators` of the
/// variant's accumulators along its block axis and one cell along the other.
fn output_units(
    problem: &MatmulProblem,
    variant: Variant,
    planes_split: PlanesSplit,
    vector_size: usize,
    accumulators: usize,
) -> (usize, usize) {
    let block = variant.cells_per_accumulator(vector_size) * accumulators;
    match variant.block_axis(planes_split) {
        PlanesSplit::M => (problem.m / block, problem.n),
        PlanesSplit::N => (problem.m, problem.n / block),
    }
}

fn accumulators_per_plane(
    problem: &MatmulProblem,
    hardware: &HardwareProperties,
    dtypes: &MatmulElems,
    variant: Variant,
    planes_split: PlanesSplit,
    vector_size: usize,
) -> usize {
    // Each of Dot's accumulators reads its own K-contiguous line, so a block would split one
    // coalesced stream into as many streams as accumulators.
    if variant == Variant::Dot {
        return 1;
    }
    let extent = match variant.block_axis(planes_split) {
        PlanesSplit::M => problem.m,
        PlanesSplit::N => problem.n,
    };
    outer_product_accumulators(hardware, dtypes, vector_size, extent)
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
                let plan = |planes_split| {
                    let accumulators = accumulators_per_plane(
                        problem,
                        &properties.hardware,
                        &dtypes,
                        variant,
                        planes_split,
                        vector_size,
                    );
                    let (m_units, n_units) =
                        output_units(problem, variant, planes_split, vector_size, accumulators);
                    let split_units = match planes_split {
                        PlanesSplit::M => m_units,
                        PlanesSplit::N => n_units,
                    };
                    (planes_split, accumulators, split_units)
                };

                let preferred = plan(variant.planes_split());
                let crossed = plan(match preferred.0 {
                    PlanesSplit::M => PlanesSplit::N,
                    PlanesSplit::N => PlanesSplit::M,
                });
                // Planes are worker threads only on a CPU; a GPU keeps the variant's axis.
                let spreads_threads = properties.hardware.num_cpu_cores.is_some();
                let (planes_split, accumulators, split_units) =
                    if spreads_threads && crossed.2 > preferred.2 {
                        crossed
                    } else {
                        preferred
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
                    accumulators,
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
        let (m_units, n_units) = output_units(
            problem,
            variant,
            blueprint.planes_split,
            vector_size,
            blueprint.accumulators,
        );
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
    use cubecl::{
        frontend::Scalar,
        ir::{AddressType, VectorSize},
        zspace::shape,
    };
    use cubek_std::MatrixLayout;

    fn hardware(load_width: u32, vector_register_count: Option<u32>) -> HardwareProperties {
        HardwareProperties {
            load_width,
            vector_register_count,
            plane_size_min: 1,
            plane_size_max: 1,
            max_bindings: u32::MAX,
            max_shared_memory_size: 32 * 1024,
            max_cube_count: (u32::MAX, u32::MAX, u32::MAX),
            max_units_per_cube: 16,
            max_cube_dim: (16, 16, 16),
            num_streaming_multiprocessors: None,
            num_cpu_cores: Some(16),
            last_level_cache_size: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            max_vector_size: VectorSize::MAX,
            cube_mma_reserved_shared_memory: 0,
        }
    }

    fn vecmat(n: usize) -> (MatmulProblem, MatmulElems) {
        let dtypes = MatmulElems::from_single_dtype(f32::elem_type_native());
        let problem = MatmulProblem::from_parameters(
            1,
            n,
            4096,
            shape![1],
            shape![1],
            MatrixLayout::RowMajor,
            MatrixLayout::RowMajor,
            MatrixLayout::RowMajor,
            None,
            None,
            dtypes.as_global_elems(),
            AddressType::U32,
        );
        (problem, dtypes)
    }

    fn accumulators(hardware: &HardwareProperties, n: usize, variant: Variant) -> usize {
        let (problem, dtypes) = vecmat(n);
        accumulators_per_plane(&problem, hardware, &dtypes, variant, PlanesSplit::N, 8)
    }

    /// Thirteen of AVX2's sixteen registers are left beside an outer product's operands, and a
    /// block of eight is the widest power of two in them.
    #[test]
    fn an_avx2_plane_keeps_eight_f32_accumulators() {
        let avx2 = hardware(256, Some(16));
        assert_eq!(accumulators(&avx2, 4096, Variant::OuterN), 8);
    }

    /// A block of dot products reads one line per accumulator, which breaks the stream the
    /// single accumulator walks.
    #[test]
    fn a_dot_plane_keeps_one_accumulator() {
        assert_eq!(
            accumulators(&hardware(256, Some(16)), 4096, Variant::Dot),
            1
        );
    }

    /// A block overhanging its axis would write past the output.
    #[test]
    fn a_block_narrows_until_it_tiles_the_axis() {
        let avx2 = hardware(256, Some(16));
        assert_eq!(accumulators(&avx2, 48, Variant::OuterN), 2);
        assert_eq!(accumulators(&avx2, 8, Variant::OuterN), 1);
    }

    #[test]
    fn a_device_without_a_register_budget_keeps_one_accumulator() {
        assert_eq!(accumulators(&hardware(128, None), 4096, Variant::OuterN), 1);
    }
}
