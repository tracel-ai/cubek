use cubecl::ir::HardwareProperties;
use cubecl::prelude::*;
use cubek_std::cube_count::CubeCountPlan;

use crate::{
    definition::{MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes},
    multi_level::{
        args::{ConfigRuntimeArg, InputRuntimeArg, MatmulArgs, OutputRuntimeArg, RuntimeConfig},
        components::{
            CubeDimResource, batch::BatchMatmulFamily, global::cube_dim_validation,
            stage::NumStages,
        },
        definition::{BatchMatmulBlueprint, Blueprint, CubeMappingLaunch},
    },
    routine::{BlueprintStrategy, DeviceSettings, Routine},
};

/// The launch pipeline for matmuls with a batch matmul (might become legacy)
pub trait BatchMatmulRoutine<RC: RuntimeConfig>: Routine<RC, Blueprint: Blueprint> {
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    fn launch<MA: MatmulArgs<Config = RC>>(
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
    ) -> Result<(), MatmulSetupError>;

    fn expand_blueprint(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError>;

    fn prepare(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings,
        expand_info: ExpandInfo<Self::Blueprint>,
    ) -> Result<LaunchInfo<Self::Blueprint>, MatmulSetupError>;

    fn num_stages() -> NumStages;

    fn device_settings(client: &Client, vector_sizes: MatmulVectorSizes) -> DeviceSettings {
        // Sometimes the GPU doesn't support plane instructions and doesn't report the
        // plane size, but we can still execute algorithms that don't use plane instructions.
        //
        // In this case, we set a plane size for the selector to work, defaulting to 32 as it
        // is a common plane size.
        let plane_dim = match client.properties().hardware.plane_size_max {
            0 => 32,
            plane_dim => plane_dim,
        };

        DeviceSettings {
            client: client.clone(),
            plane_dim,
            vector_sizes,
            max_cube_count: client.properties().hardware.max_cube_count,
        }
    }

    #[allow(clippy::result_large_err)]
    fn validate_blueprint(
        client: &Client,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError>;
}

/// Validate a blueprint against a batch-matmul family `F`. Routines delegate here from
/// their [`BatchMatmulRoutine::validate_blueprint`].
#[allow(clippy::result_large_err)]
pub fn batch_validate_blueprint<F, RC>(
    client: &Client,
    blueprint: &F::Blueprint,
    problem: &MatmulProblem,
    dtypes: &MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> Result<(), MatmulSetupError>
where
    RC: RuntimeConfig,
    F: BatchMatmulFamily<RC>,
{
    F::validate_blueprint(client, blueprint, problem, dtypes, vector_sizes)
}

#[derive(Debug)]
pub struct ExpandInfo<B: Blueprint> {
    pub blueprint: B,
    pub dtypes: MatmulElems,
}

#[derive(Debug)]
pub struct LaunchInfo<B: Blueprint> {
    pub blueprint: B,
    pub dtypes: MatmulElems,
    pub vector_sizes: MatmulVectorSizes,
    pub cube_dim: CubeDim,
    pub cube_count_plan: CubeCountPlan,
    pub address_type: AddressType,
}

impl LaunchInfo<BatchMatmulBlueprint> {
    pub fn new(
        blueprint: BatchMatmulBlueprint,
        dtypes: MatmulElems,
        problem: &MatmulProblem,
        compute_resources: CubeDimResource,
        device_settings: &DeviceSettings,
    ) -> Result<Self, MatmulSetupError> {
        let (cube_dim, cube_count_plan) =
            blueprint.cube_launch_info(compute_resources, problem, device_settings)?;
        cube_dim_validation(cube_dim)?;

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

pub(crate) fn num_concurrent_planes(properties: &HardwareProperties) -> usize {
    match properties.num_cpu_cores {
        Some(num_cores) => num_cores as usize,
        // We use the number of conccurrent planes that can work per SM at the same time on GPUs.
        //
        // This is typically the number of warp scheduler on Nvidia or the number of SIMD units
        // per CU on AMD.
        None => 4,
    }
}

/// The K-vector, its broadcast scalar and the accumulator's line an outer product holds in
/// registers beside its accumulators.
const OUTER_OPERAND_VECTORS: usize = 3;

/// How many `vector_size`-wide accumulators an outer-product plane keeps side by side along an
/// output axis of `extent` cells: as many as fit the registers beside the operands, as a power of
/// two whose block tiles the axis. Each is an independent latency chain, and all of them read
/// adjacent segments of one line, so throughput grows with the count until they spill.
pub(crate) fn outer_product_accumulators(
    hardware: &HardwareProperties,
    dtypes: &MatmulElems,
    vector_size: usize,
    extent: usize,
) -> usize {
    let Some(registers) = hardware.vector_registers(dtypes.acc_register.size()) else {
        return 1;
    };
    let reserved = OUTER_OPERAND_VECTORS * registers.registers_for(vector_size);
    let fitting = registers.vectors_fitting(vector_size, reserved).max(1);

    let mut accumulators = 1 << fitting.ilog2();
    while accumulators > 1 && !extent.is_multiple_of(accumulators * vector_size) {
        accumulators /= 2;
    }
    accumulators
}
