use crate::{
    components::{
        args::{ReduceArgs, TensorArgs, init_tensors},
        instructions::*,
    },
    launch::{ReduceLaunchInfo, ReduceStrategy},
    routines::{
        CubeReduceBlueprint, GlobalReduceBlueprint, PlaneReduceBlueprint, ReduceBlueprint,
        reduce_kernel_virtual,
    },
};
use cubecl::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct ReduceDtypes {
    pub input: StorageType,
    pub output: StorageType,
    pub accumulation: StorageType,
}

/// Launch a reduce kernel. This function assumes that all parameters are already validated.
/// See the main entrypoint `reduce` in `lib.rs` for an example how to call this function
/// with the appropriate assumptions.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_reduce<Run: Runtime>(
    client: &ComputeClient<Run>,
    input: TensorHandleRef<Run>,
    output: TensorHandleRef<Run>,
    axis: u32,
    info: ReduceLaunchInfo,
    strategy: ReduceStrategy,
    dtypes: ReduceDtypes,
    inst: ReduceOperationConfig,
) -> Result<(), LaunchError> {
    let routine = match strategy {
        ReduceStrategy::FullUnit => {
            GlobalReduceBlueprint::FullUnit(crate::routines::UnitReduceBlueprint {
                // TODO: Maybe faster to shotdown planes and do branchless check bound.
                unit_idle: info.idle,
            })
        }
        ReduceStrategy::FullPlane { independant } => {
            GlobalReduceBlueprint::FullPlane(PlaneReduceBlueprint {
                bound_checks: info.bound_checks,
                independant,
                plane_idle: info.idle,
            })
        }
        ReduceStrategy::FullCube { use_planes } => match use_planes {
            true => GlobalReduceBlueprint::Cube(CubeReduceBlueprint {
                num_shared_accumulators: info.cube_dim.y,
                bound_checks_inner: info.bound_checks,
                use_planes,
            }),
            false => GlobalReduceBlueprint::Cube(CubeReduceBlueprint {
                num_shared_accumulators: info.cube_dim.num_elems(),
                bound_checks_inner: info.bound_checks,
                use_planes,
            }),
        },
    };

    let blueprint = ReduceBlueprint {
        line_mode: info.line_mode,
        global: routine,
    };

    unsafe {
        reduce_kernel::launch_unchecked::<TensorArgs, Run>(
            client,
            info.cube_count,
            info.cube_dim,
            input.as_tensor_arg(info.line_size_input as u8),
            output.as_tensor_arg(info.line_size_output as u8),
            ScalarArg::new(axis),
            blueprint,
            inst,
            dtypes.input,
            dtypes.output,
            dtypes.accumulation,
        )
    }
}

#[cube(launch_unchecked)]
pub fn reduce_kernel<In: Numeric, Out: Numeric, Acc: Numeric, RA: ReduceArgs>(
    input: &RA::Input<In>,
    output: &mut RA::Output<Out>,
    axis_reduce: u32,
    #[comptime] blueprint: ReduceBlueprint,
    #[comptime] config: ReduceOperationConfig,
    #[define(In)] _input_dtype: StorageType,
    #[define(Out)] _output_dtype: StorageType,
    #[define(Acc)] _acc_dtype: StorageType,
) {
    let (input, mut output) = init_tensors::<RA, In, Out>(input, output);
    reduce_kernel_virtual::<In, Out, Acc>(&input, &mut output, axis_reduce, blueprint, config);
}
