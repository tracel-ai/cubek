//! The reduce routine trait: what a routine states, and the launch of one on tensors.

use crate::kernels::{ReduceBinding, ReduceRoutineBlueprint, ReduceRoutineProblem, ReduceTensors};
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;
use cubek_std::routine::{Grid, Routine};

/// A routine of this kind around microkernel `M`: it prepares a [`ReduceRoutineBlueprint`] and
/// launches it on tensors.
pub trait ReduceRoutine<M>:
    Routine<Problem = ReduceRoutineProblem, Microkernel = M, Blueprint = ReduceRoutineBlueprint<M>>
{
    /// Launch the kernel `blueprint` compiles, on `grid`, over `binding`.
    fn dispatch(
        &self,
        tensors: &ReduceTensors,
        binding: ReduceBinding,
        blueprint: ReduceRoutineBlueprint<M>,
        grid: Grid,
    );
}

/// `tensors` reduced by `routine` at `settings` on this device.
///
/// # Errors
///
/// Why the routine cannot run the problem here.
pub fn launch_reduce_routine<R: ReduceRoutine<M>, M>(
    routine: &R,
    tensors: &ReduceTensors,
    settings: &R::Settings,
) -> Result<(), Refusal> {
    let hw = Hardware::from_client(tensors.client());
    let (blueprint, grid) = routine.prepare(&hw, tensors.problem(), settings)?;
    let binding = tensors.bind(&blueprint, &grid);
    routine.dispatch(tensors, binding, blueprint, grid);
    Ok(())
}
