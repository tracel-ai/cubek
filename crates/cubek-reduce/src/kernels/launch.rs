//! The reduce routine trait: what a routine states, and the launch of one on tensors.

use crate::kernels::{ReduceBinding, ReduceRoutineBlueprint, ReduceRoutineProblem, ReduceTensors};
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{Bound, Refusal, StoredTileOperand, TileOperand, Unbound};
use cubecl::tile::{Geometry, TileTensorLaunch};
use cubek_std::Hardware;
use cubek_std::routine::{Grid, Routine};

/// Why an unbound reduction declines where its microkernel reads narrower lines than the caller's
/// input is stated at ([`ReduceTensors::unbound`]): the caller computes its lines whole.
pub const UNBOUND_LINES_NARROWER: Refusal = Refusal::Declined {
    why: "the microkernel reads narrower lines than the caller's input is stated at",
};

/// A routine of this kind around microkernel `M`: it prepares a [`ReduceRoutineBlueprint`] and
/// launches it on tensors.
pub trait ReduceRoutine<M>:
    Routine<Problem = ReduceRoutineProblem, Blueprint = ReduceRoutineBlueprint<M>>
{
    /// Launch the kernel `blueprint` compiles for `problem`, on `grid`, its input served as `I`
    /// serves it, from the arguments `binding` holds.
    fn dispatch<I: TileOperand>(
        &self,
        problem: &ReduceRoutineProblem,
        client: &Client,
        binding: ReduceBinding<<I::Arg as LaunchArg>::RuntimeArg, TileTensorLaunch>,
        blueprint: ReduceRoutineBlueprint<M>,
        grid: Grid,
    );

    /// `tensors` bound as `blueprint` states them and reduced on `grid`.
    fn launch(&self, tensors: &ReduceTensors, blueprint: ReduceRoutineBlueprint<M>, grid: Grid) {
        let binding = tensors.bind(&blueprint, &grid).map(Bound::arg, Bound::arg);
        self.dispatch::<StoredTileOperand>(
            tensors.problem(),
            tensors.client(),
            binding,
            blueprint,
            grid,
        );
    }

    /// `tensors`, whose input the caller binds, reduced on `grid` as `blueprint` states it: `input`
    /// builds the input's argument from the spec and geometry the launch derived for it, served as
    /// `I` serves it.
    ///
    /// # Errors
    ///
    /// [`UNBOUND_LINES_NARROWER`] where `blueprint` reads narrower lines than the caller's input is
    /// stated at.
    fn launch_unbound<I: TileOperand>(
        &self,
        tensors: &ReduceTensors<Geometry>,
        blueprint: ReduceRoutineBlueprint<M>,
        grid: Grid,
        input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
    ) -> Result<(), Refusal> {
        let [input_lines, _] = tensors.problem().elem_alignments;
        if !input_lines.admits(blueprint.width) {
            return Err(UNBOUND_LINES_NARROWER);
        }
        let binding = tensors.bind(&blueprint, &grid).map(input, Bound::arg);
        self.dispatch::<I>(
            tensors.problem(),
            tensors.client(),
            binding,
            blueprint,
            grid,
        );
        Ok(())
    }
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
    routine.launch(tensors, blueprint, grid);
    Ok(())
}
