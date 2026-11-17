//! The reduce routine trait: what a routine states, and the launch of one on tensors.

use crate::kernels::{ReduceBinding, ReduceRoutineBlueprint, ReduceRoutineProblem, ReduceTensors};
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::Geometry;
use cubecl::tile::launch::{Bound, Refusal, StoredTileOperand, TileOperand, Unbound};
use cubek_std::Hardware;
use cubek_std::routine::{Grid, Routine};

/// Why an unbound reduction declines where its microkernel reads or writes narrower lines than the
/// caller's operands are stated at ([`ReduceTensors::unbound`]): the caller computes its lines
/// whole.
pub const UNBOUND_LINES_NARROWER: Refusal = Refusal::Declined {
    why: "the microkernel reads or writes narrower lines than the caller's operands are stated at",
};

/// A routine of this kind around microkernel `M`: it prepares a [`ReduceRoutineBlueprint`] and
/// launches it on tensors.
pub trait ReduceRoutine<M>:
    Routine<Problem = ReduceRoutineProblem, Blueprint = ReduceRoutineBlueprint<M>>
{
    /// Launch the kernel `blueprint` compiles for `problem`, on `grid`, its input served as `I`
    /// serves it and its output as `O` does, from the arguments `binding` holds.
    fn dispatch<I: TileOperand, O: TileOperand>(
        &self,
        problem: &ReduceRoutineProblem,
        client: &Client,
        binding: ReduceBinding<
            <I::Arg as LaunchArg>::RuntimeArg,
            <O::Arg as LaunchArg>::RuntimeArg,
        >,
        blueprint: ReduceRoutineBlueprint<M>,
        grid: Grid,
    );

    /// `tensors` bound as `blueprint` states them and reduced on `grid`.
    fn launch(&self, tensors: &ReduceTensors, blueprint: ReduceRoutineBlueprint<M>, grid: Grid) {
        let binding = tensors
            .bind(&blueprint, &grid)
            .map(Bound::arg, Bound::arg);
        self.dispatch::<StoredTileOperand, StoredTileOperand>(
            tensors.problem(),
            tensors.client(),
            binding,
            blueprint,
            grid,
        );
    }

    /// `tensors`, whose input and output the caller binds, reduced on `grid` as `blueprint` states
    /// it: `input` and `out` build each operand's argument from the spec and geometry the launch
    /// derived for it, served as `I` and `O` serve them.
    ///
    /// # Errors
    ///
    /// [`UNBOUND_LINES_NARROWER`] where `blueprint` reads or writes narrower lines than the
    /// caller's operands are stated at.
    fn launch_unbound<I: TileOperand, O: TileOperand>(
        &self,
        tensors: &ReduceTensors<Geometry>,
        blueprint: ReduceRoutineBlueprint<M>,
        grid: Grid,
        input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
        out: impl FnOnce(Unbound) -> <O::Arg as LaunchArg>::RuntimeArg,
    ) -> Result<(), Refusal> {
        let [input_lines, out_lines] = tensors.problem().elem_alignments;
        if !input_lines.admits(blueprint.width) || !out_lines.admits(blueprint.out_width) {
            return Err(UNBOUND_LINES_NARROWER);
        }
        let binding = tensors.bind(&blueprint, &grid).map(input, out);
        self.dispatch::<I, O>(
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
