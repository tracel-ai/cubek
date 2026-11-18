//! The trait every reduce routine implements: how it launches what it prepared.

use crate::kernels::{
    ReduceBinding, ReduceRoutineBlueprint, ReduceRoutineProblem, ReduceTensors,
    UNBOUND_LINES_NARROWER,
};
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::Geometry;
use cubecl::tile::TileTensorLaunch;
use cubecl::tile::launch::{Bound, Refusal, StoredTileOperand, TileOperand, Unbound};
use cubek_std::routine::{Grid, Routine};

/// A reduce routine around microkernel `M`, as a launcher: it launches the
/// [`ReduceRoutineBlueprint`] it prepared on tensors.
pub(crate) trait ReduceLauncher<M>:
    Routine<Problem = ReduceRoutineProblem, Blueprint = ReduceRoutineBlueprint<M>>
{
    /// The routine's prefix in snake case, which names its tunables (`"cube_staged_reduce"`).
    const NAME: &'static str;

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
