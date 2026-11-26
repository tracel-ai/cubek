//! The trait every reduce routine implements: how it launches what it prepared.

use crate::kernels::{
    ReduceBinding, ReduceBlueprint, ReduceRoutineProblem, ReduceTensors, bound_as,
    reduce_lane_count,
};
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{
    Bound, Decline, StoredTileOperand, TileOperand, Unbound, Vectorization,
};
use cubecl::tile::{Geometry, Launcher};
use cubek_microkernel::reduce::{
    IndexedReduceCarrierFamily, ReduceCarrier, ReduceCarrierFamily, ReduceOperand,
    TopKReduceCarrierFamily, TopValuesReduceCarrierFamily, ValuesReduceCarrierFamily,
};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_std::routine::{Grid, Routine};

/// A reduce routine around microkernel `M`: it prepares a [`ReduceBlueprint`] and launches it on
/// tensors.
///
/// A reduce routine's own code reads nothing: every access is its microkernel's, so its
/// `vectorize` hands the whole pass to the microkernel, whatever its placement.
///
/// # Assumptions
///
/// - The blueprint and grid a launch takes are what the routine's `prepare` returned for the
///   tensors' problem; a launch that finds otherwise returns a
///   [`Decline::Broken`](cubecl::tile::launch::Decline::Broken).
pub(crate) trait ReduceRoutine<M>:
    Routine<Problem = ReduceRoutineProblem, Partitioning = (), Blueprint = ReduceBlueprint<M>>
{
    /// The routine's prefix in kebab case, which names its tunables (`"cube-staged-reduce"`).
    const NAME: &'static str;

    /// The argument its kernel writes the output through.
    type Out;

    /// `out`, the output as the launch binds it, as that argument.
    fn out_arg(out: Bound) -> Self::Out;

    /// Launch the kernel `blueprint` compiles for `problem`, on `grid`, accumulating into `F`'s
    /// carrier, its input served as `I` serves it, from the arguments `binding` holds, bound
    /// through `launcher`.
    ///
    /// # Errors
    ///
    /// [`Decline::Broken`] where the launch breaks what `prepare` claimed of it, nothing launched.
    fn dispatch<F: ReduceCarrierFamily, I: TileOperand>(
        problem: &ReduceRoutineProblem,
        client: &Client,
        launcher: &Launcher,
        binding: ReduceBinding<<I::Arg as LaunchArg>::RuntimeArg, Self::Out>,
        blueprint: ReduceBlueprint<M>,
        grid: Grid,
    ) -> Result<(), Decline>;

    /// The lanes `problem`'s input binds at on `hw` by what `microkernel`, whose lines are at most
    /// `widest_line` wide, asks of them ([`reduce_lane_count`]).
    ///
    /// # Errors
    ///
    /// As [`reduce_lane_count`].
    fn lane_count(
        hw: &Hardware,
        problem: &ReduceRoutineProblem,
        microkernel: &Self::Microkernel,
        widest_line: Option<usize>,
    ) -> Result<usize, Decline>
    where
        Self: Routine<Microkernel: Microkernel<Operand = ReduceOperand>>,
    {
        let mut vectorization = Vectorization::new();
        Self::vectorize(microkernel, &(), &mut vectorization);
        reduce_lane_count(hw, problem, vectorization, widest_line)
    }

    /// [`dispatch`](Self::dispatch) into the carrier `problem`'s operation accumulates into.
    ///
    /// # Errors
    ///
    /// As [`dispatch`](Self::dispatch).
    fn dispatch_carrier<I: TileOperand>(
        problem: &ReduceRoutineProblem,
        client: &Client,
        launcher: &Launcher,
        binding: ReduceBinding<<I::Arg as LaunchArg>::RuntimeArg, Self::Out>,
        blueprint: ReduceBlueprint<M>,
        grid: Grid,
    ) -> Result<(), Decline> {
        match problem.operation.carrier() {
            ReduceCarrier::Values => Self::dispatch::<ValuesReduceCarrierFamily, I>(
                problem, client, launcher, binding, blueprint, grid,
            ),
            ReduceCarrier::Indexed => Self::dispatch::<IndexedReduceCarrierFamily, I>(
                problem, client, launcher, binding, blueprint, grid,
            ),
            ReduceCarrier::TopK { .. } => Self::dispatch::<TopKReduceCarrierFamily, I>(
                problem, client, launcher, binding, blueprint, grid,
            ),
            ReduceCarrier::TopValues { .. } => Self::dispatch::<TopValuesReduceCarrierFamily, I>(
                problem, client, launcher, binding, blueprint, grid,
            ),
        }
    }

    /// `tensors` bound as `blueprint` states them and reduced on `grid`.
    ///
    /// # Errors
    ///
    /// As [`dispatch`](Self::dispatch).
    fn launch(
        tensors: &ReduceTensors,
        blueprint: ReduceBlueprint<M>,
        grid: Grid,
    ) -> Result<(), Decline> {
        let (launcher, binding) = tensors.bind(&blueprint, &grid)?;
        Self::dispatch_carrier::<StoredTileOperand>(
            tensors.problem(),
            tensors.client(),
            &launcher,
            bound_as(binding, Bound::arg, Self::out_arg),
            blueprint,
            grid,
        )
    }

    /// `tensors`, whose input the caller binds, reduced on `grid` as `blueprint` states it: `input`
    /// builds the input's argument from the spec and geometry the launch derived for it, served as
    /// `I` serves it.
    ///
    /// # Errors
    ///
    /// As [`dispatch`](Self::dispatch).
    fn launch_unbound<I: TileOperand>(
        tensors: &ReduceTensors<Geometry>,
        blueprint: ReduceBlueprint<M>,
        grid: Grid,
        input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
    ) -> Result<(), Decline> {
        let (launcher, binding) = tensors.bind(&blueprint, &grid)?;
        Self::dispatch_carrier::<I>(
            tensors.problem(),
            tensors.client(),
            &launcher,
            bound_as(binding, input, Self::out_arg),
            blueprint,
            grid,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testing::{BufferLaunch, UnboundLaunch, launch_unbound};
    use crate::kernels::{UnitDirectReduceRoutine, launch_first_served};
    use cubek_microkernel::reduce::unit::UnitDirectReduceMicrokernel;
    use cubek_std::devices;

    /// A microkernel whose lines are at most one wide binds the input at one lane.
    #[test]
    fn the_widest_line_bounds_the_lanes() {
        let hw = devices::m2_pro();
        let problem = ReduceRoutineProblem::dense(
            &[4, 4096],
            1,
            cubek_microkernel::reduce::ReduceOperation::Sum,
        );
        let settings = UnitDirectReduceMicrokernel::settings(&hw)[0];
        let microkernel = UnitDirectReduceMicrokernel::best(&hw, &problem.work_item(), &settings)
            .expect("the unit microkernel runs everywhere");
        let lane_count = UnitDirectReduceRoutine::lane_count(&hw, &problem, &microkernel, Some(1));
        assert_eq!(lane_count, Ok(1));
    }

    const LAUNCH: BufferLaunch = launch_first_served::<UnitDirectReduceRoutine, _>;
    const UNBOUND: UnboundLaunch = launch_unbound::<UnitDirectReduceRoutine, _>;

    /// Each carrier a launch accumulates into, on rows of seven values.
    mod launch {
        use super::*;
        reduction_tests!(BufferScopeChecker, LAUNCH, [9, 7], 1, [
            values: ReduceOperation::Sum,
            indexed: ReduceOperation::ArgMax,
            top_k: ReduceOperation::ArgTopK { k: 2 },
            top_values: ReduceOperation::TopK { k: 2 },
        ]);
    }

    /// [`launch`], of an input the caller binds.
    mod launch_unbound_input {
        use super::*;
        reduction_tests!(UnboundScopeChecker, UNBOUND, [9, 7], 1, [
            values: ReduceOperation::Sum,
        ]);
    }
}
