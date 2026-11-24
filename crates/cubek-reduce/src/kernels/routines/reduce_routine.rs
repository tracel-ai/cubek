//! The trait every reduce routine implements: how it launches what it prepared.

use crate::kernels::{
    ReduceBinding, ReduceBlueprint, ReduceRoutineProblem, ReduceTensors, reduce_lane_counts,
};
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{
    Bound, Decline, StoredTileOperand, TileOperand, Unbound, Vectorization,
};
use cubecl::tile::{Geometry, TileTensorLaunch};
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
pub(crate) trait ReduceRoutine<M>:
    Routine<Problem = ReduceRoutineProblem, Partitioning = (), Blueprint = ReduceBlueprint<M>>
{
    /// The routine's prefix in kebab case, which names its tunables (`"cube-staged-reduce"`).
    const NAME: &'static str;

    /// Launch the kernel `blueprint` compiles for `problem`, on `grid`, accumulating into `F`'s
    /// carrier, its input served as `I` serves it, from the arguments `binding` holds.
    fn dispatch<F: ReduceCarrierFamily, I: TileOperand>(
        problem: &ReduceRoutineProblem,
        client: &Client,
        binding: ReduceBinding<<I::Arg as LaunchArg>::RuntimeArg, TileTensorLaunch>,
        blueprint: ReduceBlueprint<M>,
        grid: Grid,
    );

    /// The lanes `problem`'s input and output bind at on `hw` by what `microkernel`, whose lines are
    /// at most `widest_line` wide, asks of them ([`reduce_lane_counts`]).
    ///
    /// # Errors
    ///
    /// As [`reduce_lane_counts`].
    fn lane_counts(
        hw: &Hardware,
        problem: &ReduceRoutineProblem,
        microkernel: &Self::Microkernel,
        widest_line: Option<usize>,
    ) -> Result<[usize; 2], Decline>
    where
        Self: Routine<Microkernel: Microkernel<Operand = ReduceOperand>>,
    {
        let mut vectorization = Vectorization::new();
        Self::vectorize(microkernel, &(), &mut vectorization);
        reduce_lane_counts(hw, problem, vectorization, widest_line)
    }

    /// [`dispatch`](Self::dispatch) into the carrier `problem`'s operation accumulates into.
    fn dispatch_carrier<I: TileOperand>(
        problem: &ReduceRoutineProblem,
        client: &Client,
        binding: ReduceBinding<<I::Arg as LaunchArg>::RuntimeArg, TileTensorLaunch>,
        blueprint: ReduceBlueprint<M>,
        grid: Grid,
    ) {
        match problem.operation.carrier() {
            ReduceCarrier::Values => Self::dispatch::<ValuesReduceCarrierFamily, I>(
                problem, client, binding, blueprint, grid,
            ),
            ReduceCarrier::Indexed => Self::dispatch::<IndexedReduceCarrierFamily, I>(
                problem, client, binding, blueprint, grid,
            ),
            ReduceCarrier::TopK { .. } => Self::dispatch::<TopKReduceCarrierFamily, I>(
                problem, client, binding, blueprint, grid,
            ),
            ReduceCarrier::TopValues { .. } => Self::dispatch::<TopValuesReduceCarrierFamily, I>(
                problem, client, binding, blueprint, grid,
            ),
        }
    }

    /// `tensors` bound as `blueprint` states them and reduced on `grid`.
    fn launch(tensors: &ReduceTensors, blueprint: ReduceBlueprint<M>, grid: Grid) {
        let binding = tensors.bind(&blueprint, &grid).map(Bound::arg, Bound::arg);
        Self::dispatch_carrier::<StoredTileOperand>(
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
    fn launch_unbound<I: TileOperand>(
        tensors: &ReduceTensors<Geometry>,
        blueprint: ReduceBlueprint<M>,
        grid: Grid,
        input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
    ) {
        let binding = tensors.bind(&blueprint, &grid).map(input, Bound::arg);
        Self::dispatch_carrier::<I>(
            tensors.problem(),
            tensors.client(),
            binding,
            blueprint,
            grid,
        );
    }
}
