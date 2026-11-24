//! The trait every reduce routine implements: how it launches what it prepared.

use crate::kernels::{ReduceBinding, ReduceBlueprint, ReduceRoutineProblem, ReduceTensors};
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{Bound, StoredTileOperand, TileOperand, Unbound};
use cubecl::tile::{Geometry, TileTensorLaunch};
use cubek_microkernel::reduce::{
    IndexedReduceCarrierFamily, ReduceCarrier, ReduceCarrierFamily, TopKReduceCarrierFamily,
    TopValuesReduceCarrierFamily, ValuesReduceCarrierFamily,
};
use cubek_std::routine::{Grid, Routine};

/// A reduce routine around microkernel `M`, as a launcher: it launches the
/// [`ReduceBlueprint`] it prepared on tensors.
///
/// A reduce routine's own code reads nothing: every access is its microkernel's, so its
/// `vectorize` hands the whole pass to the microkernel.
pub(crate) trait ReduceLauncher<M>:
    Routine<Problem = ReduceRoutineProblem, Blueprint = ReduceBlueprint<M>>
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
