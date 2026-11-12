//! The conv's routine trait: what a routine states, and its launch written once.

use crate::kernels::conv::{
    CausalConvBinding, CausalConvBlueprint, CausalConvInputs, CausalConvOutputs, CausalConvProblem,
    CausalConvTensors,
};
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;
use cubek_std::routine::{Grid, PreparedLaunches, Routine};

/// A routine of this kind around microkernel `M`: it prepares a [`CausalConvBlueprint`] and
/// dispatches its kernel on a conv's tensors. The launch that fills a prompt's rings follows the
/// dispatch, once every position has read the ring, and is the same for every routine.
pub trait CausalConvRoutine<M>:
    Routine<Microkernel = M, Blueprint = CausalConvBlueprint<M>, Problem = CausalConvProblem>
{
    /// Launch the kernel `blueprint` compiles, on `grid`, over `binding`.
    fn dispatch(
        &self,
        tensors: &CausalConvTensors,
        binding: CausalConvBinding,
        blueprint: CausalConvBlueprint<M>,
        grid: Grid,
    );

    /// The conv of `inputs`, launched on their client's device, prepared for `hw`.
    ///
    /// # Errors
    ///
    /// Why `hw` cannot run the conv on these inputs.
    fn launch_on(
        &self,
        hw: &Hardware,
        inputs: CausalConvInputs,
    ) -> Result<CausalConvOutputs, Refusal> {
        let tensors = CausalConvTensors::new(inputs)?;
        let (blueprint, grid) = self.prepare_first_served(hw, tensors.problem())?;
        Ok(launch(self, tensors, blueprint, grid))
    }

    /// [`launch_on`](Self::launch_on), preparing a problem once however often it is launched.
    ///
    /// # Errors
    ///
    /// Why `hw` cannot run the conv on these inputs.
    fn launch_prepared_on(
        &self,
        prepared: &PreparedLaunches<Self>,
        hw: &Hardware,
        inputs: CausalConvInputs,
    ) -> Result<CausalConvOutputs, Refusal>
    where
        Self: Sized,
        Self::Settings: Default,
        Self::Problem: PartialEq,
    {
        let tensors = CausalConvTensors::new(inputs)?;
        let (blueprint, grid) =
            prepared.prepare(self, hw, tensors.problem(), &Self::Settings::default())?;
        Ok(launch(self, tensors, blueprint, grid))
    }
}

/// The conv on `grid`, then the write back of a prompt's rings: no routine can skip or reorder it.
fn launch<M, R: CausalConvRoutine<M> + ?Sized>(
    routine: &R,
    tensors: CausalConvTensors,
    blueprint: CausalConvBlueprint<M>,
    grid: Grid,
) -> CausalConvOutputs {
    let binding = tensors.bind(&blueprint, &grid);
    let write_back = blueprint.write_back.clone();
    routine.dispatch(&tensors, binding, blueprint, grid);
    if let Some(write_back) = write_back {
        write_back.launch(&tensors);
    }
    tensors.into_outputs()
}
