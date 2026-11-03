//! The walk's routine trait: what a routine states, and its launch written once.

use crate::kernels::walk::{
    GatedDeltaWalkBinding, GatedDeltaWalkBlueprint, GatedDeltaWalkInputs, GatedDeltaWalkOutputs,
    GatedDeltaWalkProblem, GatedDeltaWalkTensors,
};
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;
use cubek_std::routine::{Grid, ROUTINE_LISTS_NO_SETTING, Routine};

/// A routine of this kind around microkernel `M`: it prepares a [`GatedDeltaWalkBlueprint`] and
/// launches it on a walk's tensors.
pub trait GatedDeltaWalkRoutine<M>:
    Routine<Problem = GatedDeltaWalkProblem, Microkernel = M, Blueprint = GatedDeltaWalkBlueprint<M>>
{
    /// Launch the kernel `blueprint` compiles, on `grid`, over `binding`.
    fn dispatch(
        &self,
        tensors: &GatedDeltaWalkTensors,
        binding: GatedDeltaWalkBinding,
        blueprint: GatedDeltaWalkBlueprint<M>,
        grid: Grid,
    );

    /// The walk of `inputs`, launched on their client's device, prepared for `hw`: the device as it
    /// reports itself, or narrowed to offer less.
    ///
    /// # Errors
    ///
    /// Why `hw` cannot run the walk at these settings.
    fn launch_on(
        &self,
        hw: &Hardware,
        inputs: GatedDeltaWalkInputs,
        settings: &Self::Settings,
    ) -> Result<GatedDeltaWalkOutputs, Refusal> {
        let tensors = GatedDeltaWalkTensors::new(inputs)?;
        let (blueprint, grid) = self.prepare(hw, tensors.problem(), settings)?;
        let binding = tensors.bind(&blueprint, &grid);
        self.dispatch(&tensors, binding, blueprint, grid);
        Ok(tensors.into_outputs())
    }

    /// [`launch_on`](Self::launch_on) at the routine's lead, the first setting it lists on `hw`.
    ///
    /// # Errors
    ///
    /// Why `hw` cannot run the walk at the lead.
    fn launch_lead_on(
        &self,
        hw: &Hardware,
        inputs: GatedDeltaWalkInputs,
    ) -> Result<GatedDeltaWalkOutputs, Refusal> {
        let lead = self
            .settings(hw)
            .into_iter()
            .next()
            .ok_or(ROUTINE_LISTS_NO_SETTING)?;
        self.launch_on(hw, inputs, &lead)
    }
}
