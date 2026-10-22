use crate::kernels::{
    AttentionBinding, AttentionBlueprint, AttentionRoutineProblem, AttentionTensors,
};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention::AttentionMicrokernel;
use cubek_std::Hardware;
use cubek_std::routine::{Grid, ROUTINE_LISTS_NO_SETTING, Routine};

/// A routine around microkernel `M`, laying out shared memory as `SM`: it prepares an
/// [`AttentionBlueprint`] and launches it on tensors.
///
/// A routine states what is its own, how it places its microkernel's work items and the kernel it
/// dispatches, and the launch around those is written once, here.
pub trait AttentionRoutine<M: AttentionMicrokernel, SM = ()>:
    Routine<Problem = AttentionRoutineProblem, Microkernel = M, Blueprint = AttentionBlueprint<M, SM>>
{
    /// Launch the kernel `blueprint` compiles, on `grid`, over `binding`.
    fn dispatch(
        &self,
        tensors: &AttentionTensors,
        binding: AttentionBinding,
        blueprint: AttentionBlueprint<M, SM>,
        grid: Grid,
    );

    /// The attention `tensors` state, launched on their client's device, prepared for `hw`: the
    /// device as it reports itself, or narrowed to offer less, so a candidate further down a
    /// microkernel's list is the one that runs.
    ///
    /// # Errors
    ///
    /// Why `hw` cannot run the attention at these settings.
    fn launch_on(
        &self,
        hw: &Hardware,
        tensors: AttentionTensors,
        settings: &Self::Settings,
    ) -> Result<TensorHandle, Refusal> {
        let (blueprint, grid) = self.prepare(hw, &tensors.problem(), settings)?;
        let binding = tensors.bind(&blueprint, &grid);
        self.dispatch(&tensors, binding, blueprint, grid);
        Ok(tensors.into_out())
    }

    /// [`launch_on`](Self::launch_on) at the routine's lead, the first setting it lists on `hw`:
    /// what a caller naming the routine and no setting runs.
    ///
    /// # Errors
    ///
    /// Why `hw` cannot run the attention at the lead.
    fn launch_lead_on(
        &self,
        hw: &Hardware,
        tensors: AttentionTensors,
    ) -> Result<TensorHandle, Refusal> {
        let lead = self
            .settings(hw)
            .into_iter()
            .next()
            .ok_or(ROUTINE_LISTS_NO_SETTING)?;
        self.launch_on(hw, tensors, &lead)
    }
}
