use super::{ConvBinding, ConvBlueprint, ConvProblem, ConvTensors};
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_std::routine::Routine;

/// A routine around microkernel `M`: it prepares a [`ConvBlueprint`] and launches it on tensors.
///
/// A routine states what is its own, how it places its microkernel and the kernel it dispatches,
/// and the launch around those is written once, here.
pub trait ConvRoutine<M: Microkernel>:
    Routine<Problem = ConvProblem, Microkernel = M, Blueprint = ConvBlueprint<M>>
{
    /// Launch the kernel `blueprint` compiles over `binding`.
    fn dispatch(&self, problem: &ConvProblem, binding: ConvBinding, blueprint: ConvBlueprint<M>);

    /// The convolution `tensors` state, prepared for `hw` at `settings`, or at the first setting
    /// the routine lists that prepares where none is named, and launched on their client's device.
    ///
    /// # Errors
    ///
    /// Why `hw` cannot run the convolution at these settings, or at the last one listed where none
    /// serves.
    fn launch_on(
        &self,
        hw: &Hardware,
        tensors: ConvTensors,
        settings: Option<&Self::Settings>,
    ) -> Result<(), Refusal> {
        let problem = tensors.problem();
        let (blueprint, grid) = match settings {
            Some(settings) => self.prepare(hw, &problem, settings)?,
            None => self.prepare_first_served(hw, &problem)?,
        };
        let binding = tensors.bind(&problem, &blueprint, &grid)?;
        self.dispatch(&problem, binding, blueprint);
        Ok(())
    }
}
