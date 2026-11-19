use super::{ConvBinding, ConvProblem, ConvTensors, TileConvBlueprint};
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_std::routine::{Grid, Routine};

/// A routine around microkernel `M`: it prepares a [`TileConvBlueprint`] and launches it on
/// tensors.
///
/// A routine states how it places its microkernel and which kernel it dispatches; the launch
/// around those is written once, here.
pub trait ConvRoutine<M: Microkernel>:
    Routine<Problem = ConvProblem, Microkernel = M, Blueprint = TileConvBlueprint<M>>
{
    /// Launch the kernel `blueprint` compiles over `binding`.
    fn dispatch(
        &self,
        problem: &ConvProblem,
        binding: ConvBinding,
        blueprint: TileConvBlueprint<M>,
    );

    /// The convolution `tensors` state, prepared for `hw` at `settings` and launched on their
    /// client's device.
    ///
    /// # Errors
    ///
    /// Why `hw` cannot run the convolution at these settings.
    fn launch_on(
        &self,
        hw: &Hardware,
        tensors: ConvTensors,
        settings: &Self::Settings,
    ) -> Result<(), Refusal> {
        let problem = tensors.problem();
        let prepared = self.prepare(hw, &problem, settings)?;
        self.launch_prepared(&problem, tensors, prepared)
    }

    /// The convolution `tensors` state, launched at the first of the routine's own settings that
    /// prepares it on `hw`.
    ///
    /// # Errors
    ///
    /// Why the last setting declines it where none serves.
    fn launch_first_served(&self, hw: &Hardware, tensors: ConvTensors) -> Result<(), Refusal> {
        let problem = tensors.problem();
        let prepared = self.prepare_first_served(hw, &problem)?;
        self.launch_prepared(&problem, tensors, prepared)
    }

    /// The convolution `tensors` state of `problem`, launched as `prepared` places it.
    ///
    /// # Errors
    ///
    /// A width the binding cannot serve one of the operands at.
    fn launch_prepared(
        &self,
        problem: &ConvProblem,
        tensors: ConvTensors,
        (blueprint, grid): (TileConvBlueprint<M>, Grid),
    ) -> Result<(), Refusal> {
        let binding = tensors.bind(problem, &blueprint, &grid)?;
        self.dispatch(problem, binding, blueprint);
        Ok(())
    }
}
