//! The conv's routine trait: what a routine states, and its launch written once.

use crate::kernels::conv::{CausalConvBlueprint, CausalConvReads};
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;
use cubek_std::routine::{Grid, PreparedLaunches, Routine};

/// The tensors of one conv launch: its inputs viewed as the kernel reads them, its outputs
/// allocated, and the problem they state.
pub trait ConvTensors: Sized {
    type Inputs;
    type Outputs;
    type Problem;
    /// The operands bound under one launcher.
    type Binding;

    /// The tensors of `inputs`, the outputs allocated.
    ///
    /// # Errors
    ///
    /// Why the inputs cannot be launched on: no row, position or channel, or an element the device
    /// has no buffer of.
    fn new(inputs: Self::Inputs) -> Result<Self, Refusal>;

    /// The problem the tensors state.
    fn problem(&self) -> &Self::Problem;

    /// The operands every conv reads.
    fn reads(&self) -> &CausalConvReads;

    /// Every operand bound as `blueprint` states it, on `grid`.
    fn bind<M>(&self, blueprint: &CausalConvBlueprint<M>, grid: &Grid) -> Self::Binding;

    /// The outputs, written.
    fn into_outputs(self) -> Self::Outputs;
}

/// A routine of this kind around microkernel `M`: it prepares a [`CausalConvBlueprint`] and
/// dispatches its kernel on a conv's tensors. The launch that fills a prompt's rings follows the
/// dispatch, once every position has read the ring, and is the same for every routine.
pub trait CausalConvRoutine<M>:
    Routine<Microkernel = M, Blueprint = CausalConvBlueprint<M>>
{
    type Tensors: ConvTensors<Problem = Self::Problem>;

    /// Launch the kernel `blueprint` compiles, on `grid`, over `binding`.
    fn dispatch(
        &self,
        tensors: &Self::Tensors,
        binding: <Self::Tensors as ConvTensors>::Binding,
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
        inputs: <Self::Tensors as ConvTensors>::Inputs,
    ) -> Result<<Self::Tensors as ConvTensors>::Outputs, Refusal> {
        let tensors = Self::Tensors::new(inputs)?;
        let (blueprint, grid) = self.prepare_first_served(hw, tensors.problem())?;
        Ok(self.launch(hw, tensors, blueprint, grid))
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
        inputs: <Self::Tensors as ConvTensors>::Inputs,
    ) -> Result<<Self::Tensors as ConvTensors>::Outputs, Refusal>
    where
        Self: Sized,
        Self::Settings: Default,
        Self::Problem: PartialEq,
    {
        let tensors = Self::Tensors::new(inputs)?;
        let (blueprint, grid) =
            prepared.prepare(self, hw, tensors.problem(), &Self::Settings::default())?;
        Ok(self.launch(hw, tensors, blueprint, grid))
    }

    /// The conv on `grid`, then the write back of a prompt's rings.
    fn launch(
        &self,
        hw: &Hardware,
        tensors: Self::Tensors,
        blueprint: CausalConvBlueprint<M>,
        grid: Grid,
    ) -> <Self::Tensors as ConvTensors>::Outputs {
        let binding = tensors.bind(&blueprint, &grid);
        let write_back = blueprint.write_back.clone();
        self.dispatch(&tensors, binding, blueprint, grid);
        if let Some(write_back) = write_back {
            write_back.launch(hw, tensors.reads());
        }
        tensors.into_outputs()
    }
}
