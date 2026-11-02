//! What a routine of this kind states beside the [`Routine`] it is: the launch of the walks it
//! prepared.

use crate::kernels::{
    AttentionBackwardGradients, AttentionBackwardGrid, AttentionBackwardProblem,
    AttentionBackwardTensors,
};
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;
use cubek_std::routine::Routine;

/// A routine of this kind: the walks it prepares for a problem, then launches in order, the
/// queries' first. What autotune races is the routine: its launches together.
pub trait AttentionBackwardRoutine:
    Routine<Problem = AttentionBackwardProblem, Grid = AttentionBackwardGrid>
{
    /// The setting a caller gets by default: the first [`settings`](Routine::settings) lists.
    fn lead(&self, hw: &Hardware) -> Self::Settings {
        self.settings(hw).swap_remove(0)
    }

    /// Launch the walks `blueprint` compiles, on `grid`, the queries' first, over `tensors`.
    fn dispatch(
        &self,
        tensors: &AttentionBackwardTensors,
        blueprint: Self::Blueprint,
        grid: AttentionBackwardGrid,
    );

    /// The gradients `tensors` state, launched on their client's device, prepared for `hw`.
    ///
    /// # Errors
    ///
    /// Why `hw` cannot run the backward pass at these settings.
    fn launch_on(
        &self,
        hw: &Hardware,
        tensors: AttentionBackwardTensors,
        settings: &Self::Settings,
    ) -> Result<AttentionBackwardGradients, Refusal> {
        let (blueprint, grid) = self.prepare(hw, &tensors.problem(), settings)?;
        self.dispatch(&tensors, blueprint, grid);
        Ok(tensors.into_gradients())
    }
}
