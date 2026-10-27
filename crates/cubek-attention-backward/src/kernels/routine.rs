//! What a routine of this kind states: the two walks it launches, and the launch around them.

use crate::kernels::{
    AttentionBackwardGradients, AttentionBackwardMicrokernel, AttentionBackwardProblem,
    AttentionBackwardRoutineBlueprint, AttentionBackwardTensors,
};
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;

/// A routine of this kind: its two walks prepared for a problem, then launched in order.
///
/// A routine of the tile DSL states one launch ([`Routine`](cubek_std::routine::Routine)); one of
/// this kind is two, each placed by its own microkernel and partitioning, so it states both. What
/// autotune races is the routine: the two launches together.
pub trait AttentionBackwardRoutine {
    /// The knobs a measurement settles: each walk's microkernel's, and the routine's own.
    type Settings: core::fmt::Debug + Clone + core::fmt::Display + PartialEq;
    type QueryGradient: AttentionBackwardMicrokernel;
    type KeyGradient: AttentionBackwardMicrokernel;
    /// How a walk lays out the shared memory it holds; `()` for one that holds none.
    type SharedMemory;

    /// The settings worth measuring on this device, the one a caller gets by default first.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings>;

    /// The setting a caller gets by default: the first [`settings`](Self::settings) lists.
    fn lead(&self, hw: &Hardware) -> Self::Settings {
        self.settings(hw).swap_remove(0)
    }

    /// Both walks for `problem` on this device.
    ///
    /// # Errors
    ///
    /// Why it cannot run here.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &Self::Settings,
    ) -> Result<AttentionBackwardRoutineBlueprint<Self>, Refusal>;

    /// Launch both walks `blueprint` compiles, the queries' first, over `tensors`.
    fn dispatch(
        &self,
        tensors: &AttentionBackwardTensors,
        blueprint: AttentionBackwardRoutineBlueprint<Self>,
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
        let blueprint = self.prepare(hw, &tensors.problem(), settings)?;
        self.dispatch(&tensors, blueprint);
        Ok(tensors.into_gradients())
    }
}
