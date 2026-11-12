//! What a strategy of this kind states: the routines it launches, one kernel each, in order.

use crate::kernels::{
    AttentionBackwardGradients, AttentionBackwardProblem, AttentionBackwardTensors,
};
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;
use cubek_std::routine::Grid;

/// A backward pass: routines launched in order, each one kernel, which together write the three
/// gradients.
///
/// What autotune races is a strategy at one of its settings. A setting holds each routine's, and
/// which routines run can depend on it: the strategy's blueprint holds one routine blueprint per
/// kernel it launches.
pub trait AttentionBackwardStrategy {
    /// The knobs a measurement settles: each routine's, and which routines run.
    type Settings: core::fmt::Debug + Clone + core::fmt::Display + PartialEq;
    /// Every routine's blueprint, in the order they launch.
    type Blueprint;

    /// The settings worth measuring on this device, the one a caller gets by default first.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings>;

    /// The setting a caller gets by default: the first [`settings`](Self::settings) lists.
    fn lead(&self, hw: &Hardware) -> Self::Settings {
        self.settings(hw).swap_remove(0)
    }

    /// Every routine it launches, prepared for `problem` on this device.
    ///
    /// # Errors
    ///
    /// Why one of them cannot run here.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionBackwardProblem,
        settings: &Self::Settings,
    ) -> Result<Self::Blueprint, Refusal>;

    /// Launch every routine `blueprint` holds, in order, over `tensors`.
    fn dispatch(&self, tensors: &AttentionBackwardTensors, blueprint: Self::Blueprint);

    /// The grid of its largest launch, which autotune ranks the setting by: the walk owned by the
    /// keys.
    fn widest_grid(blueprint: &Self::Blueprint) -> &Grid;

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
