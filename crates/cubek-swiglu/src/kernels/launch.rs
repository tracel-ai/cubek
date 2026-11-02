//! The SwiGLU routine trait: what a routine states. The launches into a buffer, a packed or a
//! tiled matrix are a candidate's
//! ([`SwigluRoutineCandidate`](crate::kernels::SwigluRoutineCandidate)), which holds what it
//! prepared.

use crate::kernels::{SwigluBinding, SwigluBlueprint, SwigluRoutineProblem, SwigluTensors};
use cubecl::tile::launch::DestinationLaunch;
use cubek_std::routine::{Grid, Routine};

/// A routine of this kind around microkernel `M`: it prepares a [`SwigluBlueprint`] and launches
/// it on tensors.
///
/// A routine states what is its own — how it places its microkernel's runs, and the kernel it
/// dispatches through any destination — and the launches around those are written once, on its
/// candidate ([`SwigluRoutineCandidate`](crate::kernels::SwigluRoutineCandidate)).
pub trait SwigluRoutine<M>:
    Routine<Problem = SwigluRoutineProblem, Microkernel = M, Blueprint = SwigluBlueprint<M>>
{
    /// Launch the kernel `blueprint` compiles, on `grid`, over `binding`, its output reached
    /// through `O`.
    fn dispatch<O: DestinationLaunch, Out>(
        &self,
        tensors: &SwigluTensors<Out>,
        binding: SwigluBinding<O::Operand>,
        blueprint: SwigluBlueprint<M>,
        grid: Grid,
    );
}
