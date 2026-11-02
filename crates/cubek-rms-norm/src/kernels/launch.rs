//! The RMS-norm routine trait: what a routine states, and the launch of a norm prepared for a sink
//! the caller builds. The launches into a buffer, a packed or a tiled matrix are a candidate's
//! ([`RmsNormRoutineCandidate`](crate::kernels::RmsNormRoutineCandidate)), which holds what it
//! prepared.

use crate::kernels::{RmsNormBinding, RmsNormBlueprint, RmsNormRoutineProblem, RmsNormTensors};
use cubecl::tile::Geometry;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::SinkLaunch;
use cubecl::tile::launch::{DestinationLaunch, Unbound};
use cubek_std::routine::{Grid, Routine};

/// A routine of this kind around microkernel `M`: it prepares an [`RmsNormBlueprint`] and
/// launches it on tensors.
///
/// A routine states what is its own — how it places its microkernel's rows, and the kernel it
/// dispatches through any destination — and the launches around those are written once, on its
/// candidate ([`RmsNormRoutineCandidate`](crate::kernels::RmsNormRoutineCandidate)).
pub trait RmsNormRoutine<M>:
    Routine<Problem = RmsNormRoutineProblem, Microkernel = M, Blueprint = RmsNormBlueprint<M>>
{
    /// Launch the kernel `blueprint` compiles, on `grid`, over `binding`, its output reached
    /// through `O`, each row's mean square shifted by `epsilon`.
    fn dispatch<O: DestinationLaunch, Out>(
        &self,
        tensors: &RmsNormTensors<Out>,
        binding: RmsNormBinding<O::Operand>,
        blueprint: RmsNormBlueprint<M>,
        grid: Grid,
        epsilon: f32,
    );
}

/// A norm prepared for a sink that does not exist yet: the routine, its operands, and the launch
/// it settled, whose width the sink is then built to — what a fused tail stores through.
#[derive(Debug)]
pub struct RmsNormRoutineSinkLaunch<R, M> {
    routine: R,
    tensors: RmsNormTensors<Geometry>,
    blueprint: RmsNormBlueprint<M>,
    grid: Grid,
    epsilon: f32,
}

/// What a sink launch declines: rows of no rows, which no grid spawns and no sink takes.
pub const RMS_NORM_SINK_TAKES_NO_ROWS: Refusal = Refusal::Declined {
    why: "a sink takes the norm of at least one row",
};

impl<R: RmsNormRoutine<M>, M> RmsNormRoutineSinkLaunch<R, M> {
    /// The norm of `tensors` handed to a sink, by `routine` at the launch it `prepared` for them,
    /// each row's mean square shifted by `epsilon`.
    pub fn new(
        routine: R,
        tensors: RmsNormTensors<Geometry>,
        prepared: (RmsNormBlueprint<M>, Grid),
        epsilon: f32,
    ) -> Self {
        let (blueprint, grid) = prepared;
        Self {
            routine,
            tensors,
            blueprint,
            grid,
            epsilon,
        }
    }
}

impl<R: RmsNormRoutine<M>, M> SinkLaunch for RmsNormRoutineSinkLaunch<R, M> {
    fn width(&self) -> usize {
        self.blueprint.width
    }

    fn elements(&self) -> usize {
        self.tensors.out().shape().iter().product()
    }

    fn launch<O: DestinationLaunch>(self, sink: impl FnOnce(Unbound) -> O::Operand) {
        let binding = self.tensors.bind(&self.blueprint, &self.grid).map_out(sink);
        self.routine.dispatch::<O, _>(
            &self.tensors,
            binding,
            self.blueprint,
            self.grid,
            self.epsilon,
        );
    }
}
