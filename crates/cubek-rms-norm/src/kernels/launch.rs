//! The RMS-norm routine trait: what a routine states, and the launch of a norm prepared for a sink
//! the caller builds. The launches into a buffer, a packed or a tiled matrix are a candidate's
//! ([`RmsNormRoutineCandidate`](crate::kernels::RmsNormRoutineCandidate)), which holds what it
//! prepared.

use crate::kernels::{RmsNormBinding, RmsNormBlueprint, RmsNormRoutineProblem, RmsNormTensors};
use cubecl::tile::Geometry;
use cubecl::tile::launch::Decline;
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
pub struct RmsNormRoutineSinkLaunch<R, M> {
    routine: R,
    tensors: RmsNormTensors<Geometry>,
    blueprint: RmsNormBlueprint<M>,
    grid: Grid,
    epsilon: f32,
    binding: RmsNormBinding<Unbound>,
}

/// What a sink launch declines: rows of no rows, which no grid spawns and no sink takes.
pub const RMS_NORM_SINK_TAKES_NO_ROWS: Decline = Decline::Unserved {
    why: "a sink takes the norm of at least one row",
};

impl<R: RmsNormRoutine<M>, M> RmsNormRoutineSinkLaunch<R, M> {
    /// The norm of `tensors` handed to a sink, by `routine` at the launch it `prepared` for them,
    /// each row's mean square shifted by `epsilon`, its operands bound now.
    ///
    /// # Errors
    ///
    /// As [`RmsNormTensors::bind`].
    pub fn new(
        routine: R,
        tensors: RmsNormTensors<Geometry>,
        prepared: (RmsNormBlueprint<M>, Grid),
        epsilon: f32,
    ) -> Result<Self, Decline> {
        let (blueprint, grid) = prepared;
        let binding = tensors.bind(&blueprint, &grid)?;
        Ok(Self {
            routine,
            tensors,
            blueprint,
            grid,
            epsilon,
            binding,
        })
    }
}

impl<R: core::fmt::Debug, M: core::fmt::Debug> core::fmt::Debug for RmsNormRoutineSinkLaunch<R, M> {
    // The binding is left out: the tile DSL's `Launcher` is not `Debug`.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RmsNormRoutineSinkLaunch")
            .field("routine", &self.routine)
            .field("tensors", &self.tensors)
            .field("blueprint", &self.blueprint)
            .field("grid", &self.grid)
            .field("epsilon", &self.epsilon)
            .finish_non_exhaustive()
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
        let binding = self.binding.map_out(sink);
        self.routine.dispatch::<O, _>(
            &self.tensors,
            binding,
            self.blueprint,
            self.grid,
            self.epsilon,
        );
    }
}
