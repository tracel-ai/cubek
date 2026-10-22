use crate::kernels::MatmulBinding;
use crate::kernels::tensor::{PlainTensor, StoredTensor};
use crate::microkernel::MatmulMicrokernel;
use crate::microkernel::staged_walk::MatmulCubeWrite;
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::Geometry;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::SinkLaunch;
use cubek_std::Hardware;

use crate::kernels::{MatmulBlueprint, MatmulOutput, MatmulRoutineProblem, MatmulTensors};
use cubecl::tile::{
    TileArgLaunch,
    kind::Write,
    launch::{Buffered, DestinationLaunch, Unbound},
};
use cubek_std::routine::{Grid, Routine};

/// What a routine declines before preparing anything: a sink, for a routine whose cubes reach the
/// output in a way no sink takes ([`MatmulCubeWrite::takes_sink`]).
pub const MATMUL_SINK_TAKES_NO_RELAY: Refusal = Refusal::Declined {
    why: "a sink takes no relayed output: a relay's copy into a sink has not run on a device whose cubes may wait on each other",
};

/// What a launch that allocated its output hands back.
const ALLOCATED: &str = "a launch allocating its output hands it back";

/// A routine around microkernel `M`, laying out shared memory as `SM`: it prepares a
/// [`MatmulBlueprint`] and launches it on tensors, each as it is stored ([`StoredTensor`]).
///
/// A routine states what is its own, how it places its microkernel's work items and the kernel it
/// dispatches, and the launch around those is written once, here, whether the microkernel holds
/// every row whole or tiles them, and however the weight is stored: its kernel reads the weight
/// as the tile it is bound as. A routine declines a problem whose weight it does not read.
pub trait MatmulRoutine<M: MatmulMicrokernel, SM = ()>:
    Routine<Problem = MatmulRoutineProblem, Microkernel = M, Blueprint = MatmulBlueprint<M, SM>>
{
    /// How the routine's cubes reach the output.
    const WRITE: MatmulCubeWrite = MatmulCubeWrite::Replace;

    /// Launch the kernel `blueprint` compiles, on `grid`, over `binding`, its kernel reaching the
    /// output through the destination `O`, whatever `tensors`' output is.
    fn dispatch<O: DestinationLaunch, Out: MatmulOutput>(
        &self,
        tensors: &MatmulTensors<Out>,
        binding: MatmulBinding<O::Operand, TileArgLaunch, TileArgLaunch>,
        blueprint: MatmulBlueprint<M, SM>,
        grid: Grid,
    );

    /// `lhs · rhs` on `client`'s device, prepared for `hw`: the device as it reports itself, or
    /// narrowed to offer less (no vendor fragment API, no matrix instruction at all), so a
    /// candidate further down a microkernel's list is the one that runs. A description offering
    /// more than the device launches what it cannot run.
    ///
    /// # Errors
    ///
    /// Why `hw` cannot run the product at these settings.
    fn launch_on(
        &self,
        hw: &Hardware,
        client: &Client,
        lhs: StoredTensor,
        rhs: StoredTensor,
        settings: &Self::Settings,
    ) -> Result<TensorHandle, Refusal> {
        let tensors = MatmulTensors::new::<M>(client, lhs, rhs)?;
        let (blueprint, grid) = self.prepare(hw, &tensors.problem(), settings)?;
        let binding = tensors
            .bind(&blueprint, &grid)
            .map_out(|out| (out, Write::Replace));
        self.dispatch::<Buffered, _>(&tensors, binding, blueprint, grid);
        Ok(tensors.into_out().expect(ALLOCATED))
    }

    /// `lhs · rhs` on `client`'s device written into `out`, which the caller allocated, prepared
    /// for `hw` at `settings`.
    ///
    /// # Errors
    ///
    /// What [`MatmulTensors::writing_into`] declines, and why `hw` cannot run the product at
    /// these settings.
    fn launch_into(
        &self,
        hw: &Hardware,
        client: &Client,
        lhs: PlainTensor,
        rhs: PlainTensor,
        out: PlainTensor,
        settings: &Self::Settings,
    ) -> Result<(), Refusal> {
        let tensors = MatmulTensors::writing_into::<M>(client, lhs, rhs, out)?;
        let (blueprint, grid) = self.prepare(hw, &tensors.problem(), settings)?;
        let binding = tensors
            .bind(&blueprint, &grid)
            .map_out(|out| (out, Write::Replace));
        self.dispatch::<Buffered, _>(&tensors, binding, blueprint, grid);
        Ok(())
    }

    /// [`launch_on`](Self::launch_on) into an output of `out`: what a product whose activation is
    /// packed, and so names no float element, is written in
    /// ([`MatmulTensors::written_in`]).
    ///
    /// # Errors
    ///
    /// As [`launch_on`](Self::launch_on).
    fn launch_written_in_on(
        &self,
        hw: &Hardware,
        client: &Client,
        lhs: StoredTensor,
        rhs: StoredTensor,
        out: ElemType,
        settings: &Self::Settings,
    ) -> Result<TensorHandle, Refusal> {
        let tensors = MatmulTensors::written_in(client, lhs, rhs, out)?;
        let (blueprint, grid) = self.prepare(hw, &tensors.problem(), settings)?;
        let binding = tensors
            .bind(&blueprint, &grid)
            .map_out(|out| (out, Write::Replace));
        self.dispatch::<Buffered, _>(&tensors, binding, blueprint, grid);
        Ok(tensors.into_out().expect(ALLOCATED))
    }

    /// `lhs · rhs` at the first of the routine's own settings that prepares a launch on `hw`
    /// ([`prepare_first_served`](Routine::prepare_first_served)): what a caller naming the routine
    /// and no setting runs.
    ///
    /// # Errors
    ///
    /// The last setting's decline, where none serves.
    fn launch_first_served_on(
        &self,
        hw: &Hardware,
        client: &Client,
        lhs: StoredTensor,
        rhs: StoredTensor,
    ) -> Result<TensorHandle, Refusal> {
        let tensors = MatmulTensors::new::<M>(client, lhs, rhs)?;
        let (blueprint, grid) = self.prepare_first_served(hw, &tensors.problem())?;
        let binding = tensors
            .bind(&blueprint, &grid)
            .map_out(|out| (out, Write::Replace));
        self.dispatch::<Buffered, _>(&tensors, binding, blueprint, grid);
        Ok(tensors.into_out().expect(ALLOCATED))
    }

    /// [`launch_first_served_on`](Self::launch_first_served_on) into an output of `out`: what a
    /// caller naming the routine and no setting runs on a packed activation
    /// ([`MatmulTensors::written_in`]).
    ///
    /// # Errors
    ///
    /// As [`launch_first_served_on`](Self::launch_first_served_on), and as
    /// [`MatmulTensors::written_in`].
    fn launch_first_served_in_on(
        &self,
        hw: &Hardware,
        client: &Client,
        lhs: StoredTensor,
        rhs: StoredTensor,
        out: ElemType,
    ) -> Result<TensorHandle, Refusal> {
        let tensors = MatmulTensors::written_in(client, lhs, rhs, out)?;
        let (blueprint, grid) = self.prepare_first_served(hw, &tensors.problem())?;
        let binding = tensors
            .bind(&blueprint, &grid)
            .map_out(|out| (out, Write::Replace));
        self.dispatch::<Buffered, _>(&tensors, binding, blueprint, grid);
        Ok(tensors.into_out().expect(ALLOCATED))
    }
}

/// A product prepared for a sink that does not exist yet: the routine, its operands, and the
/// launch it settled, whose output width the sink is then built to.
///
/// A fused tail is one such sink: the width its element-wise block stores in is decided around the
/// kernel's own, before the block exists, and the block is handed to the launch after.
#[derive(Debug)]
pub struct MatmulRoutineSinkLaunch<R, M, SM = ()> {
    routine: R,
    tensors: MatmulTensors<Geometry>,
    blueprint: MatmulBlueprint<M, SM>,
    grid: Grid,
}

impl<R, M, SM> MatmulRoutineSinkLaunch<R, M, SM>
where
    R: MatmulRoutine<M, SM>,
    M: MatmulMicrokernel,
{
    /// `lhs · rhs` on `client`'s device handed to a sink, `routine` prepared at `settings` for
    /// `hw`.
    ///
    /// # Errors
    ///
    /// [`MATMUL_SINK_TAKES_NO_RELAY`] for a routine whose cubes reach the output in a way no sink
    /// takes; what [`MatmulTensors::sink`](MatmulTensors#method.sink) declines; why
    /// `hw` cannot run the product at these settings.
    pub fn new(
        routine: R,
        hw: &Hardware,
        client: &Client,
        lhs: StoredTensor,
        rhs: StoredTensor,
        settings: &R::Settings,
    ) -> Result<Self, Refusal> {
        if !R::WRITE.takes_sink() {
            return Err(MATMUL_SINK_TAKES_NO_RELAY);
        }
        let tensors = MatmulTensors::sink::<M>(client, lhs, rhs)?;
        let (blueprint, grid) = routine.prepare(hw, &tensors.problem(), settings)?;
        Ok(Self {
            routine,
            tensors,
            blueprint,
            grid,
        })
    }
}

impl<R, M, SM> SinkLaunch for MatmulRoutineSinkLaunch<R, M, SM>
where
    R: MatmulRoutine<M, SM>,
    M: MatmulMicrokernel,
{
    fn width(&self) -> usize {
        self.blueprint.widths.out
    }

    fn elements(&self) -> usize {
        self.tensors.out().shape().iter().product()
    }

    fn launch<O: DestinationLaunch>(self, sink: impl FnOnce(Unbound) -> O::Operand) {
        let binding = self.tensors.bind(&self.blueprint, &self.grid).map_out(sink);
        self.routine
            .dispatch::<O, _>(&self.tensors, binding, self.blueprint, self.grid);
    }
}
