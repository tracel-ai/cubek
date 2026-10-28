use crate::kernels::MatmulBinding;
use crate::kernels::tensor::{PlainTensor, StoredTensor};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_microkernel::matmul::cube::MatmulCubeWrite;
use cubek_std::Hardware;

use crate::kernels::{
    MatmulBlueprint, MatmulIndexTable, MatmulOutput, MatmulRoutineProblem, MatmulTensors,
};
use cubecl::tile::{
    Geometry, TileArgLaunch,
    kind::Write,
    launch::{Buffered, DestinationLaunch, SinkLaunch, Unbound},
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
/// every row whole or tiles them, and however each input is stored: its kernel reads each input
/// as the tile it is bound as. A routine declines a problem whose inputs it does not read.
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

    /// Launch the kernel `blueprint` compiles, on `grid`, replacing `tensors`' output buffer: what
    /// every launch below runs once it has prepared.
    fn dispatch_into_buffer(
        &self,
        tensors: &MatmulTensors,
        blueprint: MatmulBlueprint<M, SM>,
        grid: Grid,
    ) {
        let binding = tensors
            .bind(&blueprint, &grid)
            .map_out(|out| (out, Write::Replace));
        self.dispatch::<Buffered, _>(tensors, binding, blueprint, grid);
    }

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
        self.dispatch_into_buffer(&tensors, blueprint, grid);
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
        lhs: StoredTensor,
        rhs: StoredTensor,
        out: PlainTensor,
        settings: &Self::Settings,
    ) -> Result<(), Refusal> {
        let tensors = MatmulTensors::writing_into::<M>(client, lhs, rhs, out)?;
        let (blueprint, grid) = self.prepare(hw, &tensors.problem(), settings)?;
        self.dispatch_into_buffer(&tensors, blueprint, grid);
        Ok(())
    }

    /// `lhs · rhs` on `client`'s device, every batch reading its entry of `table`, written into
    /// `out`, `[batches.., rows, n]`, which the caller allocated as its product lies, prepared for
    /// `hw` at `settings` ([`MatmulTensors::indexed`]).
    ///
    /// # Errors
    ///
    /// As [`MatmulTensors::indexed`], and why `hw` cannot run the product at these settings,
    /// [`MATMUL_INDEX_UNREAD`](crate::kernels::MATMUL_INDEX_UNREAD) for a routine whose kernel walks
    /// no indexed input among them.
    // The indexed operands are the dense ones beside the table and the output the caller laid out,
    // each named where the caller hands it.
    #[allow(clippy::too_many_arguments)]
    fn launch_indexed_into(
        &self,
        hw: &Hardware,
        client: &Client,
        lhs: PlainTensor,
        rhs: StoredTensor,
        table: MatmulIndexTable,
        out: PlainTensor,
        settings: &Self::Settings,
    ) -> Result<(), Refusal> {
        let tensors = MatmulTensors::indexed::<M>(client, lhs, rhs, table, out)?;
        let (blueprint, grid) = self.prepare(hw, &tensors.problem(), settings)?;
        self.dispatch_into_buffer(&tensors, blueprint, grid);
        Ok(())
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
        self.dispatch_into_buffer(&tensors, blueprint, grid);
        Ok(tensors.into_out().expect(ALLOCATED))
    }

    /// [`launch_first_served_on`](Self::launch_first_served_on) into an output of `out`: what a
    /// caller naming the routine and no setting runs on a packed lhs
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
        self.dispatch_into_buffer(&tensors, blueprint, grid);
        Ok(tensors.into_out().expect(ALLOCATED))
    }
}

/// A product prepared for a sink the caller builds afterwards: the routine at its setting, its
/// operands, and the launch it settled, bound but for the output, whose spec and geometry the sink
/// is built from ([`SinkLaunch::launch`]).
///
/// A fused element-wise tail is one such sink: its block stores in lines of the width the kernel
/// settled ([`SinkLaunch::width`]), so the caller plans it around this launch before binding it.
pub struct MatmulSinkLaunch<R, M, SM = ()> {
    routine: R,
    tensors: MatmulTensors<Geometry>,
    blueprint: MatmulBlueprint<M, SM>,
    grid: Grid,
    binding: MatmulBinding<Unbound, TileArgLaunch, TileArgLaunch>,
}

impl<R: MatmulRoutine<M, SM>, M: MatmulMicrokernel, SM> MatmulSinkLaunch<R, M, SM> {
    /// `tensors`' product on `routine`, launched as `blueprint` and `grid` state it, its operands
    /// bound now.
    ///
    /// # Errors
    ///
    /// [`MATMUL_SINK_TAKES_NO_RELAY`], for a routine whose cubes reach the output in a way no
    /// sink takes.
    pub fn new(
        routine: R,
        tensors: MatmulTensors<Geometry>,
        blueprint: MatmulBlueprint<M, SM>,
        grid: Grid,
    ) -> Result<Self, Refusal> {
        if !R::WRITE.takes_sink() {
            return Err(MATMUL_SINK_TAKES_NO_RELAY);
        }
        let binding = tensors.bind(&blueprint, &grid);
        Ok(Self {
            routine,
            tensors,
            blueprint,
            grid,
            binding,
        })
    }
}

impl<R, M, SM> core::fmt::Debug for MatmulSinkLaunch<R, M, SM>
where
    R: core::fmt::Debug,
    M: core::fmt::Debug,
    SM: core::fmt::Debug,
{
    // The binding is left out: the tile DSL's `Launcher` is not `Debug`.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MatmulSinkLaunch")
            .field("routine", &self.routine)
            .field("tensors", &self.tensors)
            .field("blueprint", &self.blueprint)
            .field("grid", &self.grid)
            .finish_non_exhaustive()
    }
}

impl<R: MatmulRoutine<M, SM>, M: MatmulMicrokernel, SM> SinkLaunch for MatmulSinkLaunch<R, M, SM> {
    fn width(&self) -> usize {
        self.blueprint.widths.out
    }

    fn elements(&self) -> usize {
        self.tensors.out().shape().iter().product()
    }

    fn launch<O: DestinationLaunch>(self, sink: impl FnOnce(Unbound) -> O::Operand) {
        let binding = self.binding.map_out(sink);
        self.routine
            .dispatch::<O, _>(&self.tensors, binding, self.blueprint, self.grid);
    }
}
