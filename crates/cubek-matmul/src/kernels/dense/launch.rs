use crate::kernels::MatmulBinding;
use crate::kernels::tensor::{PlainTensor, StoredTensor};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Decline;
use cubek_microkernel::matmul::cube::MatmulCubeWrite;
use cubek_microkernel::matmul::{MatmulMicrokernel, MatmulRowsKind};
use cubek_std::Hardware;

use crate::kernels::{
    MATMUL_TENSORS_READ_FOR_OTHER_ROWS, MatmulBlueprint, MatmulIndexTable, MatmulOutput,
    MatmulRoutineProblem, MatmulTensors, TableStored,
};
use cubecl::tile::{
    Geometry, Launcher, TileTensorLaunch,
    launch::{DestinationLaunch, SinkLaunch, Unbound},
};
use cubek_std::routine::{Grid, Routine};

/// What a launch that allocated its output hands back.
const ALLOCATED: &str = "a launch allocating its output hands it back";

/// A routine around microkernel `M`, laying out shared memory as `SM`: it prepares a
/// [`MatmulBlueprint`] and launches it on tensors, each as it is stored ([`StoredTensor`]).
///
/// A routine states what is its own, how it places its microkernel's work items and the kernel it
/// dispatches, and the launch around those is written once, here, whether the microkernel holds
/// every row whole or tiles them, and however each input is stored: its kernel reads each input
/// as the tile it is bound as. A routine declines a problem whose inputs it does not read.
///
/// # Assumptions
///
/// [`bind_extras`](Self::bind_extras) is handed tensors a launch holding its rows as
/// [`ROWS_KIND`](Self::ROWS_KIND) says reads ([`MatmulTensors::is_read_for`]), and
/// [`dispatch`](Self::dispatch) the binding [`MatmulTensors::bind`] and the extras
/// [`bind_extras`](Self::bind_extras) returned for the same tensors, blueprint and grid.
///
/// # Guarantees
///
/// [`dispatch`](Self::dispatch) launches whatever it is handed: every refusal of a launch is met
/// while binding, before its kernel launches.
pub trait MatmulRoutine<M: MatmulMicrokernel, SM = ()>:
    Routine<Problem = MatmulRoutineProblem, Microkernel = M, Blueprint = MatmulBlueprint<M, SM>>
{
    /// How its launches hold a product's rows: every row whole, compiled into the kernel, or a
    /// tile of them a box. It decides how its tensors and its view read the problem, before any
    /// microkernel is chosen.
    const ROWS_KIND: MatmulRowsKind;

    /// How the routine's cubes reach the output.
    const WRITE: MatmulCubeWrite = MatmulCubeWrite::Replace;

    /// What its kernel reads beside the three operands its tensors bind, bound with them before
    /// its kernel launches: `()` for a kernel that reads nothing more.
    type Extras;

    /// What its kernel reads beside `tensors`' operands, bound under `launcher` as `blueprint`
    /// states it: what [`dispatch`](Self::dispatch) launches beside them. Binding them may
    /// allocate scratch and zero it.
    ///
    /// # Errors
    ///
    /// [`Decline::Broken`] where binding finds false what `prepare` claimed of the launch.
    fn bind_extras<Out: MatmulOutput>(
        &self,
        tensors: &MatmulTensors<Out>,
        launcher: &Launcher,
        blueprint: &MatmulBlueprint<M, SM>,
    ) -> Result<Self::Extras, Decline>;

    /// Launch the kernel `blueprint` compiles, on `grid`, over `binding` and `extras`, the latter
    /// as [`bind_extras`](Self::bind_extras) bound them, its kernel reaching the output through the
    /// destination `O`, whatever `tensors`' output is.
    fn dispatch<O: DestinationLaunch, Out: MatmulOutput>(
        &self,
        tensors: &MatmulTensors<Out>,
        binding: MatmulBinding<O::Operand, TileTensorLaunch, TileTensorLaunch>,
        extras: Self::Extras,
        blueprint: MatmulBlueprint<M, SM>,
        grid: Grid,
    );

    /// Launch the kernel `blueprint` compiles, on `grid`, replacing `tensors`' output buffer: what
    /// every launch below runs once it has prepared.
    ///
    /// # Errors
    ///
    /// [`MATMUL_TENSORS_READ_FOR_OTHER_ROWS`]. As [`MatmulTensors::bind`] and
    /// [`bind_extras`](Self::bind_extras).
    fn dispatch_into_buffer(
        &self,
        tensors: &MatmulTensors,
        blueprint: MatmulBlueprint<M, SM>,
        grid: Grid,
    ) -> Result<(), Decline> {
        if !tensors.is_read_for(Self::ROWS_KIND) {
            return Err(MATMUL_TENSORS_READ_FOR_OTHER_ROWS);
        }
        let binding = tensors
            .bind(&blueprint, &grid)?
            .try_map_out(|out| tensors.output_through_table(out))?;
        let extras = self.bind_extras(tensors, &binding.launcher, &blueprint)?;
        self.dispatch::<TableStored, _>(tensors, binding, extras, blueprint, grid);
        Ok(())
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
    ) -> Result<TensorHandle, Decline> {
        let tensors = MatmulTensors::new(client, lhs, rhs, Self::ROWS_KIND)?;
        let (blueprint, grid) = Self::prepare(hw, &tensors.problem(), settings)?;
        self.dispatch_into_buffer(&tensors, blueprint, grid)?;
        Ok(tensors.into_out().expect(ALLOCATED))
    }

    /// `lhs · rhs` on `client`'s device, prepared for `hw` at `settings` and for a sink the caller
    /// binds afterwards ([`MatmulSinkLaunch`]). Nothing is compiled until the launch is.
    ///
    /// # Errors
    ///
    /// What [`MatmulTensors::sink`] declines, and why `hw` cannot run the product at these
    /// settings.
    fn sink_launch_on(
        self,
        hw: &Hardware,
        client: &Client,
        lhs: StoredTensor,
        rhs: StoredTensor,
        settings: &Self::Settings,
    ) -> Result<MatmulSinkLaunch<Self, M, SM>, Decline>
    where
        Self: Sized,
    {
        let tensors = MatmulTensors::sink(client, lhs, rhs, Self::ROWS_KIND)?;
        let (blueprint, grid) = Self::prepare(hw, &tensors.problem(), settings)?;
        MatmulSinkLaunch::new(self, tensors, blueprint, grid)
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
    ) -> Result<(), Decline> {
        let tensors = MatmulTensors::writing_into(client, lhs, rhs, out, Self::ROWS_KIND)?;
        let (blueprint, grid) = Self::prepare(hw, &tensors.problem(), settings)?;
        self.dispatch_into_buffer(&tensors, blueprint, grid)
    }

    /// `lhs · rhs` on `client`'s device, every batch reading its entry of `table`, written into
    /// `out`, `[batches.., rows, n]`, which the caller allocated as its product lies, prepared for
    /// `hw` at `settings` ([`MatmulTensors::indexed`]). Under
    /// [`MatmulIndex::GroupedRows`](crate::kernels::MatmulIndex::GroupedRows) the lhs and `out`
    /// are the flat `[rows, k]` and `[rows, n]`, and every row group must be at most the table's
    /// `max_rows` rows, or its later output rows are left unwritten.
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
    ) -> Result<(), Decline> {
        let tensors = MatmulTensors::indexed(client, lhs, rhs, table, out, Self::ROWS_KIND)?;
        let (blueprint, grid) = Self::prepare(hw, &tensors.problem(), settings)?;
        self.dispatch_into_buffer(&tensors, blueprint, grid)
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
    ) -> Result<TensorHandle, Decline> {
        let tensors = MatmulTensors::new(client, lhs, rhs, Self::ROWS_KIND)?;
        let (blueprint, grid) = Self::prepare_first_served(hw, &tensors.problem())?;
        self.dispatch_into_buffer(&tensors, blueprint, grid)?;
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
    ) -> Result<TensorHandle, Decline> {
        let tensors = MatmulTensors::written_in(client, lhs, rhs, out)?;
        let (blueprint, grid) = Self::prepare_first_served(hw, &tensors.problem())?;
        self.dispatch_into_buffer(&tensors, blueprint, grid)?;
        Ok(tensors.into_out().expect(ALLOCATED))
    }
}

/// A product prepared for a sink the caller builds afterwards: the routine at its setting, its
/// operands, and the launch it settled, bound but for the output, whose spec and geometry the sink
/// is built from ([`SinkLaunch::launch`]).
///
/// A fused element-wise tail is one such sink: its block stores in lines of the width the kernel
/// settled ([`SinkLaunch::width`]), so the caller plans it around this launch before binding it.
pub struct MatmulSinkLaunch<R: MatmulRoutine<M, SM>, M: MatmulMicrokernel, SM = ()> {
    routine: R,
    tensors: MatmulTensors<Geometry>,
    blueprint: MatmulBlueprint<M, SM>,
    grid: Grid,
    binding: MatmulBinding<Unbound, TileTensorLaunch, TileTensorLaunch>,
    extras: R::Extras,
}

impl<R: MatmulRoutine<M, SM>, M: MatmulMicrokernel, SM> MatmulSinkLaunch<R, M, SM> {
    /// `tensors`' product on `routine`, launched as `blueprint` and `grid` state it, its operands
    /// bound now.
    ///
    /// # Errors
    ///
    /// [`MATMUL_TENSORS_READ_FOR_OTHER_ROWS`]. As [`MatmulTensors::bind`] and
    /// [`MatmulRoutine::bind_extras`].
    pub fn new(
        routine: R,
        tensors: MatmulTensors<Geometry>,
        blueprint: MatmulBlueprint<M, SM>,
        grid: Grid,
    ) -> Result<Self, Decline> {
        if !tensors.is_read_for(R::ROWS_KIND) {
            return Err(MATMUL_TENSORS_READ_FOR_OTHER_ROWS);
        }
        let binding = tensors.bind(&blueprint, &grid)?;
        let extras = routine.bind_extras(&tensors, &binding.launcher, &blueprint)?;
        Ok(Self {
            routine,
            tensors,
            blueprint,
            grid,
            binding,
            extras,
        })
    }
}

impl<R, M, SM> core::fmt::Debug for MatmulSinkLaunch<R, M, SM>
where
    R: MatmulRoutine<M, SM> + core::fmt::Debug,
    M: MatmulMicrokernel + core::fmt::Debug,
    SM: core::fmt::Debug,
{
    // The binding and the extras are left out: the tile DSL's `Launcher` is not `Debug`.
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
        self.routine.dispatch::<O, _>(
            &self.tensors,
            binding,
            self.extras,
            self.blueprint,
            self.grid,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::UnitRegistersMatmulRoutine;
    use cubecl::ir::FloatKind;
    use cubek_test_utils::client;

    /// Zeros of `shape` in `f32`.
    fn zeros(shape: &[usize]) -> StoredTensor {
        let element = ElemType::Float(FloatKind::F32);
        StoredTensor::Plain(PlainTensor::of_handle(TensorHandle::zeros(
            &client(),
            shape.to_vec(),
            element,
        )))
    }

    #[test]
    fn tensors_read_for_other_rows_decline() {
        let (client, hw) = (client(), Hardware::from_client(&client()));
        let tensors = |rows_kind| {
            MatmulTensors::new(&client, zeros(&[4, 8]), zeros(&[8, 16]), rows_kind)
                .expect("dense float operands")
        };
        let (blueprint, grid) = UnitRegistersMatmulRoutine::prepare_first_served(
            &hw,
            &tensors(MatmulRowsKind::Tiled).problem(),
        )
        .expect("the unit routine serves a small product");

        let declined = UnitRegistersMatmulRoutine
            .dispatch_into_buffer(&tensors(MatmulRowsKind::Whole), blueprint, grid)
            .err();

        assert_eq!(declined, Some(MATMUL_TENSORS_READ_FOR_OTHER_ROWS));
    }

    #[test]
    fn a_sink_read_for_other_rows_declines() {
        let (client, hw) = (client(), Hardware::from_client(&client()));
        let tiled = MatmulTensors::new(
            &client,
            zeros(&[4, 8]),
            zeros(&[8, 16]),
            MatmulRowsKind::Tiled,
        )
        .expect("dense float operands");
        let (blueprint, grid) =
            UnitRegistersMatmulRoutine::prepare_first_served(&hw, &tiled.problem())
                .expect("the unit routine serves a small product");
        let sink = MatmulTensors::sink(
            &client,
            zeros(&[4, 8]),
            zeros(&[8, 16]),
            MatmulRowsKind::Whole,
        )
        .expect("dense float operands");

        let declined =
            MatmulSinkLaunch::new(UnitRegistersMatmulRoutine, sink, blueprint, grid).err();

        assert_eq!(declined, Some(MATMUL_TENSORS_READ_FOR_OTHER_ROWS));
    }
}
