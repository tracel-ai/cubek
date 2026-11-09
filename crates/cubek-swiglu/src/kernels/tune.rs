//! The SwiGLU autotune: the cube routine's settings, raced per key, over its default as the floor.
//!
//! The autotune is not bounded: the gate reads the projection the step just wrote, so a caller
//! meets it warm, and a round read cold would time a kernel nobody runs.

use crate::kernels::{
    CubeSwigluRoutine, CubeSwigluRoutineSettings, SwigluBinding, SwigluBlueprint, SwigluRoutine,
    SwigluRoutineProblem, SwigluTensors,
};
use cubecl::Device;
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::kind::Write;
use cubecl::tile::launch::{Buffered, DestinationLaunch, Refusal};
use cubecl::tune::{AutotuneOutput, LocalTuner, TunableSet, local_tuner};
use cubecl::{self as cubecl, AutotuneKey, CubeDim, CubeTuneId};
use cubek_matmul::kernels::tensor::{LhsOutput, PackedTensor, PlainTensor};
use cubek_microkernel::row_store::Encoded;
use cubek_std::Hardware;
use cubek_std::autotune::{Autotune, CandidateIdentity, GroupPriority, Identity};
use cubek_std::autotune::{CandidatePriority, TuneCandidate, short_of_the_device};
use cubek_std::routine::{Grid, PreparedLaunches, Routine};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// What makes two gated activations want the same kernel.
///
/// The width of a half is **exact**: it is a `d_ff`, fixed by the model config, and the kernel
/// compiles it. The row count is anchored, being the axis a caller varies.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct SwigluKey {
    elem: ElemType,
    /// The width of a half, which the output takes.
    dim: usize,
    #[autotune(anchor(exp(min = 1, max = 65536, base = 2)))]
    rows: usize,
}

impl SwigluKey {
    /// What a tuning round measures for this key: dense zero rows as many as the key's bucket
    /// stands for, the problem the key describes ([`SwigluRoutineProblem::from`]), or the call as
    /// it came where it holds as many.
    fn representative(&self, call: &SwigluCall) -> SwigluCall {
        let rows = call.both.binding.shape.iter().product::<usize>() / (2 * self.dim);
        if self.rows <= rows {
            return call.clone();
        }
        let both = PlainTensor::zeros(
            &call.client,
            &[self.rows, 2 * self.dim],
            [2 * self.dim, 1],
            call.both.elem,
        );
        SwigluCall {
            client: call.client.clone(),
            both,
        }
    }
}

/// The key of the gate of `both`'s fused rows: what tells it apart is their element, width and
/// count.
impl From<&PlainTensor> for SwigluKey {
    fn from(both: &PlainTensor) -> Self {
        let shape = &both.binding.shape;
        let both_halves = shape[shape.len() - 1];
        let rows = shape.iter().product::<usize>() / both_halves;
        Self::new(both.elem, both_halves / 2, rows)
    }
}

/// The problem a tuning key describes, at its bucket's rows: dense fused rows of the key's
/// element.
impl From<&SwigluKey> for SwigluRoutineProblem {
    fn from(key: &SwigluKey) -> Self {
        Self::dense(key.rows, key.dim, key.elem)
    }
}

/// One gate, `silu(gate) ⊙ up` over `both`, `(..., 2 · d)` laid out `[gate | up]`, as a launch
/// and an autotune read it: the fused rows on `client`'s device.
#[derive(Clone)]
pub struct SwigluCall {
    pub client: Client,
    pub both: PlainTensor,
}

/// What an autotune's candidate hands back: the gate.
#[derive(Clone)]
pub struct SwigluOutput {
    pub handle: TensorHandle,
}

impl AutotuneOutput for SwigluOutput {}

/// What a candidate of the autotune is identified by: the candidate itself, so a launch outside a
/// round reads the launches the winner prepared in it, or the floor by name.
type SwigluIdentity = CandidateIdentity<SwigluRoutineCandidate<CubeSwigluRoutine>>;

/// The autotune's candidates, as the tuner holds them.
type SwigluTunables = Arc<TunableSet<SwigluKey, SwigluCall, SwigluOutput, SwigluIdentity>>;

static TUNER: LocalTuner<SwigluKey, CubeTuneId> = local_tuner!("activation");

/// One candidate of the autotune: a routine of this kind at one of its settings, prepared for the
/// device the autotune is built for, and the launches of the gate written once.
///
/// Its priority is how far the prepared launch's grid falls short of filling the device
/// ([`short_of_the_device`]): a decode step's single row by a run of a few lines is the shape that
/// leaves it idle.
#[derive(Debug, Clone)]
pub struct SwigluRoutineCandidate<R: Routine> {
    routine: R,
    settings: R::Settings,
    hw: Hardware,
    /// The launches this candidate prepared, shared by its clones: they launch the same routine
    /// at the same setting on the same device.
    prepared: Arc<PreparedLaunches<R>>,
}

impl<R: Routine> SwigluRoutineCandidate<R> {
    /// `routine` at `settings`, on `hw`, having prepared nothing yet.
    pub fn new(routine: R, settings: R::Settings, hw: &Hardware) -> Self {
        Self {
            routine,
            settings,
            hw: hw.clone(),
            prepared: Arc::default(),
        }
    }

    /// The setting this candidate launches the routine at.
    pub fn settings(&self) -> &R::Settings {
        &self.settings
    }
}

impl SwigluRoutineCandidate<CubeSwigluRoutine> {
    /// The cube routine at its default, which gates every float row on every device: the
    /// autotune's floor.
    pub fn floor(hw: &Hardware) -> Self {
        Self::new(CubeSwigluRoutine, CubeSwigluRoutineSettings::DEFAULT, hw)
    }
}

impl<R: Routine + Copy> SwigluRoutineCandidate<R> {
    /// Every setting `routine` lists on `hw`, as candidates of the autotune, its default first.
    pub fn list(routine: R, hw: &Hardware) -> Vec<Self> {
        routine
            .settings(hw)
            .into_iter()
            .map(|settings| Self::new(routine, settings, hw))
            .collect()
    }
}

impl<R, M> SwigluRoutineCandidate<R>
where
    R: SwigluRoutine<M> + Routine<Microkernel = M, Blueprint = SwigluBlueprint<M>>,
{
    /// `silu(gate) ⊙ up` over `both`, `(..., 2 · d)` laid out `[gate | up]`, in `f32`, launched
    /// on its device ([`SwigluTensors::new`]).
    ///
    /// # Errors
    ///
    /// Why this device cannot run the gate at this setting.
    pub fn launch(&self, call: &SwigluCall) -> Result<TensorHandle, Refusal> {
        let tensors = SwigluTensors::new(&call.client, call.both.clone())?;
        self.launch_on::<Buffered, _>(&tensors, |blueprint, grid| {
            tensors
                .bind(blueprint, grid)
                .map_out(|out| (out, Write::Replace))
        })?;
        Ok(tensors.into_out())
    }

    /// The gate of [`launch`](Self::launch), written into `out`, a float `[rows, d]` matrix
    /// stored in tiles the caller allocated, the gated rows landing in its tiles as they are
    /// written ([`SwigluTensors::tiled`]).
    ///
    /// # Errors
    ///
    /// Why this device cannot run the gate at this setting, or why `out` cannot hold it.
    pub fn launch_tiled(&self, call: &SwigluCall, out: PlainTensor) -> Result<(), Refusal> {
        let tensors = SwigluTensors::tiled(&call.client, call.both.clone(), out)?;
        self.launch_on::<Buffered, _>(&tensors, |blueprint, grid| {
            tensors
                .bind(blueprint, grid)
                .map_out(|out| (out, Write::Replace))
        })
    }

    /// The gate of [`launch`](Self::launch), encoded into `out`, a packed `[rows, d]` matrix the
    /// caller allocated under one level of block scales along each row: each block's scale
    /// calibrated from the gated rows rounded to their
    /// element, so the result is the gate's output quantized ([`SwigluTensors::packed`]).
    ///
    /// # Errors
    ///
    /// Why this device cannot encode the gate at this setting, or why `out` cannot hold it.
    pub fn launch_packed(&self, call: &SwigluCall, out: PackedTensor) -> Result<(), Refusal> {
        let tensors = SwigluTensors::packed(&call.client, call.both.clone(), out)?;
        self.launch_on::<Encoded, _>(&tensors, |blueprint, grid| tensors.bind(blueprint, grid))
    }

    /// The gate of [`launch`](Self::launch), written into `out` as a product reads its lhs:
    /// [`launch_packed`](Self::launch_packed) or [`launch_tiled`](Self::launch_tiled).
    ///
    /// # Errors
    ///
    /// Those of the launch `out` names.
    pub fn launch_into(&self, call: &SwigluCall, out: LhsOutput) -> Result<(), Refusal> {
        match out {
            LhsOutput::Packed(out) => self.launch_packed(call, out),
            LhsOutput::Tiled(out) => self.launch_tiled(call, out),
        }
    }

    /// The gate on `tensors`, bound by `bind` and written through `O`. No elements is no launch: a
    /// grid of no cubes is one no device spawns.
    fn launch_on<O: DestinationLaunch, Out>(
        &self,
        tensors: &SwigluTensors<Out>,
        bind: impl FnOnce(&SwigluBlueprint<M>, &Grid) -> SwigluBinding<O::Operand>,
    ) -> Result<(), Refusal> {
        if tensors.problem().is_empty() {
            return Ok(());
        }
        let (blueprint, grid) = self.prepare(tensors.problem())?;
        let binding = bind(&blueprint, &grid);
        self.routine
            .dispatch::<O, _>(tensors, binding, blueprint, grid);
        Ok(())
    }

    /// The launch this candidate prepared for `problem`, or prepares now and holds.
    fn prepare(
        &self,
        problem: &SwigluRoutineProblem,
    ) -> Result<(SwigluBlueprint<M>, Grid), Refusal> {
        self.prepared
            .prepare(&self.routine, &self.hw, problem, &self.settings)
    }

    /// The launch this candidate prepares for the problem `key` describes, or `None` where the
    /// routine declines it on this device. Not held: a key's problem places the candidate in a
    /// round, and may never be launched.
    fn prepared(&self, key: &SwigluKey) -> Option<(SwigluBlueprint<M>, Grid)> {
        self.routine
            .prepare(&self.hw, &SwigluRoutineProblem::from(key), &self.settings)
            .ok()
    }

    /// The gate of `call` at this candidate's setting.
    fn run(&self, call: SwigluCall) -> Result<SwigluOutput, String> {
        self.launch(&call)
            .map(|handle| SwigluOutput { handle })
            .map_err(|error| error.to_string())
    }
}

/// The candidate's name: its setting's, which is what a persisted winner is checksummed by.
impl<R: Routine<Settings: core::fmt::Display>> core::fmt::Display for SwigluRoutineCandidate<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.settings.fmt(f)
    }
}

impl<R, M: PartialEq> TuneCandidate for SwigluRoutineCandidate<R>
where
    R: SwigluRoutine<M> + Routine<Microkernel = M, Blueprint = SwigluBlueprint<M>>,
{
    type Strategy = R::Settings;
    type Key = SwigluKey;
    /// The blueprint and the cube: the cube is kept out of a blueprint, and two settings preparing
    /// one blueprint can still launch different cubes.
    type Blueprint = (SwigluBlueprint<M>, CubeDim);

    fn strategy(&self) -> &R::Settings {
        &self.settings
    }

    /// First, lowered a level for each halving the prepared grid falls short of filling the
    /// device; last where the routine declines the problem, which it then does at launch before
    /// compiling anything.
    fn priority(&self, key: &SwigluKey) -> CandidatePriority {
        match self.prepared(key) {
            Some((_, grid)) => {
                CandidatePriority::FIRST.lowered(short_of_the_device(&self.hw, &grid.launch))
            }
            None => CandidatePriority::LAST,
        }
    }

    fn blueprint(&self, key: &SwigluKey) -> Option<Self::Blueprint> {
        self.prepared(key)
            .map(|(blueprint, grid)| (blueprint, grid.launch.cube_dim))
    }
}

/// The gate of `call` through the autotune, its candidates grouped at priority `tile` on `device`.
pub fn launch_tuned(call: SwigluCall, device: &Device, tile: GroupPriority) -> TensorHandle {
    // Nothing to gate is nothing to tune: an empty input shares the key of a single row, and a
    // round measured on it would time no launch and keep a winner for the decode bucket.
    if call.both.binding.shape.contains(&0) {
        let hw = Hardware::from_client(&call.client);
        return SwigluRoutineCandidate::floor(&hw)
            .launch(&call)
            .expect("the floor gates every float row");
    }
    let client = call.client.clone();
    // Keyed by the device: the set is built from what this device supports.
    let id = CubeTuneId::new(&client, device);
    let tunables = tunables(&id, &client, tile);
    TUNER.execute(&id, &client, tunables, call).handle
}

/// The gate of `call`'s rows written into `out`, a matrix of its rows as a product reads its lhs
/// ([`SwigluRoutineCandidate::launch_into`]): at the setting the autotune elected for the rows, so
/// the output inherits the kernel the unfused round chose, or at the default, the floor, where no
/// round has settled or the elected setting cannot write it — a stage that outgrows the device. A
/// refusal is a launch's before it dispatches, so the floor writes into the same `out`.
///
/// # Errors
///
/// What the floor refuses: a scheme the store does not write, a device without plane
/// instructions, a run too long for a cube's stage, an output that is not the rows in tiles.
pub fn launch_into_tuned(
    call: &SwigluCall,
    device: &Device,
    tile: GroupPriority,
    out: LhsOutput,
) -> Result<(), Refusal> {
    if let Some(elected) = fastest_swiglu_routine(call, device, tile)
        && elected.launch_into(call, out.clone()).is_ok()
    {
        return Ok(());
    }
    let hw = Hardware::from_client(&call.client);
    SwigluRoutineCandidate::floor(&hw).launch_into(call, out)
}

/// The candidate the autotune found fastest for the gate of `call`'s rows on `device`, or `None`
/// where no round has settled the key, or the floor named apart won it.
fn fastest_swiglu_routine(
    call: &SwigluCall,
    device: &Device,
    tile: GroupPriority,
) -> Option<SwigluRoutineCandidate<CubeSwigluRoutine>> {
    let id = CubeTuneId::new(&call.client, device);
    let tunables = tunables(&id, &call.client, tile);
    TUNER
        .fastest_identity(&id, &tunables, &SwigluKey::from(&call.both))?
        .routine()
}

/// The autotune for the device `id` names, built on first touch, its candidates grouped at
/// priority `tile`. The one place it is built: a tuner caches a set per initializer *type*, so a
/// second `init` site would build a second set.
fn tunables(id: &CubeTuneId, client: &Client, tile: GroupPriority) -> SwigluTunables {
    let hw = Hardware::from_client(client);
    TUNER.init(id, move || candidates(hw.clone(), tile))
}

/// The routine's candidates on `hw`, identified by their setting, over its default as the floor.
fn candidates(
    hw: Hardware,
    priority: GroupPriority,
) -> TunableSet<SwigluKey, SwigluCall, SwigluOutput, SwigluIdentity> {
    let mut autotune = Autotune::<SwigluKey, SwigluCall, SwigluOutput, SwigluIdentity>::measured_on(
        |call: &SwigluCall| SwigluKey::from(&call.both),
        "as-keyed",
        SwigluKey::representative,
    );
    let tile = autotune.group("tile", priority);
    let floor = SwigluRoutineCandidate::floor(&hw);
    autotune
        .identified(
            &tile,
            SwigluRoutineCandidate::list(CubeSwigluRoutine, &hw),
            |candidate| SwigluIdentity::Routine(candidate.clone()),
            |candidate, call| candidate.run(call),
        )
        // The default, which runs everywhere: taking `tile` out of the autotune stops the family
        // being measured, not being served.
        .floor(SwigluIdentity::named(
            "cube-swiglu-floor",
            move |call: SwigluCall| floor.run(call),
        ))
        .finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::FloatKind;
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    /// On devices a dev machine cannot show, every setting prepares: the routine is the floor, so a
    /// decline would leave a key with nothing to measure.
    #[test]
    fn every_device_prepares_every_setting() {
        for key in [
            SwigluKey::new(F16, 8192, 1),
            SwigluKey::new(F16, 2816, 512 * 4),
            SwigluKey::new(F16, 111, 3),
        ] {
            for hw in [
                devices::rdna_range(),
                devices::cdna_wave64(),
                devices::h100(),
                devices::wgpu_no_subgroups(),
                devices::cpu_12(),
            ] {
                for candidate in SwigluRoutineCandidate::list(CubeSwigluRoutine, &hw) {
                    assert!(
                        candidate.blueprint(&key).is_some(),
                        "{}",
                        candidate.strategy()
                    );
                }
            }
        }
    }

    /// A decode step's single row by the default's run falls short of the device, and a prompt's
    /// rows fill it.
    #[test]
    fn the_priority_reads_how_far_the_grid_falls_short() {
        let hw = devices::strix_halo();
        let lead = SwigluRoutineCandidate::list(CubeSwigluRoutine, &hw).swap_remove(0);
        assert_eq!(
            lead.priority(&SwigluKey::new(F16, 2816, 512 * 4)),
            CandidatePriority::FIRST
        );
        assert!(lead.priority(&SwigluKey::new(F16, 2816, 1)) < CandidatePriority::FIRST);
    }
}
