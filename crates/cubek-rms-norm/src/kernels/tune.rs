//! The RMS-norm autotune: every routine's settings, raced per key, over the cube groups' floor.
//!
//! The autotune is not bounded: a norm reads the activations the step just wrote, so a caller
//! meets it warm, and a round read cold would time a kernel nobody runs.

use crate::kernels::{
    CubeGroupsRmsNormRoutine, CubeGroupsRmsNormRoutineSettings, CubeRmsNormRoutine,
    CubeRmsNormRoutineSettings, PlaneRmsNormRoutine, PlaneRmsNormRoutineSettings,
    RMS_NORM_SINK_TAKES_NO_ROWS, RmsNormBinding, RmsNormBlueprint, RmsNormRoutine,
    RmsNormRoutineProblem, RmsNormRoutineSinkLaunch, RmsNormTensors,
};
use cubecl::Device;
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::kind::Write;
use cubecl::tile::launch::{Buffered, DestinationLaunch, Refusal, SinkLaunchVisitor};
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

/// What makes two norms want the same kernel.
///
/// The row width is **exact**: it is a `d_model` or a `head_dim`, fixed by the model config, and
/// every owner sizes its registers by it. The row count is anchored — it is the axis a caller
/// varies freely (batch × sequence) and the one that decides whether the grid is full.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct RmsNormKey {
    elem: ElemType,
    /// The normalized row's width.
    dim: usize,
    #[autotune(anchor(exp(min = 1, max = 65536, base = 2)))]
    rows: usize,
}

impl RmsNormKey {
    /// What a tuning round measures for this key: the call's weight, and dense zero rows as many
    /// as the key's bucket stands for, the problem the key describes
    /// ([`RmsNormRoutineProblem::from`]). A weight a row scales its own
    /// rows' count and no other, so its round measures the call as it came.
    fn representative(&self, call: &RmsNormCall) -> RmsNormCall {
        let rows = call.input.binding.shape.iter().product::<usize>() / self.dim;
        let weight_a_row = call.weight.binding.shape.iter().product::<usize>() > self.dim;
        if self.rows <= rows || weight_a_row {
            return call.clone();
        }
        let input = PlainTensor::zeros(
            &call.client,
            &[self.rows, self.dim],
            [self.dim, 1],
            call.input.elem,
        );
        RmsNormCall {
            input,
            ..call.clone()
        }
    }
}

/// The key of the norm of `x`'s rows: what tells it apart is the rows' element, width and count.
impl From<&PlainTensor> for RmsNormKey {
    fn from(x: &PlainTensor) -> Self {
        let shape = &x.binding.shape;
        let d = shape[shape.len() - 1];
        let rows = shape.iter().product::<usize>() / d;
        Self::new(x.elem, d, rows)
    }
}

/// The problem a tuning key describes, at its bucket's rows: dense operands of the key's element.
impl From<&RmsNormKey> for RmsNormRoutineProblem {
    fn from(key: &RmsNormKey) -> Self {
        Self::dense(key.rows, key.dim, key.elem)
    }
}

/// One norm, `input / sqrt(mean(input², last) + epsilon) · weight`, as a launch and an autotune
/// read it: the rows and the weight on `client`'s device.
#[derive(Clone)]
pub struct RmsNormCall {
    pub client: Client,
    pub input: PlainTensor,
    pub weight: PlainTensor,
    pub epsilon: f32,
}

/// What an autotune's candidate hands back: the norm.
#[derive(Clone)]
pub struct RmsNormOutput {
    pub handle: TensorHandle,
}

impl AutotuneOutput for RmsNormOutput {}

/// A routine of this kind at one of its settings, as a value: what a fused tail's election hashes,
/// and what names a candidate ([`RmsNormCandidate::settings`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RmsNormRoutineSettings {
    /// [`CubeGroupsRmsNormRoutine`], a candidate and the autotune's floor.
    CubeGroups(CubeGroupsRmsNormRoutineSettings),
    /// [`PlaneRmsNormRoutine`], a candidate.
    Plane(PlaneRmsNormRoutineSettings),
    /// [`CubeRmsNormRoutine`], a candidate.
    Cube(CubeRmsNormRoutineSettings),
}

/// A candidate of the autotune, whichever routine it launches: what the autotune identifies a
/// candidate by, and what the tuner hands back for the one it found fastest
/// ([`fastest_rms_norm_routine`]), so a launch outside a round reads the launches the winner
/// prepared in it.
#[derive(Debug, Clone)]
pub enum RmsNormCandidate {
    /// [`CubeGroupsRmsNormRoutine`], a candidate and the autotune's floor.
    CubeGroups(RmsNormRoutineCandidate<CubeGroupsRmsNormRoutine>),
    /// [`PlaneRmsNormRoutine`], a candidate.
    Plane(RmsNormRoutineCandidate<PlaneRmsNormRoutine>),
    /// [`CubeRmsNormRoutine`], a candidate.
    Cube(RmsNormRoutineCandidate<CubeRmsNormRoutine>),
}

impl RmsNormCandidate {
    /// The routine this candidate launches, at its setting.
    pub fn settings(&self) -> RmsNormRoutineSettings {
        match self {
            Self::CubeGroups(candidate) => (*candidate.settings()).into(),
            Self::Plane(candidate) => (*candidate.settings()).into(),
            Self::Cube(candidate) => (*candidate.settings()).into(),
        }
    }

    /// The norm of `input`'s rows scaled by `weight`, written into `out` as a product reads its
    /// lhs ([`RmsNormRoutineCandidate::launch_into`]).
    ///
    /// # Errors
    ///
    /// Why the routine cannot write the norm into `out` at its setting — the plane routine, whose
    /// planes no cube holds, never encodes — or why `out` cannot hold it.
    pub fn launch_into(&self, call: &RmsNormCall, out: LhsOutput) -> Result<(), Refusal> {
        match self {
            Self::CubeGroups(candidate) => candidate.launch_into(call, out),
            Self::Plane(candidate) => candidate.launch_into(call, out),
            Self::Cube(candidate) => candidate.launch_into(call, out),
        }
    }

    /// The norm of `input`'s rows scaled by `weight` prepared for a sink
    /// ([`RmsNormRoutineCandidate::sink_launch`]), and given to `visitor`.
    pub fn sink_launch<V: SinkLaunchVisitor>(&self, call: &RmsNormCall, visitor: V) -> V::Output {
        match self {
            Self::CubeGroups(candidate) => visitor.visit(candidate.sink_launch(call)),
            Self::Plane(candidate) => visitor.visit(candidate.sink_launch(call)),
            Self::Cube(candidate) => visitor.visit(candidate.sink_launch(call)),
        }
    }
}

/// The candidate's name: its setting's.
impl core::fmt::Display for RmsNormCandidate {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.settings().fmt(f)
    }
}

impl From<RmsNormRoutineCandidate<CubeGroupsRmsNormRoutine>> for RmsNormCandidate {
    fn from(candidate: RmsNormRoutineCandidate<CubeGroupsRmsNormRoutine>) -> Self {
        Self::CubeGroups(candidate)
    }
}

impl From<RmsNormRoutineCandidate<PlaneRmsNormRoutine>> for RmsNormCandidate {
    fn from(candidate: RmsNormRoutineCandidate<PlaneRmsNormRoutine>) -> Self {
        Self::Plane(candidate)
    }
}

impl From<RmsNormRoutineCandidate<CubeRmsNormRoutine>> for RmsNormCandidate {
    fn from(candidate: RmsNormRoutineCandidate<CubeRmsNormRoutine>) -> Self {
        Self::Cube(candidate)
    }
}

/// The routine's own settings' name: what the autotune names the candidate by, and so what a
/// persisted winner is checksummed by.
impl core::fmt::Display for RmsNormRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CubeGroups(settings) => settings.fmt(f),
            Self::Plane(settings) => settings.fmt(f),
            Self::Cube(settings) => settings.fmt(f),
        }
    }
}

impl From<CubeGroupsRmsNormRoutineSettings> for RmsNormRoutineSettings {
    fn from(settings: CubeGroupsRmsNormRoutineSettings) -> Self {
        Self::CubeGroups(settings)
    }
}

impl From<PlaneRmsNormRoutineSettings> for RmsNormRoutineSettings {
    fn from(settings: PlaneRmsNormRoutineSettings) -> Self {
        Self::Plane(settings)
    }
}

impl From<CubeRmsNormRoutineSettings> for RmsNormRoutineSettings {
    fn from(settings: CubeRmsNormRoutineSettings) -> Self {
        Self::Cube(settings)
    }
}

/// What a candidate of the autotune is identified by: the candidate itself, or the floor by name.
type RmsNormIdentity = CandidateIdentity<RmsNormCandidate>;

/// The autotune's candidates, as the tuner holds them.
type RmsNormTunables = Arc<TunableSet<RmsNormKey, RmsNormCall, RmsNormOutput, RmsNormIdentity>>;

static TUNER: LocalTuner<RmsNormKey, CubeTuneId> = local_tuner!("norm");

/// One candidate of the autotune: a routine of this kind at one of its settings, prepared for the
/// device the autotune is built for, and the launches of the norm written once.
///
/// Its priority is how far the prepared launch's grid falls short of filling the device
/// ([`short_of_the_device`]): a norm of a few rows by cubes of one row is the shape that leaves it
/// idle.
#[derive(Debug, Clone)]
pub struct RmsNormRoutineCandidate<R: Routine> {
    routine: R,
    settings: R::Settings,
    hw: Hardware,
    /// The launches this candidate prepared, shared by its clones: they launch the same routine
    /// at the same setting on the same device.
    prepared: Arc<PreparedLaunches<R>>,
}

impl<R: Routine> RmsNormRoutineCandidate<R> {
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

impl RmsNormRoutineCandidate<CubeGroupsRmsNormRoutine> {
    /// The cube groups at their default, which serve every float row on every device: the
    /// autotune's floor.
    pub fn floor(hw: &Hardware) -> Self {
        Self::new(
            CubeGroupsRmsNormRoutine,
            CubeGroupsRmsNormRoutineSettings::DEFAULT,
            hw,
        )
    }
}

impl<R: Routine + Copy> RmsNormRoutineCandidate<R> {
    /// Every setting `routine` lists on `hw`, as candidates of the autotune, its default first.
    pub fn list(routine: R, hw: &Hardware) -> Vec<Self> {
        routine
            .settings(hw)
            .into_iter()
            .map(|settings| Self::new(routine, settings, hw))
            .collect()
    }
}

impl<R, M> RmsNormRoutineCandidate<R>
where
    R: RmsNormRoutine<M> + Routine<Microkernel = M, Blueprint = RmsNormBlueprint<M>>,
{
    /// `input / sqrt(mean(input², last) + epsilon) · weight`, the square-sum in `f32`, launched on
    /// the operands' device ([`RmsNormTensors::new`]).
    ///
    /// # Errors
    ///
    /// Why this device cannot run the norm at this setting.
    pub fn launch(&self, call: &RmsNormCall) -> Result<TensorHandle, Refusal> {
        let tensors = RmsNormTensors::new(&call.client, call.input.clone(), call.weight.clone())?;
        self.launch_on::<Buffered, _>(call, &tensors, |blueprint, grid| {
            tensors
                .bind(blueprint, grid)
                .map_out(|out| (out, Write::Replace))
        })?;
        Ok(tensors.into_out())
    }

    /// The norm of [`launch`](Self::launch), written into `out`, a float `[rows, dim]` matrix
    /// stored in tiles the caller allocated, its rows landing in its tiles as they are written
    /// ([`RmsNormTensors::tiled`]).
    ///
    /// # Errors
    ///
    /// Why this device cannot run the norm at this setting, or why `out` cannot hold it.
    pub fn launch_tiled(&self, call: &RmsNormCall, out: PlainTensor) -> Result<(), Refusal> {
        let tensors =
            RmsNormTensors::tiled(&call.client, call.input.clone(), call.weight.clone(), out)?;
        self.launch_on::<Buffered, _>(call, &tensors, |blueprint, grid| {
            tensors
                .bind(blueprint, grid)
                .map_out(|out| (out, Write::Replace))
        })
    }

    /// The norm of [`launch`](Self::launch), encoded into `out`, a packed `[rows, dim]` matrix
    /// the caller allocated under one level of block scales along each row: each block's scale
    /// calibrated from the rows rounded to their
    /// element, so the result is the norm's output quantized ([`RmsNormTensors::packed`]).
    ///
    /// # Errors
    ///
    /// Why this device cannot encode the norm at this setting, or why `out` cannot hold it.
    pub fn launch_packed(&self, call: &RmsNormCall, out: PackedTensor) -> Result<(), Refusal> {
        let tensors =
            RmsNormTensors::packed(&call.client, call.input.clone(), call.weight.clone(), out)?;
        self.launch_on::<Encoded, _>(call, &tensors, |blueprint, grid| {
            tensors.bind(blueprint, grid)
        })
    }

    /// The norm of [`launch`](Self::launch), written into `out` as a product reads its lhs:
    /// [`launch_packed`](Self::launch_packed) or [`launch_tiled`](Self::launch_tiled).
    ///
    /// # Errors
    ///
    /// Those of the launch `out` names.
    pub fn launch_into(&self, call: &RmsNormCall, out: LhsOutput) -> Result<(), Refusal> {
        match out {
            LhsOutput::Packed(out) => self.launch_packed(call, out),
            LhsOutput::Tiled(out) => self.launch_tiled(call, out),
        }
    }

    /// The norm of `input`, scaled by `weight`, handed to a sink.
    ///
    /// # Errors
    ///
    /// [`RMS_NORM_SINK_TAKES_NO_ROWS`]; what [`RmsNormTensors::sink`] declines; why this device
    /// cannot run the norm at this setting.
    pub fn sink_launch(&self, call: &RmsNormCall) -> Result<RmsNormRoutineSinkLaunch<R, M>, Refusal>
    where
        R: Copy,
    {
        let tensors = RmsNormTensors::sink(&call.client, call.input.clone(), call.weight.clone())?;
        if tensors.problem().rows == 0 {
            return Err(RMS_NORM_SINK_TAKES_NO_ROWS);
        }
        let prepared = self.prepare(tensors.problem())?;
        Ok(RmsNormRoutineSinkLaunch::new(
            self.routine,
            tensors,
            prepared,
            call.epsilon,
        ))
    }

    /// The norm of `call` on `tensors`, bound by `bind` and written through `O`. No rows is no
    /// launch: a grid of no cubes is one no device spawns.
    fn launch_on<O: DestinationLaunch, Out>(
        &self,
        call: &RmsNormCall,
        tensors: &RmsNormTensors<Out>,
        bind: impl FnOnce(&RmsNormBlueprint<M>, &Grid) -> RmsNormBinding<O::Operand>,
    ) -> Result<(), Refusal> {
        if tensors.problem().rows == 0 {
            return Ok(());
        }
        let (blueprint, grid) = self.prepare(tensors.problem())?;
        let binding = bind(&blueprint, &grid);
        self.routine
            .dispatch::<O, _>(tensors, binding, blueprint, grid, call.epsilon);
        Ok(())
    }

    /// The launch this candidate prepared for `problem`, or prepares now and holds.
    fn prepare(
        &self,
        problem: &RmsNormRoutineProblem,
    ) -> Result<(RmsNormBlueprint<M>, Grid), Refusal> {
        self.prepared
            .prepare(&self.routine, &self.hw, problem, &self.settings)
    }

    /// The launch this candidate prepares for the problem `key` describes, or `None` where the
    /// routine declines it on this device. Not held: a key's problem places the candidate in a
    /// round, and may never be launched.
    fn prepared(&self, key: &RmsNormKey) -> Option<(RmsNormBlueprint<M>, Grid)> {
        self.routine
            .prepare(&self.hw, &RmsNormRoutineProblem::from(key), &self.settings)
            .ok()
    }

    /// The norm of `call` at this candidate's setting.
    fn run(&self, call: RmsNormCall) -> Result<RmsNormOutput, String> {
        self.launch(&call)
            .map(|handle| RmsNormOutput { handle })
            .map_err(|error| error.to_string())
    }
}

impl<R, M: PartialEq> TuneCandidate for RmsNormRoutineCandidate<R>
where
    R: RmsNormRoutine<M> + Routine<Microkernel = M, Blueprint = RmsNormBlueprint<M>>,
{
    type Strategy = R::Settings;
    type Key = RmsNormKey;
    /// The blueprint and the cube: the cube is kept out of a blueprint, and two settings preparing
    /// one blueprint can still launch different cubes.
    type Blueprint = (RmsNormBlueprint<M>, CubeDim);

    fn strategy(&self) -> &R::Settings {
        &self.settings
    }

    /// First, lowered a level for each halving the prepared grid falls short of filling the
    /// device; last where the routine declines the problem, which it then does at launch before
    /// compiling anything.
    fn priority(&self, key: &RmsNormKey) -> CandidatePriority {
        match self.prepared(key) {
            Some((_, grid)) => {
                CandidatePriority::FIRST.lowered(short_of_the_device(&self.hw, &grid.launch))
            }
            None => CandidatePriority::LAST,
        }
    }

    fn blueprint(&self, key: &RmsNormKey) -> Option<Self::Blueprint> {
        self.prepared(key)
            .map(|(blueprint, grid)| (blueprint, grid.launch.cube_dim))
    }
}

/// The norm of `call` through the autotune, its candidates grouped at priority `tile` on `device`.
pub fn launch_tuned(call: RmsNormCall, device: &Device, tile: GroupPriority) -> TensorHandle {
    // Nothing to normalize is nothing to tune: an empty input shares the key of a single row,
    // and a round measured on it would time no launch and keep a winner for the decode bucket.
    if call.input.binding.shape.contains(&0) {
        let (handle, _) = PlainTensor::empty_contiguous(
            &call.client,
            call.input.binding.shape.clone(),
            call.input.elem,
        );
        return handle;
    }
    let client = call.client.clone();
    // Keyed by the device: the set is built from what this device supports.
    let id = CubeTuneId::new(&client, device);
    let tunables = tunables(&id, &client, tile);
    TUNER.execute(&id, &client, tunables, call).handle
}

/// The norm of `call` written into `out`, a matrix of its rows as a product reads its lhs
/// ([`RmsNormRoutineCandidate::launch_into`]): by the routine the autotune elected for the rows, so
/// the output inherits the kernel the unfused round chose, or by the floor where no round has
/// settled or the elected routine cannot write it — the plane routine, whose planes no cube holds,
/// never encodes, and a setting's stage can outgrow the device. A refusal is a launch's before it
/// dispatches, so the floor writes into the same `out`.
///
/// # Errors
///
/// What the floor refuses: a scheme the store does not write, a device without plane
/// instructions, rows in batch entries, a row too long for a cube's stage, an output that is not
/// the rows in tiles.
pub fn launch_into_tuned(
    call: &RmsNormCall,
    device: &Device,
    tile: GroupPriority,
    out: LhsOutput,
) -> Result<(), Refusal> {
    if let Some(elected) = fastest_rms_norm_routine(&call.client, device, &call.input, tile)
        && elected.launch_into(call, out.clone()).is_ok()
    {
        return Ok(());
    }
    let hw = Hardware::from_client(&call.client);
    RmsNormRoutineCandidate::floor(&hw).launch_into(call, out)
}

/// The routine, at its setting, that the autotune found fastest for the norm of `input`'s rows on
/// `client`'s `device`, or `None` where no round has settled the key, or the floor named apart won
/// it: what a fused tail runs the norm on, so it inherits the kernel the unfused round chose.
pub fn fastest_rms_norm_routine(
    client: &Client,
    device: &Device,
    input: &PlainTensor,
    tile: GroupPriority,
) -> Option<RmsNormCandidate> {
    let id = CubeTuneId::new(client, device);
    let tunables = tunables(&id, client, tile);
    let key = RmsNormKey::from(input);
    TUNER.fastest_identity(&id, &tunables, &key)?.routine()
}

/// The autotune for the device `id` names, built on first touch, its candidates grouped at
/// priority `tile`. The one place it is built: a tuner caches a set per initializer *type*, so a
/// second `init` site would build a second set.
fn tunables(id: &CubeTuneId, client: &Client, tile: GroupPriority) -> RmsNormTunables {
    let hw = Hardware::from_client(client);
    TUNER.init(id, move || candidates(hw.clone(), tile))
}

/// Every routine's candidates on `hw`, identified by their routine and setting, over the cube
/// groups' lead as the floor.
fn candidates(
    hw: Hardware,
    priority: GroupPriority,
) -> TunableSet<RmsNormKey, RmsNormCall, RmsNormOutput, RmsNormIdentity> {
    let mut autotune =
        Autotune::<RmsNormKey, RmsNormCall, RmsNormOutput, RmsNormIdentity>::measured_on(
            |call: &RmsNormCall| RmsNormKey::from(&call.input),
            "as-keyed",
            RmsNormKey::representative,
        );
    let tile = autotune.group("tile", priority);
    let floor = RmsNormRoutineCandidate::floor(&hw);
    autotune
        .identified(
            &tile,
            RmsNormRoutineCandidate::list(CubeRmsNormRoutine, &hw),
            |candidate| RmsNormIdentity::Routine(candidate.clone().into()),
            |candidate, call| candidate.run(call),
        )
        .identified(
            &tile,
            RmsNormRoutineCandidate::list(PlaneRmsNormRoutine, &hw),
            |candidate| RmsNormIdentity::Routine(candidate.clone().into()),
            |candidate, call| candidate.run(call),
        )
        .identified(
            &tile,
            RmsNormRoutineCandidate::list(CubeGroupsRmsNormRoutine, &hw),
            |candidate| RmsNormIdentity::Routine(candidate.clone().into()),
            |candidate, call| candidate.run(call),
        )
        // The cube groups' lead, which runs everywhere: taking `tile` out of the autotune stops
        // the family being measured, not being served.
        .floor(RmsNormIdentity::named(
            "cube-groups-rms-norm-floor",
            move |call: RmsNormCall| floor.run(call),
        ))
        .finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::FloatKind;
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    /// On devices a dev machine cannot show: the cube groups, this kind's floor, prepare
    /// everywhere, and a plane or cube setting is tried last exactly where it claims no launch.
    #[test]
    fn every_device_keeps_the_floor_and_a_decline_claims_nothing() {
        for key in [
            RmsNormKey::new(F16, 2560, 1),
            RmsNormKey::new(F16, 128, 512 * 24),
            RmsNormKey::new(F16, 8192, 1),
        ] {
            for hw in [
                devices::rdna_range(),
                devices::cdna_wave64(),
                devices::h100(),
                devices::wgpu_no_subgroups(),
                devices::cpu_12(),
            ] {
                for floor in RmsNormRoutineCandidate::list(CubeGroupsRmsNormRoutine, &hw) {
                    assert!(floor.blueprint(&key).is_some(), "{}", floor.strategy());
                }
                for plane in RmsNormRoutineCandidate::list(PlaneRmsNormRoutine, &hw) {
                    let declined = plane.blueprint(&key).is_none();
                    let last = plane.priority(&key) == CandidatePriority::LAST;
                    assert_eq!(declined, last, "{}", plane.strategy());
                }
                for cube in RmsNormRoutineCandidate::list(CubeRmsNormRoutine, &hw) {
                    let declined = cube.blueprint(&key).is_none();
                    let last = cube.priority(&key) == CandidatePriority::LAST;
                    assert_eq!(declined, last, "{}", cube.strategy());
                }
            }
        }
    }

    /// Settings that snap to one launch are one blueprint, so the round measures it once: a
    /// 512-wide row leaves the cube's plane counts fewer distinct counts than the list holds, and a
    /// 65536-wide row grows every group to the same units, leaving only the groups a cube apart.
    #[test]
    fn settings_snapping_to_one_launch_are_one_blueprint() {
        fn distinct<B: PartialEq>(blueprints: impl Iterator<Item = Option<B>>) -> (usize, usize) {
            let mut seen: Vec<B> = Vec::new();
            let mut listed = 0;
            for blueprint in blueprints {
                listed += 1;
                let blueprint = blueprint.expect("the routine prepares the row");
                if !seen.contains(&blueprint) {
                    seen.push(blueprint);
                }
            }
            (seen.len(), listed)
        }
        let hw = devices::h100();

        let key = RmsNormKey::new(F16, 512, 512);
        let (cubes, listed) = distinct(
            RmsNormRoutineCandidate::list(CubeRmsNormRoutine, &hw)
                .iter()
                .map(|candidate| candidate.blueprint(&key)),
        );
        assert!(
            cubes < listed,
            "{cubes} blueprints of {listed} cube settings"
        );

        let key = RmsNormKey::new(F16, 65536, 1);
        let (groups, listed) = distinct(
            RmsNormRoutineCandidate::list(CubeGroupsRmsNormRoutine, &hw)
                .iter()
                .map(|candidate| candidate.blueprint(&key)),
        );
        assert!(
            groups < listed,
            "{groups} blueprints of {listed} cube-groups settings"
        );
    }

    /// A decode step's single row by cubes of one row falls short of the device, and a prompt's
    /// rows fill it.
    #[test]
    fn the_priority_reads_how_far_the_grid_falls_short() {
        let hw = devices::strix_halo();
        let lead = RmsNormRoutineCandidate::list(PlaneRmsNormRoutine, &hw).swap_remove(0);
        assert_eq!(
            lead.priority(&RmsNormKey::new(F16, 128, 512 * 24)),
            CandidatePriority::FIRST
        );
        assert!(lead.priority(&RmsNormKey::new(F16, 128, 1)) < CandidatePriority::FIRST);
    }
}
