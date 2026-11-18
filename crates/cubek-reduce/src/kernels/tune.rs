//! The reduce family's autotune: every routine at every setting the device lists, raced per
//! problem, over the unit routine as the floor.

use crate::kernels::{
    CubeStagedReduceRoutine, HypercubeAtomicReduceRoutine, HypercubeLastArrivalReduceRoutine,
    PlaneShuffleReduceRoutine, ReduceBlueprint, ReduceElements, ReduceKernel, ReduceLauncher,
    ReduceRoutineProblem, ReduceTensors, UnitDirectReduceRoutine, UnitDirectZeroRoutine,
};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::prelude::LaunchArg;
use cubecl::throughput::MemoryAccess;
use cubecl::tile::Geometry;
use cubecl::tile::launch::{Refusal, TileOperand, Unbound};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tune::{LocalTuner, TunableSet, local_tuner};
use cubecl::{AutotuneKey, CubeDim, CubeTuneId, Device};
use cubek_microkernel::reduce::{ReduceLayout, ReduceOperation};
use cubek_std::Hardware;
use cubek_std::autotune::{
    Autotune, CandidateIdentity, CandidatePriority, Ceiling, GroupPriority, Identity, Roofline,
    TuneCandidate, short_of_the_device,
};
use cubek_std::routine::{EstimatedProblem, Grid, Routine};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// What tells one reduction from another for the autotune: its elements, its operation, how its
/// lines lie, how many dims its rows run along, how long a row is and how many rows there are.
///
/// The row dims are kept whole, since the cube and the hypercube decline more than they place
/// their cubes along: a winner for a matrix would otherwise be launched on a tensor of more dims.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct ReduceKey {
    elements: ReduceElements,
    indices: Option<ElemType>,
    operation: ReduceOperation,
    layout: ReduceLayout,
    row_dims: usize,
    #[autotune(anchor(exp(min = 16, max = 1048576, base = 4)))]
    reduced: usize,
    #[autotune(anchor(exp(max = 1048576, base = 8)))]
    rows: usize,
}

impl<In> From<&ReduceTensors<In>> for ReduceKey {
    fn from(tensors: &ReduceTensors<In>) -> Self {
        Self::from(tensors.problem())
    }
}

impl From<&ReduceRoutineProblem> for ReduceKey {
    fn from(problem: &ReduceRoutineProblem) -> Self {
        Self::new(
            problem.elements,
            problem.indices,
            problem.operation,
            problem.layout(),
            problem.shape.len().saturating_sub(1),
            problem.reduced_extent(),
            problem.row_count(),
        )
    }
}

impl From<&ReduceKey> for ReduceRoutineProblem {
    /// A dense tensor of `rows` rows of `reduced` values, its rows along the innermost of its row
    /// dims, the others one wide: the reduced dim the innermost where the lines lie along it, and
    /// next to it where they lie across it.
    fn from(key: &ReduceKey) -> Self {
        let mut shape = vec![1; key.row_dims.saturating_sub(1)];
        let axis = match (key.layout, key.row_dims) {
            (_, 0) => {
                shape.push(key.reduced);
                0
            }
            (ReduceLayout::Along, _) => {
                shape.extend([key.rows, key.reduced]);
                shape.len() - 1
            }
            (ReduceLayout::Across, _) => {
                shape.extend([key.reduced, key.rows]);
                shape.len() - 2
            }
        };
        let mut out_shape = shape.clone();
        out_shape[axis] = key.operation.written_per_row();
        Self {
            elem_alignments: [&shape, &out_shape]
                .map(|shape| ElemAlignment::dense_rows(shape[shape.len() - 1])),
            contiguous: Some(shape.len() - 1),
            shape,
            axis,
            operation: key.operation,
            elements: key.elements,
            indices: key.indices,
        }
    }
}

/// One candidate of the autotune: a routine at one of its settings, on the device the autotune is
/// built for.
#[derive(Debug, Clone)]
pub(crate) struct ReduceRoutineCandidate<R: Routine> {
    routine: R,
    settings: R::Settings,
    /// The routine's prefix and its setting: what the tuning cache names the candidate by.
    name: String,
    hw: Hardware,
}

impl<R, M> ReduceRoutineCandidate<R>
where
    R: ReduceLauncher<M> + Routine<Microkernel = M> + Copy,
    R::Settings: core::fmt::Display,
{
    /// Every setting `routine` lists on `hw`, as candidates, each named by the routine's prefix
    /// and its setting.
    pub fn list(routine: R, hw: &Hardware) -> Vec<Self> {
        routine
            .settings(hw)
            .into_iter()
            .map(|settings| Self {
                routine,
                name: format!("{} {settings}", R::NAME),
                settings,
                hw: hw.clone(),
            })
            .collect()
    }
}

impl<R, M> ReduceRoutineCandidate<R>
where
    R: ReduceLauncher<M> + Routine<Microkernel = M>,
{
    /// `tensors` reduced at this candidate's setting.
    fn run(&self, tensors: ReduceTensors) -> Result<(), String> {
        let (blueprint, grid) = self
            .routine
            .prepare(&self.hw, tensors.problem(), &self.settings)
            .map_err(|refusal| refusal.to_string())?;
        self.routine.launch(&tensors, blueprint, grid);
        Ok(())
    }

    /// `tensors`, whose input the caller binds, reduced at this candidate's setting
    /// ([`ReduceLauncher::launch_unbound`]).
    ///
    /// # Errors
    ///
    /// Why the routine cannot run the problem at its setting, or why it cannot read the caller's
    /// input.
    pub fn launch_unbound<I: TileOperand>(
        &self,
        tensors: &ReduceTensors<Geometry>,
        input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
    ) -> Result<(), Refusal> {
        let (blueprint, grid) =
            self.routine
                .prepare(&self.hw, tensors.problem(), &self.settings)?;
        self.routine
            .launch_unbound::<I>(tensors, blueprint, grid, input)
    }

    /// The launch this candidate prepares for the problem `key` describes, or `None` where the
    /// routine declines it on this device.
    fn prepared(&self, key: &ReduceKey) -> Option<(ReduceBlueprint<M>, Grid)> {
        self.routine
            .prepare(&self.hw, &ReduceRoutineProblem::from(key), &self.settings)
            .ok()
    }
}

impl ReduceRoutineCandidate<HypercubeAtomicReduceRoutine> {
    /// [`launch_unbound`](Self::launch_unbound) after the output is zeroed, since the routine only
    /// adds into it.
    ///
    /// # Errors
    ///
    /// Why the routine cannot run the problem at its setting, why the zeroing cannot run it, or
    /// why it cannot read the caller's input.
    fn launch_zeroed_unbound<I: TileOperand>(
        &self,
        tensors: &ReduceTensors<Geometry>,
        input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
    ) -> Result<(), Refusal> {
        let (blueprint, grid) =
            self.routine
                .prepare(&self.hw, tensors.problem(), &self.settings)?;
        let (zero_blueprint, zero_grid) =
            UnitDirectZeroRoutine.prepare_first_served(&self.hw, tensors.problem())?;
        UnitDirectZeroRoutine.zero_unbound(tensors, zero_blueprint, zero_grid);
        self.routine
            .launch_unbound::<I>(tensors, blueprint, grid, input)
    }
}

impl<R, M: PartialEq> TuneCandidate for ReduceRoutineCandidate<R>
where
    R: ReduceLauncher<M> + Routine<Microkernel = M>,
{
    type Strategy = String;
    type Key = ReduceKey;
    /// The blueprint and the cube: two settings preparing one blueprint can still launch
    /// different cubes.
    type Blueprint = (ReduceBlueprint<M>, CubeDim);

    fn strategy(&self) -> &String {
        &self.name
    }

    /// First, lowered a level for each halving the prepared grid falls short of filling the
    /// device; last where the routine declines the problem.
    fn priority(&self, key: &ReduceKey) -> CandidatePriority {
        match self.prepared(key) {
            Some((_, grid)) => {
                CandidatePriority::FIRST.lowered(short_of_the_device(&self.hw, &grid.launch))
            }
            None => CandidatePriority::LAST,
        }
    }

    fn blueprint(&self, key: &ReduceKey) -> Option<Self::Blueprint> {
        self.prepared(key)
            .map(|(blueprint, grid)| (blueprint, grid.launch.cube_dim))
    }
}

/// A candidate of the autotune, whichever routine it launches: what the autotune identifies a
/// candidate by, and what the tuner hands back for the one it found fastest
/// ([`ReduceCandidate::fastest`]), so a launch outside a round, a fused one, runs the winner.
#[derive(Debug, Clone)]
pub struct ReduceCandidate {
    launcher: ReduceCandidateLauncher,
}

impl ReduceCandidate {
    /// `tensors`, whose input the caller binds, reduced at this candidate's setting.
    ///
    /// # Errors
    ///
    /// Why the routine cannot run the problem at its setting, or why it cannot read the caller's
    /// input.
    pub fn launch_unbound<I: TileOperand>(
        &self,
        tensors: &ReduceTensors<Geometry>,
        input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
    ) -> Result<(), Refusal> {
        self.launcher.launch_unbound::<I>(tensors, input)
    }

    /// The candidate, a routine at its setting, that the autotune found fastest for the
    /// reductions `key` describes on `client`'s `device`, or `None` where no round in this process
    /// has settled the key, or the floor named apart won it: what a fused reduce runs, so it
    /// inherits the kernel the unfused round chose.
    pub fn fastest(client: &Client, device: &Device, key: &ReduceKey) -> Option<Self> {
        let id = CubeTuneId::new(client, device);
        let tunables = tunables(&id, client);
        TUNER.fastest_identity(&id, &tunables, key)?.routine()
    }

    /// The candidate a routine at one setting is, launched through `dispatch`.
    fn launching(dispatch: ReduceCandidateLauncherDispatch) -> Self {
        Self {
            launcher: ReduceCandidateLauncher { dispatch },
        }
    }
}

/// The candidate's name: its routine's prefix and setting, which the tuning cache is keyed by.
impl core::fmt::Display for ReduceCandidate {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.launcher.name())
    }
}

/// Launches a [`ReduceCandidate`] through whichever routine it holds.
#[derive(Debug, Clone)]
struct ReduceCandidateLauncher {
    dispatch: ReduceCandidateLauncherDispatch,
}

/// The routine a [`ReduceCandidateLauncher`] launches through, one variant a routine.
#[derive(Debug, Clone)]
enum ReduceCandidateLauncherDispatch {
    UnitDirect(ReduceRoutineCandidate<UnitDirectReduceRoutine>),
    PlaneShuffle(ReduceRoutineCandidate<PlaneShuffleReduceRoutine>),
    CubeStaged(ReduceRoutineCandidate<CubeStagedReduceRoutine>),
    HypercubeLastArrival(ReduceRoutineCandidate<HypercubeLastArrivalReduceRoutine>),
    /// Run after the output is zeroed.
    HypercubeAtomic(ReduceRoutineCandidate<HypercubeAtomicReduceRoutine>),
}

impl ReduceCandidateLauncher {
    /// `tensors`, whose input the caller binds, reduced at the candidate's setting
    /// ([`ReduceRoutineCandidate::launch_unbound`]).
    fn launch_unbound<I: TileOperand>(
        &self,
        tensors: &ReduceTensors<Geometry>,
        input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
    ) -> Result<(), Refusal> {
        match &self.dispatch {
            ReduceCandidateLauncherDispatch::UnitDirect(candidate) => {
                candidate.launch_unbound::<I>(tensors, input)
            }
            ReduceCandidateLauncherDispatch::PlaneShuffle(candidate) => {
                candidate.launch_unbound::<I>(tensors, input)
            }
            ReduceCandidateLauncherDispatch::CubeStaged(candidate) => {
                candidate.launch_unbound::<I>(tensors, input)
            }
            ReduceCandidateLauncherDispatch::HypercubeLastArrival(candidate) => {
                candidate.launch_unbound::<I>(tensors, input)
            }
            ReduceCandidateLauncherDispatch::HypercubeAtomic(candidate) => {
                candidate.launch_zeroed_unbound::<I>(tensors, input)
            }
        }
    }

    /// The candidate's name, which the tuning cache is keyed by.
    fn name(&self) -> &str {
        match &self.dispatch {
            ReduceCandidateLauncherDispatch::UnitDirect(candidate) => candidate.strategy(),
            ReduceCandidateLauncherDispatch::PlaneShuffle(candidate) => candidate.strategy(),
            ReduceCandidateLauncherDispatch::CubeStaged(candidate) => candidate.strategy(),
            ReduceCandidateLauncherDispatch::HypercubeLastArrival(candidate) => {
                candidate.strategy()
            }
            ReduceCandidateLauncherDispatch::HypercubeAtomic(candidate) => candidate.strategy(),
        }
    }
}

impl From<ReduceRoutineCandidate<UnitDirectReduceRoutine>> for ReduceCandidateLauncherDispatch {
    fn from(candidate: ReduceRoutineCandidate<UnitDirectReduceRoutine>) -> Self {
        Self::UnitDirect(candidate)
    }
}

impl From<ReduceRoutineCandidate<PlaneShuffleReduceRoutine>> for ReduceCandidateLauncherDispatch {
    fn from(candidate: ReduceRoutineCandidate<PlaneShuffleReduceRoutine>) -> Self {
        Self::PlaneShuffle(candidate)
    }
}

impl From<ReduceRoutineCandidate<CubeStagedReduceRoutine>> for ReduceCandidateLauncherDispatch {
    fn from(candidate: ReduceRoutineCandidate<CubeStagedReduceRoutine>) -> Self {
        Self::CubeStaged(candidate)
    }
}

impl From<ReduceRoutineCandidate<HypercubeLastArrivalReduceRoutine>>
    for ReduceCandidateLauncherDispatch
{
    fn from(candidate: ReduceRoutineCandidate<HypercubeLastArrivalReduceRoutine>) -> Self {
        Self::HypercubeLastArrival(candidate)
    }
}

impl From<ReduceRoutineCandidate<HypercubeAtomicReduceRoutine>>
    for ReduceCandidateLauncherDispatch
{
    fn from(candidate: ReduceRoutineCandidate<HypercubeAtomicReduceRoutine>) -> Self {
        Self::HypercubeAtomic(candidate)
    }
}

/// What a candidate of the autotune is identified by: the candidate itself, or the floor by name.
type ReduceIdentity = CandidateIdentity<ReduceCandidate>;

/// The autotune's candidates, as the tuner holds them.
type ReduceTunables = Arc<TunableSet<ReduceKey, ReduceTensors, (), ReduceIdentity>>;

static TUNER: LocalTuner<ReduceKey, CubeTuneId> = local_tuner!("reduce");

/// `tensors` reduced by the routine the autotune finds fastest for them on `device`.
pub(crate) fn launch_tuned(tensors: ReduceTensors, device: &Device) {
    let client = tensors.client().clone();
    let id = CubeTuneId::new(&client, device);
    let tunables = tunables(&id, &client);
    TUNER.execute(&id, &client, tunables, tensors);
}

/// The autotune for the device `id` names, built on first touch. The one place it is built: a
/// tuner caches a set per initializer *type*, so a second `init` site would build a second set.
fn tunables(id: &CubeTuneId, client: &Client) -> ReduceTunables {
    let hw = Hardware::from_client(client);
    TUNER.init(id, move || candidates(hw.clone()))
}

/// Every routine's candidates on `hw`, identified by their routine and setting, over the unit
/// routine as the floor, which serves every problem on every device.
fn candidates(hw: Hardware) -> TunableSet<ReduceKey, ReduceTensors, (), ReduceIdentity> {
    let mut autotune =
        Autotune::<ReduceKey, ReduceTensors, (), ReduceIdentity>::new(|tensors: &ReduceTensors| {
            ReduceKey::from(tensors)
        });
    let tile = autotune.group("tile", GroupPriority::DEFAULT);
    let floor_hw = hw.clone();
    let zero_hw = hw.clone();
    autotune
        .identified(
            &tile,
            ReduceRoutineCandidate::list(UnitDirectReduceRoutine, &hw),
            |candidate| {
                ReduceIdentity::Routine(ReduceCandidate::launching(candidate.clone().into()))
            },
            |candidate, tensors| candidate.run(tensors),
        )
        .identified(
            &tile,
            ReduceRoutineCandidate::list(PlaneShuffleReduceRoutine, &hw),
            |candidate| {
                ReduceIdentity::Routine(ReduceCandidate::launching(candidate.clone().into()))
            },
            |candidate, tensors| candidate.run(tensors),
        )
        .identified(
            &tile,
            ReduceRoutineCandidate::list(CubeStagedReduceRoutine, &hw),
            |candidate| {
                ReduceIdentity::Routine(ReduceCandidate::launching(candidate.clone().into()))
            },
            |candidate, tensors| candidate.run(tensors),
        )
        .identified(
            &tile,
            ReduceRoutineCandidate::list(HypercubeLastArrivalReduceRoutine, &hw),
            |candidate| {
                ReduceIdentity::Routine(ReduceCandidate::launching(candidate.clone().into()))
            },
            |candidate, tensors| candidate.run(tensors),
        )
        .identified(
            &tile,
            ReduceRoutineCandidate::list(HypercubeAtomicReduceRoutine, &hw),
            |candidate| {
                ReduceIdentity::Routine(ReduceCandidate::launching(candidate.clone().into()))
            },
            move |candidate, tensors| {
                // It only adds into the output, which the zeroing, timed with it, sets first.
                let (blueprint, grid) = UnitDirectZeroRoutine
                    .prepare_first_served(&zero_hw, tensors.problem())
                    .map_err(|refusal| refusal.to_string())?;
                UnitDirectZeroRoutine.launch(&tensors, blueprint, grid);
                candidate.run(tensors)
            },
        )
        .bounded(|_, tensors: &ReduceTensors| Roofline {
            client: tensors.client().clone(),
            ceiling: Ceiling::Memory(MemoryAccess::Read),
            work: tensors.problem().demand(),
        })
        .floor(ReduceIdentity::named(
            "unit_direct_reduce floor",
            move |tensors: ReduceTensors| {
                ReduceKernel::Unit
                    .launch(&floor_hw, &tensors)
                    .map_err(|refusal| refusal.to_string())
            },
        ))
        .finish()
}
