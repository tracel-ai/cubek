//! The reduce family's autotune: every routine at every setting the device lists, raced per
//! problem, over the unit routine as the floor.

use crate::kernels::{
    CubeReduceRoutine, HypercubeReduceRoutine, PlaneReduceRoutine, ReduceElements, ReduceKernel,
    ReduceRoutine, ReduceRoutineBlueprint, ReduceRoutineProblem, ReduceTensors, UnitReduceRoutine,
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
            problem.rows().len(),
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
pub struct ReduceRoutineCandidate<R: Routine> {
    routine: R,
    settings: R::Settings,
    /// The kernel and its setting: what the tuning cache names the candidate by.
    name: String,
    hw: Hardware,
}

impl<R: Routine + Copy> ReduceRoutineCandidate<R>
where
    R::Settings: core::fmt::Display,
{
    /// Every setting `routine`, the kernel `kernel` names, lists on `hw`, as candidates.
    pub fn list(routine: R, kernel: ReduceKernel, hw: &Hardware) -> Vec<Self> {
        routine
            .settings(hw)
            .into_iter()
            .map(|settings| Self {
                routine,
                name: format!("{} {settings}", kernel.name()),
                settings,
                hw: hw.clone(),
            })
            .collect()
    }
}

impl<R, M> ReduceRoutineCandidate<R>
where
    R: ReduceRoutine<M> + Routine<Microkernel = M>,
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
    /// ([`ReduceRoutine::launch_unbound`]).
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
    fn prepared(&self, key: &ReduceKey) -> Option<(ReduceRoutineBlueprint<M>, Grid)> {
        self.routine
            .prepare(&self.hw, &ReduceRoutineProblem::from(key), &self.settings)
            .ok()
    }
}

impl<R, M: PartialEq> TuneCandidate for ReduceRoutineCandidate<R>
where
    R: ReduceRoutine<M> + Routine<Microkernel = M>,
{
    type Strategy = String;
    type Key = ReduceKey;
    /// The blueprint and the cube: two settings preparing one blueprint can still launch
    /// different cubes.
    type Blueprint = (ReduceRoutineBlueprint<M>, CubeDim);

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
/// ([`fastest_reduce_candidate`]), so a launch outside a round, a fused one, runs the winner.
#[derive(Debug, Clone)]
pub enum ReduceCandidate {
    /// [`ReduceKernel::Unit`].
    Unit(ReduceRoutineCandidate<UnitReduceRoutine>),
    /// [`ReduceKernel::PlaneShuffle`].
    PlaneShuffle(ReduceRoutineCandidate<PlaneReduceRoutine>),
    /// [`ReduceKernel::CubeStaged`].
    CubeStaged(ReduceRoutineCandidate<CubeReduceRoutine>),
    /// [`ReduceKernel::HypercubeLastArrival`].
    HypercubeLastArrival(ReduceRoutineCandidate<HypercubeReduceRoutine>),
}

impl ReduceCandidate {
    /// `tensors`, whose input the caller binds, reduced at this candidate's setting
    /// ([`ReduceRoutineCandidate::launch_unbound`]).
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
        match self {
            Self::Unit(candidate) => candidate.launch_unbound::<I>(tensors, input),
            Self::PlaneShuffle(candidate) => candidate.launch_unbound::<I>(tensors, input),
            Self::CubeStaged(candidate) => candidate.launch_unbound::<I>(tensors, input),
            Self::HypercubeLastArrival(candidate) => candidate.launch_unbound::<I>(tensors, input),
        }
    }
}

/// The candidate's name: its kernel and setting, which the tuning cache is keyed by.
impl core::fmt::Display for ReduceCandidate {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match self {
            Self::Unit(candidate) => candidate.strategy(),
            Self::PlaneShuffle(candidate) => candidate.strategy(),
            Self::CubeStaged(candidate) => candidate.strategy(),
            Self::HypercubeLastArrival(candidate) => candidate.strategy(),
        };
        f.write_str(name)
    }
}

impl From<ReduceRoutineCandidate<UnitReduceRoutine>> for ReduceCandidate {
    fn from(candidate: ReduceRoutineCandidate<UnitReduceRoutine>) -> Self {
        Self::Unit(candidate)
    }
}

impl From<ReduceRoutineCandidate<PlaneReduceRoutine>> for ReduceCandidate {
    fn from(candidate: ReduceRoutineCandidate<PlaneReduceRoutine>) -> Self {
        Self::PlaneShuffle(candidate)
    }
}

impl From<ReduceRoutineCandidate<CubeReduceRoutine>> for ReduceCandidate {
    fn from(candidate: ReduceRoutineCandidate<CubeReduceRoutine>) -> Self {
        Self::CubeStaged(candidate)
    }
}

impl From<ReduceRoutineCandidate<HypercubeReduceRoutine>> for ReduceCandidate {
    fn from(candidate: ReduceRoutineCandidate<HypercubeReduceRoutine>) -> Self {
        Self::HypercubeLastArrival(candidate)
    }
}

/// What a candidate of the autotune is identified by: the candidate itself, or the floor by name.
type ReduceIdentity = CandidateIdentity<ReduceCandidate>;

/// The autotune's candidates, as the tuner holds them.
type ReduceTunables = Arc<TunableSet<ReduceKey, ReduceTensors, (), ReduceIdentity>>;

static TUNER: LocalTuner<ReduceKey, CubeTuneId> = local_tuner!("reduce");

/// `tensors` reduced by the routine the autotune finds fastest for them on `device`.
pub fn launch_tuned(tensors: ReduceTensors, device: &Device) {
    let client = tensors.client().clone();
    let id = CubeTuneId::new(&client, device);
    let tunables = tunables(&id, &client);
    TUNER.execute(&id, &client, tunables, tensors);
}

/// The candidate, a routine at its setting, that the autotune found fastest for the reductions
/// `key` describes on `client`'s `device`, or `None` where no round in this process has settled
/// the key, or the floor named apart won it: what a fused reduce runs, so it inherits the kernel
/// the unfused round chose.
pub fn fastest_reduce_candidate(
    client: &Client,
    device: &Device,
    key: &ReduceKey,
) -> Option<ReduceCandidate> {
    let id = CubeTuneId::new(client, device);
    let tunables = tunables(&id, client);
    TUNER.fastest_identity(&id, &tunables, key)?.routine()
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
    autotune
        .identified(
            &tile,
            ReduceRoutineCandidate::list(UnitReduceRoutine, ReduceKernel::Unit, &hw),
            |candidate| ReduceIdentity::Routine(candidate.clone().into()),
            |candidate, tensors| candidate.run(tensors),
        )
        .identified(
            &tile,
            ReduceRoutineCandidate::list(PlaneReduceRoutine, ReduceKernel::PlaneShuffle, &hw),
            |candidate| ReduceIdentity::Routine(candidate.clone().into()),
            |candidate, tensors| candidate.run(tensors),
        )
        .identified(
            &tile,
            ReduceRoutineCandidate::list(CubeReduceRoutine, ReduceKernel::CubeStaged, &hw),
            |candidate| ReduceIdentity::Routine(candidate.clone().into()),
            |candidate, tensors| candidate.run(tensors),
        )
        .identified(
            &tile,
            ReduceRoutineCandidate::list(
                HypercubeReduceRoutine,
                ReduceKernel::HypercubeLastArrival,
                &hw,
            ),
            |candidate| ReduceIdentity::Routine(candidate.clone().into()),
            |candidate, tensors| candidate.run(tensors),
        )
        .bounded(|_, tensors: &ReduceTensors| Roofline {
            client: tensors.client().clone(),
            ceiling: Ceiling::Memory(MemoryAccess::Read),
            work: tensors.problem().demand(),
        })
        .floor(ReduceIdentity::named(
            "unit-reduce-floor",
            move |tensors: ReduceTensors| {
                ReduceKernel::Unit
                    .launch(&floor_hw, &tensors)
                    .map_err(|refusal| refusal.to_string())
            },
        ))
        .finish()
}
