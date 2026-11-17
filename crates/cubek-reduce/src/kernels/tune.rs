//! The reduce family's autotune: every routine at every setting the device lists, raced per
//! problem, over the unit routine as the floor.

use crate::kernels::{
    CubeReduceRoutine, HypercubeReduceRoutine, PlaneReduceRoutine, ReduceElements, ReduceKernel,
    ReduceRoutine, ReduceRoutineBlueprint, ReduceRoutineProblem, ReduceTensors, UnitReduceRoutine,
};
use cubecl::ir::ElemType;
use cubecl::throughput::MemoryAccess;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tune::{LocalTuner, TunableSet, local_tuner};
use cubecl::{AutotuneKey, CubeDim, CubeTuneId, Device};
use cubek_microkernel::reduce::{ReduceLayout, ReduceOperation};
use cubek_std::Hardware;
use cubek_std::autotune::{
    Autotune, CandidatePriority, Ceiling, GroupPriority, Roofline, TuneCandidate,
    short_of_the_device,
};
use cubek_std::routine::{EstimatedProblem, Grid, Routine};
use serde::{Deserialize, Serialize};

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

impl From<&ReduceTensors> for ReduceKey {
    fn from(tensors: &ReduceTensors) -> Self {
        let problem = tensors.problem();
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

static TUNER: LocalTuner<ReduceKey, CubeTuneId> = local_tuner!("reduce");

/// `tensors` reduced by the routine the autotune finds fastest for them on `device`.
pub fn launch_tuned(tensors: ReduceTensors, device: &Device) {
    let client = tensors.client().clone();
    let id = CubeTuneId::new(&client, device);
    let hw = Hardware::from_client(&client);
    let tunables = TUNER.init(&id, move || candidates(hw.clone()));
    TUNER.execute(&id, &client, tunables, tensors);
}

/// Every routine's candidates on `hw`, over the unit routine as the floor, which serves every
/// problem on every device.
fn candidates(hw: Hardware) -> TunableSet<ReduceKey, ReduceTensors, ()> {
    let mut autotune = Autotune::<ReduceKey, ReduceTensors, ()>::new(|tensors: &ReduceTensors| {
        ReduceKey::from(tensors)
    });
    let tile = autotune.group("tile", GroupPriority::DEFAULT);
    let floor_hw = hw.clone();
    autotune
        .prioritized(
            &tile,
            ReduceRoutineCandidate::list(UnitReduceRoutine, ReduceKernel::Unit, &hw),
            |candidate, tensors| candidate.run(tensors),
        )
        .prioritized(
            &tile,
            ReduceRoutineCandidate::list(PlaneReduceRoutine, ReduceKernel::PlaneShuffle, &hw),
            |candidate, tensors| candidate.run(tensors),
        )
        .prioritized(
            &tile,
            ReduceRoutineCandidate::list(CubeReduceRoutine, ReduceKernel::CubeStaged, &hw),
            |candidate, tensors| candidate.run(tensors),
        )
        .prioritized(
            &tile,
            ReduceRoutineCandidate::list(
                HypercubeReduceRoutine,
                ReduceKernel::HypercubeLastArrival,
                &hw,
            ),
            |candidate, tensors| candidate.run(tensors),
        )
        .bounded(|_, tensors: &ReduceTensors| Roofline {
            client: tensors.client().clone(),
            ceiling: Ceiling::Memory(MemoryAccess::Read),
            work: tensors.problem().demand(),
        })
        .floor(cubecl::tune::Tunable::new(
            "unit-reduce-floor",
            move |tensors: ReduceTensors| {
                ReduceKernel::Unit
                    .launch(&floor_hw, &tensors)
                    .map_err(|refusal| refusal.to_string())
            },
        ))
        .finish()
}
