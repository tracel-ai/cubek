use crate::kernels::{
    AttentionBinding, AttentionBlueprint, AttentionCacheBinding, AttentionRoutine,
    AttentionRoutineProblem, AttentionTensors,
};
use core::marker::PhantomData;
use core::num::NonZeroUsize;
use cubecl::tile::StageStorage;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::Vectorization;
use cubek_microkernel::attention::output::cube::CubeAttentionMicrokernelSettings;
use cubek_microkernel::attention::output::hypercube::{
    HypercubeAttentionMicrokernel, HypercubeAttentionMicrokernelSettings, HypercubeMerge,
    HypercubeMergeShape,
};
use cubek_microkernel::attention::output::plane::{
    PlaneShuffleAttentionMicrokernel, PlaneShuffleAttentionMicrokernelSettings,
};
use cubek_microkernel::attention::{AttentionOperand, AttentionPartitioning, QueryOrder};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Attention with several cubes owning every box of queries, every cube walking its own run of the
/// keys: the hypercube microkernel runs a cube of plane shuffles over a cube's run, its planes along
/// the run, each plane leaves its queries' partial in a slot of its own, and the slots are merged
/// into the output where `M` says ([`HypercubeShuffleMerge`]). The cubes cover the queries, the
/// heads and the runs, one batch and key head each.
///
/// What few queries against many keys want, a decode step's above all: a box's queries leave the
/// device idle where only one cube walks their keys, and the cubes sharing the box read no key
/// twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HypercubeShuffleAttentionRoutine<M> {
    merge: PhantomData<M>,
}

/// Where a hypercube shuffle's partials meet: what its two routines differ in, each merge owning
/// the kernels that leave and merge its partials.
pub trait HypercubeShuffleMerge: core::fmt::Debug + Copy + Eq + core::hash::Hash {
    const MERGE: HypercubeMerge;

    /// Why `hw` cannot merge `problem`'s slots, shaped `shape`, this way, beyond what the merge
    /// asks of the device ([`HypercubeMerge::admits`]).
    ///
    /// # Errors
    ///
    /// A merge the device cannot launch.
    fn validate(
        hw: &Hardware,
        problem: &AttentionRoutineProblem,
        shape: HypercubeMergeShape,
    ) -> Result<(), Refusal>;

    /// Launches the cubes over `binding`'s tiles or packed cache, and the merge of their slots,
    /// shaped `shape`, into the output.
    fn dispatch(
        tensors: &AttentionTensors,
        binding: AttentionBinding,
        blueprint: HypercubeShuffleAttentionRoutineBlueprint,
        grid: Grid,
        shape: HypercubeMergeShape,
    );
}

/// The hypercube microkernel both shuffle routines place: cubes of plane shuffles.
pub type HypercubeShuffleAttentionMicrokernel =
    HypercubeAttentionMicrokernel<PlaneShuffleAttentionMicrokernel>;

pub type HypercubeShuffleAttentionRoutineBlueprint =
    AttentionBlueprint<HypercubeShuffleAttentionMicrokernel, StageStorage>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeShuffleAttentionRoutineSettings<M> {
    pub microkernel:
        HypercubeAttentionMicrokernelSettings<PlaneShuffleAttentionMicrokernelSettings>,
    /// Cubes sharing one box, each on its own run of the keys: part of the partitioning, which
    /// distributes it whatever the keys ([`AttentionPartitioning::new`]).
    pub cubes_along_keys: NonZeroUsize,
    pub merge: PhantomData<M>,
}

impl<M> HypercubeShuffleAttentionRoutine<M> {
    pub const fn new() -> Self {
        Self { merge: PhantomData }
    }
}

impl<M> Default for HypercubeShuffleAttentionRoutine<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M> HypercubeShuffleAttentionRoutineSettings<M> {
    /// The plane shuffle's default on thirty-two cubes a box: a decode step's group of four over a
    /// cache of thousands of keys, on a device of a few dozen cores.
    pub const DEFAULT: Self = Self {
        microkernel: HypercubeAttentionMicrokernelSettings::<
            PlaneShuffleAttentionMicrokernelSettings,
        >::DEFAULT,
        cubes_along_keys: NonZeroUsize::new(32).unwrap(),
        merge: PhantomData,
    };
}

impl<M: HypercubeShuffleMerge> core::fmt::Display for HypercubeShuffleAttentionRoutineSettings<M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "hypercube-{}-shuffle-attention {} cubes_along_keys={}",
            M::MERGE,
            self.microkernel,
            self.cubes_along_keys
        )
    }
}

impl<M: HypercubeShuffleMerge> Routine for HypercubeShuffleAttentionRoutine<M> {
    type Problem = AttentionRoutineProblem;
    type Microkernel = HypercubeShuffleAttentionMicrokernel;
    type Settings = HypercubeShuffleAttentionRoutineSettings<M>;
    type Blueprint = HypercubeShuffleAttentionRoutineBlueprint;
    type Partitioning = AttentionPartitioning;

    /// The default first, then every one of eight, sixteen and thirty-two cubes on a box, by every
    /// setting its microkernel lists: queries a plane, units on a key, keys in flight, and planes
    /// along a cube's run of the keys. A product, not a star: more cubes shorten each run but leave
    /// more slots to merge, more planes put more reads in flight on each cube, and fewer units on a
    /// key put more keys in flight, so the best of any alone is not where they meet.
    ///
    /// Then what the product leaves out, off the default: two and four cubes on a box; one and
    /// eight keys in flight; four units on a key at one to eight keys in flight, and eight at one
    /// and eight, crossed because both size the step and the registers; and thirty-two units on a
    /// key. A step spans one key a team to thirty-two a plane.
    ///
    /// Nothing where the device cannot merge this way, where every one of them declines.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        if M::MERGE.admits(hw).is_err() {
            return Vec::new();
        }
        let count = |count: usize| NonZeroUsize::new(count).unwrap();
        let lead = Self::Settings::DEFAULT;
        let mut settings: Vec<Self::Settings> = [8, 16, 32]
            .into_iter()
            .flat_map(|cubes| {
                HypercubeShuffleAttentionMicrokernel::settings(hw)
                    .into_iter()
                    .map(move |microkernel| Self::Settings {
                        microkernel,
                        cubes_along_keys: count(cubes),
                        ..lead
                    })
            })
            .collect();
        // The default first: a stable sort keeps the rest in order.
        settings.sort_by_key(|settings| *settings != lead);
        let with_plane = |plane| Self::Settings {
            microkernel: HypercubeAttentionMicrokernelSettings {
                cube: CubeAttentionMicrokernelSettings {
                    plane,
                    ..lead.microkernel.cube
                },
            },
            ..lead
        };
        settings.extend([2, 4].map(|cubes| Self::Settings {
            cubes_along_keys: count(cubes),
            ..lead
        }));
        settings.extend([1, 8].map(|keys_in_flight| {
            with_plane(PlaneShuffleAttentionMicrokernelSettings {
                keys_in_flight: count(keys_in_flight),
                ..lead.microkernel.cube.plane
            })
        }));
        for (units_per_key, keys_in_flight) in [(4, 1), (4, 2), (4, 4), (4, 8), (8, 1), (8, 8)] {
            settings.push(with_plane(PlaneShuffleAttentionMicrokernelSettings {
                units_per_key: count(units_per_key),
                keys_in_flight: count(keys_in_flight),
                ..lead.microkernel.cube.plane
            }));
        }
        settings.push(with_plane(PlaneShuffleAttentionMicrokernelSettings {
            units_per_key: count(32),
            ..lead.microkernel.cube.plane
        }));
        settings
    }

    /// Nothing of its own: it only distributes the cubes, and their microkernel makes every
    /// access.
    fn vectorize(
        &self,
        microkernel: &HypercubeShuffleAttentionMicrokernel,
        partitioning: &AttentionPartitioning,
        vectorization: &mut Vectorization<AttentionOperand>,
    ) {
        microkernel.vectorize(partitioning, vectorization);
    }

    /// Nothing of its own: its cubes' microkernel walks each cube's run of the keys and counts
    /// each cube's partial.
    fn estimate(
        &self,
        problem: &AttentionRoutineProblem,
        blueprint: &HypercubeShuffleAttentionRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        blueprint.microkernel.estimate(
            &problem.work_item(),
            &blueprint.partitioning,
            grid.planes(),
            estimate,
        );
    }

    /// A cube holds its planes along the keys, and the shared memory its microkernel and its merge
    /// ask for. The cubes sharing a box are its settings' count, whatever the keys
    /// ([`AttentionPartitioning::new`]). Declines where the device cannot merge this way.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &AttentionRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        M::MERGE.admits(hw)?;
        let item = problem.work_item();
        let microkernel =
            HypercubeShuffleAttentionMicrokernel::best(hw, &item, &settings.microkernel)?;
        let partitioning = AttentionPartitioning::new(
            problem.space(),
            microkernel.cube.tiling(),
            microkernel.cube.page_steps,
            settings.cubes_along_keys,
            QueryOrder::InOrder,
        );
        let storage = microkernel.cube.stage_storage(&partitioning.stage());
        let cube = CubeResources {
            plane_width: hw.plane_dim()?,
            shared_memory: microkernel.shared_memory(hw, &item, &storage)
                + M::MERGE.shared_memory(),
        };
        let grid = Grid::new(hw, partitioning.as_tile(), cube)?;
        M::validate(
            hw,
            problem,
            microkernel.merge_shape(&item, &partitioning, M::MERGE),
        )?;
        let mut vectorization = Vectorization::new();
        self.vectorize(&microkernel, &partitioning, &mut vectorization);
        let blueprint = AttentionBlueprint::new(
            hw,
            problem,
            microkernel,
            partitioning,
            vectorization,
            storage,
        );
        Ok((blueprint, grid))
    }
}

impl<M: HypercubeShuffleMerge> AttentionRoutine<HypercubeShuffleAttentionMicrokernel, StageStorage>
    for HypercubeShuffleAttentionRoutine<M>
{
    /// The cubes' launch and the merge `M` holds. The keys and values are tiles, or a packed cache
    /// decoded as the planes read it.
    fn dispatch(
        &self,
        tensors: &AttentionTensors,
        binding: AttentionBinding,
        blueprint: HypercubeShuffleAttentionRoutineBlueprint,
        grid: Grid,
    ) {
        // The plane shuffle lines every operand alike, and the kernels bind its tiles at one width.
        if let AttentionCacheBinding::Tiles { .. } = binding.cache {
            assert!(
                blueprint.widths.key == blueprint.widths.query
                    && blueprint.widths.value == blueprint.widths.query,
                "the plane shuffle's operands share one width"
            );
        }
        let shape = blueprint.microkernel.merge_shape(
            &tensors.problem().work_item(),
            &blueprint.partitioning,
            M::MERGE,
        );
        M::dispatch(tensors, binding, blueprint, grid, shape);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{
        AttentionCacheLayout, HypercubeLastArrivalShuffleAttentionRoutine,
        HypercubeSecondLaunchShuffleAttentionRoutine,
    };
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_microkernel::attention::output::hypercube::HYPERCUBE_CUBES_UNMET;
    use cubek_microkernel::attention::{Q, S};
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// A decode step over a long cache, and a short chunk of a prompt against it.
    fn problems() -> [AttentionRoutineProblem; 2] {
        [
            AttentionRoutineProblem::dense((1, 32, 8), (1, 4096), (128, 128), F16),
            AttentionRoutineProblem::dense((1, 8, 2), (16, 1000), (64, 64), F32),
        ]
    }

    /// `hw` where one cube's stores do not reach another within a launch.
    fn unmet(hw: Hardware) -> Hardware {
        {
            let mut hw = hw;
            devices::features(&mut hw).device_memory_scope = false;
            hw
        }
    }

    /// Every setting `routine` lists on each of `devices` prepares.
    fn assert_every_setting_prepares<M: HypercubeShuffleMerge>(
        routine: HypercubeShuffleAttentionRoutine<M>,
        devices: &[Hardware],
    ) {
        for hw in devices {
            let settings = routine.settings(hw);
            assert!(!settings.is_empty());
            for settings in settings {
                for problem in problems() {
                    let prepared = routine.prepare(hw, &problem, &settings);
                    assert!(prepared.is_ok(), "{settings} on {hw:?}: {prepared:?}");
                }
            }
        }
    }

    /// Every setting of either merge prepares, a second launch on a device whose cubes cannot meet
    /// within a launch too.
    #[test]
    fn every_setting_prepares() {
        let devices = [devices::m2_pro(), devices::h100()];
        assert_every_setting_prepares(HypercubeLastArrivalShuffleAttentionRoutine::new(), &devices);
        assert_every_setting_prepares(
            HypercubeSecondLaunchShuffleAttentionRoutine::new(),
            &[devices::m2_pro(), devices::h100(), unmet(devices::m2_pro())],
        );
    }

    /// Every setting declines where the device commits to no plane width, rather than launching
    /// on lanes it would misread.
    #[test]
    fn every_setting_declines_without_a_plane_width() {
        fn declines<M: HypercubeShuffleMerge>(routine: HypercubeShuffleAttentionRoutine<M>) {
            for hw in [devices::wgpu_no_subgroups(), devices::rdna_range()] {
                for settings in routine.settings(&devices::m2_pro()) {
                    for problem in problems() {
                        let declined = routine.prepare(&hw, &problem, &settings);
                        assert!(declined.is_err(), "{settings} on {hw:?}: {declined:?}");
                    }
                }
            }
        }
        declines(HypercubeLastArrivalShuffleAttentionRoutine::new());
        declines(HypercubeSecondLaunchShuffleAttentionRoutine::new());
    }

    /// The last arrival's last cube reads the others' slots within the launch, so where one cube's
    /// stores do not reach another it lists nothing, and its default declines saying why.
    #[test]
    fn the_last_arrival_declines_where_cubes_cannot_meet() {
        let hw = unmet(devices::m2_pro());
        let routine = HypercubeLastArrivalShuffleAttentionRoutine::new();
        assert!(routine.settings(&hw).is_empty());
        let [problem, _] = problems();
        let declined = routine.prepare(
            &hw,
            &problem,
            &HypercubeShuffleAttentionRoutineSettings::DEFAULT,
        );
        assert_eq!(declined.err(), Some(HYPERCUBE_CUBES_UNMET));
    }

    /// The cubes sharing a box each walk a run of the keys: thirty-two on a decode step's box of
    /// 4096 keys walk 128 each, a step being two teams' two keys, and the grid holds thirty-two
    /// cubes for every head of the group, one plane each.
    #[test]
    fn the_cubes_share_a_box_along_the_keys() {
        let [problem, _] = problems();
        let (blueprint, grid) = HypercubeLastArrivalShuffleAttentionRoutine::new()
            .prepare(
                &devices::m2_pro(),
                &problem,
                &HypercubeShuffleAttentionRoutineSettings::DEFAULT,
            )
            .expect("the default prepares on the M2");
        let partitioning = &blueprint.partitioning;
        assert_eq!(partitioning.cubes_along_keys().get(), 32);
        assert_eq!(partitioning.stage().extent(S), 4);
        assert_eq!(partitioning.cube_box().extent(S), 128);
        assert_eq!(partitioning.cube_box().extent(Q), 4);
        // One head a box, the group's four by eight key heads.
        assert_eq!(partitioning.cubes(), 4 * 32);
        assert_eq!(grid.cubes(), Some(8 * 4 * 32));
    }

    /// Over pages of 64 positions the cubes take pages in turns rather than steps: thirty-two
    /// cubes on 4096 keys take two pages each, 128 keys, still staging one step of four keys.
    #[test]
    fn the_cubes_share_a_box_a_page_at_a_time() {
        let [decode, _] = problems();
        let paged = AttentionRoutineProblem {
            cache: AttentionCacheLayout::Pages { page_size: 64 },
            ..decode
        };
        let (blueprint, _) = HypercubeLastArrivalShuffleAttentionRoutine::new()
            .prepare(
                &devices::m2_pro(),
                &paged,
                &HypercubeShuffleAttentionRoutineSettings::DEFAULT,
            )
            .expect("the default prepares on the M2");
        let partitioning = &blueprint.partitioning;
        assert_eq!(
            blueprint
                .microkernel
                .cube
                .page_steps
                .map(|steps| steps.get()),
            Some(16)
        );
        assert_eq!(partitioning.cubes_along_keys().get(), 32);
        assert_eq!(partitioning.stage().extent(S), 4);
        assert_eq!(partitioning.cube_box().extent(S), 128);
    }
}
