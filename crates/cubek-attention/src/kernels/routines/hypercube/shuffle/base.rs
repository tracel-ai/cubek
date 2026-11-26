use crate::kernels::{
    AttentionBinding, AttentionBlueprint, AttentionCacheBinding, AttentionCacheLayout,
    AttentionPartitioning, AttentionRoutineProblem, AttentionSplitSlots, AttentionTensors,
    HypercubeMerge, PACKED_QUERIES_UNREAD, PLANE_SHUFFLE_ONE_WIDTH, PLANE_SHUFFLE_WIDTHS_DIFFER,
    QueryOrder,
};
use core::marker::PhantomData;
use core::num::NonZeroUsize;
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::prelude::{CubeCount, CubeDim, TensorArg};
use cubecl::tile::launch::{Decline, Vectorization};
use cubecl::tile::{Launcher, StageStorage, TileTensorLaunch};
use cubek_microkernel::attention::cube::{
    CubeAttentionMicrokernelSettings, CubeAttentionPlaneSettings,
};
use cubek_microkernel::attention::hypercube::{
    HypercubeAttentionMicrokernel, HypercubeAttentionMicrokernelSettings,
};
use cubek_microkernel::attention::plane::PlaneShuffleAttentionMicrokernelSettings;
use cubek_microkernel::attention::{AttentionOperand, AttentionVisibility};
use cubek_microkernel::split_keys_merge::hypercube::HypercubeMergeShape;
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{Breakable, CubeResources, Grid, Routine};

/// Attention with several cubes owning every box of queries, every cube walking its own run of the
/// keys: the hypercube microkernel runs a cube of plane shuffles over a cube's run, its planes
/// along the run, and each plane leaves its queries' partial in a slot of its own. Where `M` says
/// ([`HypercubeShuffleMerge`]), the last cube of a box merges the slots into the output, or the
/// routine stops at the slots, for the split-keys merge launched after it. The cubes cover the
/// queries, the heads and the runs, one batch and key head each.
///
/// What few queries against many keys want, a decode step's above all: a box's queries leave the
/// device idle where only one cube walks their keys, and the cubes sharing the box read no key
/// twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HypercubeShuffleAttentionRoutine<M> {
    merge: PhantomData<M>,
}

/// Where a hypercube shuffle's partials meet: what its two routines differ in, each owning the
/// kernel it launches.
pub trait HypercubeShuffleMerge: core::fmt::Debug + Copy + Eq + core::hash::Hash {
    const MERGE: HypercubeMerge;
    /// What a report calls the routine, in kebab-case.
    const NAME: &'static str;
}

/// What a hypercube shuffle's kernel takes alike whether it reads its keys and values as tiles or
/// decodes a packed cache, each built once: the launch's grid, the operands bound beside the cache
/// under their launcher, the slots, and the problem's scalars.
pub(crate) struct HypercubeShuffleLaunch<'a> {
    pub client: &'a Client,
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
    /// Scalars a vector of every operand holds: the plane shuffle lines every one alike.
    pub width_scalars: usize,
    pub launcher: Launcher,
    pub query: TileTensorLaunch,
    pub mask: Option<TileTensorLaunch>,
    pub lengths: TensorArg,
    pub slots: [TensorArg; 3],
    pub query_count: u32,
    pub key_count: u32,
    pub scale: f32,
    pub microkernel: HypercubeAttentionMicrokernel,
    pub shape: HypercubeMergeShape,
    pub visibility: AttentionVisibility,
    /// The elements a kernel over tiles reads each operand in.
    pub operand_elements: [ElemType; 5],
    /// The elements a kernel over a packed cache computes in.
    pub kernel_elements: [ElemType; 2],
}

impl<'a> HypercubeShuffleLaunch<'a> {
    /// The launch `blueprint` prepared on `grid` over `tensors`, bound as `binding`, its cubes
    /// leaving their partials in `slots` laid out as `shape`; and the cache, which each kernel
    /// reads its own way.
    pub(crate) fn new(
        tensors: &'a AttentionTensors,
        binding: AttentionBinding,
        blueprint: &HypercubeShuffleAttentionRoutineBlueprint,
        grid: Grid,
        shape: HypercubeMergeShape,
        slots: &AttentionSplitSlots,
    ) -> (Self, AttentionCacheBinding) {
        let problem = tensors.problem();
        let launch = Self {
            client: tensors.client(),
            cube_count: grid.launch.cube_count,
            cube_dim: grid.launch.cube_dim,
            width_scalars: blueprint.widths.query,
            launcher: binding.launcher,
            query: binding.query,
            mask: binding.mask,
            lengths: binding.lengths,
            slots: slots.to_args(),
            query_count: problem.seq_q as u32,
            key_count: problem.seq_kv as u32,
            scale: tensors.scale(),
            microkernel: blueprint.microkernel,
            shape,
            visibility: problem.visibility,
            operand_elements: problem.work_item().operand_elements(),
            kernel_elements: problem.work_item().kernel_elements(),
        };
        (launch, binding.cache)
    }
}

pub type HypercubeShuffleAttentionRoutineBlueprint =
    AttentionBlueprint<HypercubeAttentionMicrokernel, StageStorage>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeShuffleAttentionRoutineSettings<M> {
    pub microkernel: HypercubeAttentionMicrokernelSettings,
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
        microkernel: HypercubeAttentionMicrokernelSettings::SHUFFLE,
        cubes_along_keys: NonZeroUsize::new(32).unwrap(),
        merge: PhantomData,
    };
}

impl<M: HypercubeShuffleMerge> core::fmt::Display for HypercubeShuffleAttentionRoutineSettings<M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} {} cubes_along_keys={}",
            M::NAME,
            self.microkernel,
            self.cubes_along_keys
        )
    }
}

impl<M: HypercubeShuffleMerge> Routine for HypercubeShuffleAttentionRoutine<M> {
    type Problem = AttentionRoutineProblem;
    type Microkernel = HypercubeAttentionMicrokernel;
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
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        if M::MERGE.admits(hw).is_err() {
            return Vec::new();
        }
        let count = |count: usize| NonZeroUsize::new(count).unwrap();
        let lead = Self::Settings::DEFAULT;
        let mut settings: Vec<Self::Settings> = [8, 16, 32]
            .into_iter()
            .flat_map(|cubes| {
                HypercubeAttentionMicrokernelSettings::SHUFFLE
                    .listed(hw)
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
                    plane: CubeAttentionPlaneSettings::Shuffle(plane),
                    ..lead.microkernel.cube
                },
            },
            ..lead
        };
        let lead_plane = PlaneShuffleAttentionMicrokernelSettings::DEFAULT;
        settings.extend([2, 4].map(|cubes| Self::Settings {
            cubes_along_keys: count(cubes),
            ..lead
        }));
        settings.extend([1, 8].map(|keys_in_flight| {
            with_plane(PlaneShuffleAttentionMicrokernelSettings {
                keys_in_flight: count(keys_in_flight),
                ..lead_plane
            })
        }));
        for (units_per_key, keys_in_flight) in [(4, 1), (4, 2), (4, 4), (4, 8), (8, 1), (8, 8)] {
            settings.push(with_plane(PlaneShuffleAttentionMicrokernelSettings {
                units_per_key: count(units_per_key),
                keys_in_flight: count(keys_in_flight),
                ..lead_plane
            }));
        }
        settings.push(with_plane(PlaneShuffleAttentionMicrokernelSettings {
            units_per_key: count(32),
            ..lead_plane
        }));
        settings
    }

    /// Nothing of its own: it only distributes the cubes, and their microkernel makes every
    /// access.
    fn vectorize(
        microkernel: &HypercubeAttentionMicrokernel,
        partitioning: &AttentionPartitioning,
        vectorization: &mut Vectorization<AttentionOperand>,
    ) {
        microkernel.vectorize(&partitioning.cube_space(), vectorization);
    }

    /// Nothing of its own: its cubes' microkernel walks each cube's run of the keys and counts
    /// each cube's partial.
    fn estimate(
        problem: &AttentionRoutineProblem,
        blueprint: &HypercubeShuffleAttentionRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        blueprint.microkernel.estimate(
            &problem.work_item(),
            &blueprint.partitioning.cube_space(),
            grid.planes(),
            estimate,
        );
    }

    /// A cube holds its planes along the keys, and the shared memory its microkernel and its merge
    /// ask for. The cubes sharing a box are its settings' count, whatever the keys
    /// ([`AttentionPartitioning::new`]). Declines where the device cannot merge this way, and
    /// serves a cache of any length, at any device fill: neither is a call's to decline, since an
    /// autotune key prepares its candidates at one representative for every call of its bucket.
    /// A second launch's merge is prepared by the strategy that launches it.
    fn prepare(
        hw: &Hardware,
        problem: &AttentionRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Decline> {
        if problem.packed_queries {
            return Err(PACKED_QUERIES_UNREAD);
        }
        M::MERGE.admits(hw)?;
        let item = problem.work_item();
        let microkernel = HypercubeAttentionMicrokernel::best(hw, &item, &settings.microkernel)?;
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
        let mut vectorization = Vectorization::new();
        Self::vectorize(&microkernel, &partitioning, &mut vectorization);
        let blueprint = AttentionBlueprint::new(
            hw,
            problem,
            microkernel,
            partitioning,
            vectorization,
            storage,
        );
        // The plane shuffle lines every operand alike, so its tiles share one width; a packed cache
        // binds no tile.
        let widths = &blueprint.widths;
        let packed = matches!(problem.cache, AttentionCacheLayout::Packed(_));
        if !packed && [widths.key, widths.value] != [widths.query; 2] {
            return Err(PLANE_SHUFFLE_WIDTHS_DIFFER).or_broken(PLANE_SHUFFLE_ONE_WIDTH);
        }
        Ok((blueprint, grid))
    }
}

impl<M: HypercubeShuffleMerge> HypercubeShuffleAttentionRoutine<M> {
    /// How `blueprint` lays out the slots its cubes leave for `problem`, as its merge reads them.
    pub(crate) fn merge_shape(
        &self,
        problem: &AttentionRoutineProblem,
        blueprint: &HypercubeShuffleAttentionRoutineBlueprint,
    ) -> HypercubeMergeShape {
        blueprint.microkernel.merge_shape(
            &problem.work_item(),
            blueprint.partitioning.cubes_along_keys(),
            M::MERGE.seq_q(&blueprint.microkernel),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{
        AttentionCacheLayout, HYPERCUBE_CUBES_UNMET, HypercubeLastArrivalShuffleAttentionRoutine,
        HypercubeShufflePartialAttentionRoutine,
    };
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_microkernel::attention::{Q, S};
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// A decode step over a long cache, and a short chunk of a prompt against one whose keys do
    /// not fill the last step: each long enough for every cube of every setting to walk some.
    fn problems() -> [AttentionRoutineProblem; 2] {
        [
            AttentionRoutineProblem::dense((1, 32, 8), (1, 4096), (128, 128), F16),
            AttentionRoutineProblem::dense((1, 8, 2), (16, 4000), (64, 64), F32),
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

    /// Every setting `R` lists on each of `devices` prepares.
    fn assert_every_setting_prepares<R: Routine<Problem = AttentionRoutineProblem>>(
        devices: &[Hardware],
    ) {
        for hw in devices {
            let settings = R::settings(hw);
            assert!(!settings.is_empty());
            for settings in settings {
                for problem in problems() {
                    let prepared = R::prepare(hw, &problem, &settings);
                    assert!(prepared.is_ok(), "{settings} on {hw:?}: {prepared:?}");
                }
            }
        }
    }

    /// Every setting of either routine prepares, the partial one on a device whose cubes cannot
    /// meet within a launch too.
    #[test]
    fn every_setting_prepares() {
        let devices = [devices::m2_pro(), devices::h100()];
        assert_every_setting_prepares::<HypercubeLastArrivalShuffleAttentionRoutine>(&devices);
        assert_every_setting_prepares::<HypercubeShufflePartialAttentionRoutine>(&[
            devices::m2_pro(),
            devices::h100(),
            unmet(devices::m2_pro()),
        ]);
    }

    /// Every setting serves a cache of any length, down to one key, and a prompt whose boxes of
    /// queries already fill the device: a call inside an autotune key's bucket must run what the
    /// key elected at its representative.
    #[test]
    fn every_setting_serves_any_cache_length_and_fill() {
        let hw = devices::h100();
        let decode = |keys| AttentionRoutineProblem::dense((1, 32, 8), (1, keys), (128, 128), F16);
        let prompt = AttentionRoutineProblem::dense((1, 32, 32), (4096, 4096), (128, 128), F16);
        for settings in HypercubeShufflePartialAttentionRoutine::settings(&hw) {
            for problem in [
                decode(1),
                decode(5),
                decode(100),
                decode(1500),
                prompt.clone(),
            ] {
                let prepared =
                    HypercubeShufflePartialAttentionRoutine::prepare(&hw, &problem, &settings);
                assert!(prepared.is_ok(), "{settings}: {:?}", prepared.err());
            }
        }
    }

    /// Every setting declines where the device commits to no plane width, rather than launching
    /// on lanes it would misread.
    #[test]
    fn every_setting_declines_without_a_plane_width() {
        fn declines<R: Routine<Problem = AttentionRoutineProblem>>() {
            for hw in [devices::wgpu_no_subgroups(), devices::rdna_range()] {
                for settings in R::settings(&devices::m2_pro()) {
                    for problem in problems() {
                        let declined = R::prepare(&hw, &problem, &settings);
                        assert!(declined.is_err(), "{settings} on {hw:?}: {declined:?}");
                    }
                }
            }
        }
        declines::<HypercubeLastArrivalShuffleAttentionRoutine>();
        declines::<HypercubeShufflePartialAttentionRoutine>();
    }

    /// The last arrival's last cube reads the others' slots within the launch, so where one cube's
    /// stores do not reach another it lists nothing, and its default declines saying why.
    #[test]
    fn the_last_arrival_declines_where_cubes_cannot_meet() {
        let hw = unmet(devices::m2_pro());
        assert!(HypercubeLastArrivalShuffleAttentionRoutine::settings(&hw).is_empty());
        let [problem, _] = problems();
        let declined = HypercubeLastArrivalShuffleAttentionRoutine::prepare(
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
        let (blueprint, grid) = HypercubeLastArrivalShuffleAttentionRoutine::prepare(
            &devices::m2_pro(),
            &problem,
            &HypercubeShuffleAttentionRoutineSettings::DEFAULT,
        )
        .expect("the default prepares on the M2");
        let partitioning = &blueprint.partitioning;
        assert_eq!(partitioning.cubes_along_keys().get(), 32);
        assert_eq!(partitioning.stage().extent(S), 4);
        assert_eq!(partitioning.cube_space().extent(S), 128);
        assert_eq!(partitioning.cube_space().extent(Q), 4);
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
        let (blueprint, _) = HypercubeLastArrivalShuffleAttentionRoutine::prepare(
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
        assert_eq!(partitioning.cube_space().extent(S), 128);
    }
}
