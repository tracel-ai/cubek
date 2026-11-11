use crate::kernels::{
    AttentionBinding, AttentionBlueprint, AttentionRoutineProblem, AttentionSplitSlots,
    AttentionTensors,
};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention::AttentionMicrokernel;
use cubek_microkernel::split_keys_merge::hypercube::HypercubeMergeShape;
use cubek_std::Hardware;
use cubek_std::routine::{Grid, ROUTINE_LISTS_NO_SETTING, Routine};

/// A routine around microkernel `M`, laying out shared memory as `SM`, that writes the attention:
/// it prepares an [`AttentionBlueprint`] and launches it on tensors.
///
/// A routine states what is its own, how it places its microkernel's work items and the kernel it
/// dispatches. The launch around those is written once, in [`OneRoutine`].
pub trait AttentionRoutine<M: AttentionMicrokernel, SM = ()>:
    Routine<Problem = AttentionRoutineProblem, Microkernel = M, Blueprint = AttentionBlueprint<M, SM>>
{
    /// Launch the kernel `blueprint` compiles, on `grid`, over `binding`.
    fn dispatch(
        &self,
        tensors: &AttentionTensors,
        binding: AttentionBinding,
        blueprint: AttentionBlueprint<M, SM>,
        grid: Grid,
    );
}

/// What writes the attention at each of its routine's settings: a routine alone ([`OneRoutine`]),
/// or a partial routine then the split-keys merge
/// ([`SplitKeysAttention`](crate::kernels::SplitKeysAttention)). What a test or a benchmark row
/// launches, one setting at a time.
pub trait AttentionLaunch {
    /// The routine whose settings a launch is at: the one that writes the attention, or the
    /// partial one, since the merge has none.
    type Routine: Routine<Problem = AttentionRoutineProblem>;

    /// The routine whose settings a launch is at.
    fn routine(&self) -> &Self::Routine;

    /// The attention `tensors` state, launched on their client's device, prepared for `hw` at
    /// `settings`: the device as it reports itself, or narrowed to offer less, so a candidate
    /// further down a microkernel's list is the one that runs.
    ///
    /// # Errors
    ///
    /// Why `hw` cannot run the attention at these settings.
    fn launch_on(
        &self,
        hw: &Hardware,
        tensors: AttentionTensors,
        settings: &<Self::Routine as Routine>::Settings,
    ) -> Result<TensorHandle, Refusal>;

    /// [`launch_on`](Self::launch_on) at the routine's lead, the first setting it lists on `hw`:
    /// what a caller naming no setting runs.
    ///
    /// # Errors
    ///
    /// Why `hw` cannot run the attention at the lead.
    fn launch_lead_on(
        &self,
        hw: &Hardware,
        tensors: AttentionTensors,
    ) -> Result<TensorHandle, Refusal> {
        let lead = self
            .routine()
            .settings(hw)
            .into_iter()
            .next()
            .ok_or(ROUTINE_LISTS_NO_SETTING)?;
        self.launch_on(hw, tensors, &lead)
    }
}

/// The attention as routine `R` writes it, launched alone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OneRoutine<R>(pub R);

impl<R, M: AttentionMicrokernel, SM> AttentionLaunch for OneRoutine<R>
where
    R: AttentionRoutine<M, SM> + Routine<Microkernel = M, Blueprint = AttentionBlueprint<M, SM>>,
{
    type Routine = R;

    fn routine(&self) -> &R {
        &self.0
    }

    fn launch_on(
        &self,
        hw: &Hardware,
        tensors: AttentionTensors,
        settings: &R::Settings,
    ) -> Result<TensorHandle, Refusal> {
        let (blueprint, grid) = self.0.prepare(hw, &tensors.problem(), settings)?;
        let binding = tensors.bind(&blueprint, &grid);
        self.0.dispatch(&tensors, binding, blueprint, grid);
        Ok(tensors.into_out())
    }
}

/// A routine around microkernel `M`, laying out shared memory as `SM`, that writes partial
/// attention: the cubes sharing each box of queries walk their own runs of the keys, each leaving
/// its queries' output, unnormalized, and their softmax state in a slot of its own.
///
/// It writes no output. The split-keys merge
/// ([`PlaneSplitKeysMergeRoutine`](crate::kernels::PlaneSplitKeysMergeRoutine)) merges the slots
/// into it, and a strategy launches the two ([`SplitKeysAttention`](crate::kernels::SplitKeysAttention)).
pub trait PartialAttentionRoutine<M: AttentionMicrokernel, SM = ()>:
    Routine<Problem = AttentionRoutineProblem, Microkernel = M, Blueprint = AttentionBlueprint<M, SM>>
{
    /// How the slots `blueprint` leaves for `problem` lie, as the merge reads them.
    fn merge_shape(
        &self,
        problem: &AttentionRoutineProblem,
        blueprint: &AttentionBlueprint<M, SM>,
    ) -> HypercubeMergeShape;

    /// The width of the lines the slots' outputs are stored at.
    fn slot_width(&self, blueprint: &AttentionBlueprint<M, SM>) -> usize;

    /// Launch the kernel `blueprint` compiles, on `grid`, over `binding`, its partials left in
    /// `slots`.
    fn dispatch(
        &self,
        tensors: &AttentionTensors,
        binding: AttentionBinding,
        blueprint: AttentionBlueprint<M, SM>,
        grid: Grid,
        slots: &AttentionSplitSlots,
    );
}

#[cfg(test)]
mod tests {
    use crate::kernels::tune::key::AttentionKey;
    use crate::kernels::*;
    use cubecl::features::MmaConfig;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_std::routine::Routine;

    use cubek_std::Hardware;
    use cubek_std::devices;
    use cubek_std::microkernel::KernelEstimate;
    use cubek_std::routine::EstimatedProblem;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// Every routine's estimate at every setting it lists, on a prompt and on a decode step: each
    /// owner down the walk set its own part, and the cubes perform at least what the problem asks.
    #[test]
    fn every_routine_estimates_every_setting_whole() {
        assert_every_estimate_is_whole(UnitAttentionRoutine);
        assert_every_estimate_is_whole(CubeStagedAttentionRoutine);
        assert_every_estimate_is_whole(HypercubeLastArrivalShuffleAttentionRoutine::new());
        assert_every_estimate_is_whole(HypercubeShufflePartialAttentionRoutine::new());
    }

    /// An M2 Pro offering the square `8×8×8` instruction for `f16` and `f32` summed in `f32`.
    fn m2_pro() -> Hardware {
        let square = |element| MmaConfig {
            a_type: element,
            b_type: element,
            cd_type: F32,
            m: 8,
            n: 8,
            k: 8,
        };
        devices::offering_cmma(devices::m2_pro(), [square(F16), square(F32)])
    }

    /// Qwen3-8B's attention: a prompt of 512 positions, and a decode step against a cache of 4096,
    /// the problems autotune reads off its keys.
    fn problems() -> [AttentionRoutineProblem; 2] {
        [(512, 512, true), (1, 4096, false)].map(|(queries, keys, causal)| {
            let visibility = AttentionVisibility::new(causal, None, QueryPlacement::Last);
            let key = AttentionKey::new(
                F16,
                32,
                8,
                128,
                queries,
                keys,
                1,
                visibility,
                ScoreScale::standard(128),
                None,
                false,
                false,
            );
            AttentionRoutineProblem::try_from(&key).expect("an even group")
        })
    }

    fn assert_every_estimate_is_whole<M, SM, R>(routine: R)
    where
        R: Routine<
                Problem = AttentionRoutineProblem,
                Microkernel = M,
                Blueprint = AttentionBlueprint<M, SM>,
            >,
    {
        let hw = &m2_pro();
        let mut prepared = 0;
        for settings in routine.settings(hw) {
            for problem in problems() {
                let Ok((blueprint, grid)) = routine.prepare(hw, &problem, &settings) else {
                    continue;
                };
                prepared += 1;
                let mut estimate = KernelEstimate::from(&grid);
                routine.estimate(&problem, &blueprint, &grid, &mut estimate);
                let (staging, cubes) = (estimate.staging.as_ref(), estimate.cubes);
                assert!(estimate.work.is_some(), "{settings}: no work");
                let staging = staging.unwrap_or_else(|| panic!("{settings}: no staging"));
                assert!(
                    staging.stages > 0 && !staging.reads.is_empty(),
                    "{settings}"
                );
                let performed = staging.operations * cubes.expect("a static grid") as f64;
                let useful = problem.demand().compute_ops as f64;
                assert!(performed >= useful, "{settings}: {performed} < {useful}");
            }
        }
        assert!(prepared > 0, "no setting of the routine prepares here");
    }
}
