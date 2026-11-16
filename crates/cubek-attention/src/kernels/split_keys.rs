//! Attention as two routines launched in order: a partial routine, then the split-keys merge.

use crate::kernels::{
    AttentionBlueprint, AttentionLaunch, AttentionRoutineProblem, AttentionTensors,
    PartialAttentionRoutine, PlaneSplitKeysMergeBlueprint, PlaneSplitKeysMergeRoutine,
};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention::AttentionMicrokernel;
use cubek_std::Hardware;
use cubek_std::routine::{Grid, Routine};

/// Attention in two launches, each its own routine: the partial routine `R`, whose cubes sharing a
/// box of queries each leave their run of the keys in a slot, then the split-keys merge of the
/// slots into the output ([`PlaneSplitKeysMergeRoutine`]).
///
/// A strategy that launches two routines. A pin of a hypercube merged by a second launch runs it,
/// and so does such an autotune candidate. Its settings are the partial routine's, since the merge
/// has none.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SplitKeysAttention<R> {
    pub partial: R,
}

impl<R, M: AttentionMicrokernel, SM> SplitKeysAttention<R>
where
    R: PartialAttentionRoutine<M, SM>
        + Routine<Microkernel = M, Blueprint = AttentionBlueprint<M, SM>>,
{
    /// The merge of the slots `blueprint`, the partial routine's launch, leaves for `problem`.
    ///
    /// # Errors
    ///
    /// Why the device cannot launch the merge.
    pub fn merge(
        &self,
        hw: &Hardware,
        problem: &AttentionRoutineProblem,
        blueprint: &AttentionBlueprint<M, SM>,
    ) -> Result<PlaneSplitKeysMergeBlueprint, Refusal> {
        PlaneSplitKeysMergeRoutine.prepare(
            hw,
            problem,
            self.partial.merge_shape(problem, blueprint),
            self.partial.slot_width(blueprint),
        )
    }

    /// Both launches in order over `tensors`: the partial routine's `blueprint` on `grid`, then
    /// `merge` over the slots it leaves.
    pub fn dispatch(
        &self,
        tensors: &AttentionTensors,
        blueprint: AttentionBlueprint<M, SM>,
        grid: Grid,
        merge: PlaneSplitKeysMergeBlueprint,
    ) {
        let slots = tensors.split_slots(merge.merge.slots.splits);
        let binding = tensors.bind(&blueprint, &grid);
        self.partial
            .dispatch(tensors, binding, blueprint, grid, &slots);
        PlaneSplitKeysMergeRoutine.dispatch(tensors, merge, slots);
    }
}

impl<R, M: AttentionMicrokernel, SM> AttentionLaunch for SplitKeysAttention<R>
where
    R: PartialAttentionRoutine<M, SM>
        + Routine<Microkernel = M, Blueprint = AttentionBlueprint<M, SM>>,
{
    type Routine = R;

    /// The partial routine: the merge has no settings.
    fn routine(&self) -> &R {
        &self.partial
    }

    /// Both routines prepared for `hw`, the partial one at `settings`, then launched in order.
    fn launch_on(
        &self,
        hw: &Hardware,
        tensors: AttentionTensors,
        settings: &R::Settings,
    ) -> Result<TensorHandle, Refusal> {
        let problem = tensors.problem();
        let (blueprint, grid) = self.partial.prepare(hw, &problem, settings)?;
        let merge = self.merge(hw, &problem, &blueprint)?;
        self.dispatch(&tensors, blueprint, grid, merge);
        Ok(tensors.into_out())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::test_support::{offering, qwen};
    use crate::kernels::{
        HypercubeFragmentPartialAttentionRoutine, HypercubeShufflePartialAttentionRoutine,
    };
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    /// Both routines of `split` prepare at every setting the partial one lists on `hw`, for each
    /// of `problems`: the merge serves every slot shape the partial routine leaves.
    fn assert_both_prepare<R, M: AttentionMicrokernel, SM>(
        split: SplitKeysAttention<R>,
        hw: &Hardware,
        problems: &[AttentionRoutineProblem],
    ) where
        R: PartialAttentionRoutine<M, SM>
            + Routine<Microkernel = M, Blueprint = AttentionBlueprint<M, SM>>,
    {
        let mut merged = 0;
        for settings in split.partial.settings(hw) {
            for problem in problems {
                let Ok((blueprint, _)) = split.partial.prepare(hw, problem, &settings) else {
                    continue;
                };
                let merge = split.merge(hw, problem, &blueprint);
                assert!(merge.is_ok(), "{settings}: {merge:?}");
                merged += 1;
            }
        }
        assert!(
            merged > 0,
            "no setting of the partial routine prepares on {hw:?}"
        );
    }

    /// The shuffle's partials are merged at every setting, on a decode step over a cache of any
    /// length and on a prompt that already fills the device, a device whose cubes cannot meet
    /// within a launch included.
    #[test]
    fn the_shuffle_partials_merge_at_every_setting() {
        let decode = |keys| AttentionRoutineProblem::dense((1, 32, 8), (1, keys), (128, 128), F16);
        let prompt = AttentionRoutineProblem::dense((1, 32, 32), (4096, 4096), (128, 128), F16);
        let problems = [decode(1), decode(100), decode(4096), prompt];
        let mut unmet = devices::m2_pro();
        devices::features(&mut unmet).device_memory_scope = false;
        for hw in [devices::m2_pro(), devices::h100(), unmet] {
            assert_both_prepare(
                SplitKeysAttention {
                    partial: HypercubeShufflePartialAttentionRoutine::new(),
                },
                &hw,
                &problems,
            );
        }
    }

    /// The fragments' partials are merged at every setting, on a decode step and a prompt's chunk.
    #[test]
    fn the_fragment_partials_merge_at_every_setting() {
        for hw in [
            offering(devices::m2_pro(), 8),
            offering(devices::h100(), 16),
        ] {
            assert_both_prepare(
                SplitKeysAttention {
                    partial: HypercubeFragmentPartialAttentionRoutine,
                },
                &hw,
                &[qwen(1, 4096), qwen(16, 1000)],
            );
        }
    }
}
