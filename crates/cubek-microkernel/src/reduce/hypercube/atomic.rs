use crate::reduce::cube::{
    CubeStagedReduceMicrokernel, CubeStagedReduceMicrokernelSettings,
    cube_staged_reduce_microkernel,
};
use crate::reduce::{Accumulable, ReduceOperand, ReduceOperation, ReduceWorkItem};
use cubecl::tile::launch::{Decline, Vectorization};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// Rows split across cubes, each adding the sum of its part into the rows' cells of an output that
/// holds zero before it, by atomic adds.
///
/// It sums, and runs where the cube's microkernel runs. That the device adds atomically in the output's
/// element is its launch's to check, which knows it.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeAtomicReduceMicrokernel {
    #[cube(comptime)]
    part: CubeStagedReduceMicrokernel,
}

impl HypercubeAtomicReduceMicrokernel {
    pub fn new(part: CubeStagedReduceMicrokernel) -> Self {
        Self { part }
    }

    /// The microkernel each cube sums its part with.
    pub fn part(&self) -> CubeStagedReduceMicrokernel {
        self.part
    }
}

#[cube]
impl HypercubeAtomicReduceMicrokernel {
    /// `cube`'s part of the rows of `input` added into their cells of `out`
    /// (`hypercube_atomic_reduce_microkernel`).
    ///
    /// # Assumptions
    ///
    /// - Those of [`ReduceWorkItem`](crate::reduce::ReduceWorkItem), of its part's item.
    /// - Its part's item is a sum, what [`admits`](Microkernel::admits) accepts.
    /// - Every cell of `out` holds zero before the launch: each cube only adds into it.
    /// - The cubes of the launch split each row into parts, each part some cube's alone.
    ///
    /// # Panics
    ///
    /// When the kernel compiles, where its part's item breaks
    /// [`ReduceWorkItem`](crate::reduce::ReduceWorkItem)'s Assumptions on `indices`, as a sum
    /// written with its indices.
    pub fn reduce<
        In: Numeric,
        Acc: Numeric,
        Out: Numeric,
        N: Size,
        C: Accumulable<Acc, N> + Clone,
    >(
        &self,
        cube: &Region,
        input: &Tile,
        out: &Tile,
    ) {
        hypercube_atomic_reduce_microkernel::<In, Acc, Out, N, C>(
            cube,
            input,
            out,
            comptime!(HypercubeAtomicReduceMicrokernel { part: self.part }),
        )
    }
}

impl core::fmt::Display for HypercubeAtomicReduceMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.part)
    }
}

/// Why an operation other than a sum declines: the cubes meet by atomic adds.
const ONLY_SUMS: Decline = Decline::Unserved {
    why: "the atomic hypercube only sums",
};

impl Microkernel for HypercubeAtomicReduceMicrokernel {
    type WorkItem = ReduceWorkItem;
    type Settings = CubeStagedReduceMicrokernelSettings;
    type Operand = ReduceOperand;
    type CubeSpace = ();

    /// The cube's: a device that runs the cube runs the hypercube built on it.
    const DECLINE: &'static str = CubeStagedReduceMicrokernel::DECLINE;

    /// As the cube it is built from.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        CubeStagedReduceMicrokernel::settings(hw)
    }

    fn candidates(
        hw: &Hardware,
        item: &ReduceWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let admitted = Self::admits(item).is_ok();
        CubeStagedReduceMicrokernel::candidates(hw, item, settings)
            .filter(move |_| admitted)
            .map(Self::new)
    }

    fn admits(item: &ReduceWorkItem) -> Result<(), Decline> {
        match item.operation {
            ReduceOperation::Sum => Ok(()),
            _ => Err(ONLY_SUMS),
        }
    }

    /// The hypercube of the cube's best where `item` is a sum, the cube's own Decline coming through
    /// where the device runs no cube.
    fn best(
        hw: &Hardware,
        item: &ReduceWorkItem,
        settings: &Self::Settings,
    ) -> Result<Self, Decline> {
        Self::admits(item)?;
        CubeStagedReduceMicrokernel::best(hw, item, settings).map(Self::new)
    }

    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<ReduceOperand>) {
        self.part.vectorize(&(), vectorization);
    }

    fn estimate(
        &self,
        item: &ReduceWorkItem,
        space: &(),
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        self.part.estimate(item, space, planes, estimate);
    }
}

/// A cube adds into cells other cubes of the launch add into too: the sum is whole once all have.
#[cube]
pub(crate) fn hypercube_atomic_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    C: Accumulable<Acc, N> + Clone,
>(
    cube: &Region,
    input: &Tile,
    out: &Tile,
    #[comptime] microkernel: HypercubeAtomicReduceMicrokernel,
) {
    cube_staged_reduce_microkernel::<In, Acc, Out, Out, N, C>(
        cube,
        input,
        out,
        &ComptimeOption::new_None(),
        microkernel.part,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::testing::HypercubeAtomicMicrokernelLauncher;
    use cubek_std::devices;

    #[test]
    fn it_reduces_no_max() {
        let hw = devices::m2_pro();
        let settings = HypercubeAtomicReduceMicrokernel::settings(&hw)[0];
        let item = ReduceWorkItem::along_f32(ReduceOperation::Max);
        assert_eq!(
            HypercubeAtomicReduceMicrokernel::best(&hw, &item, &settings),
            Err(ONLY_SUMS)
        );
    }

    #[test]
    fn it_lists_no_candidate_for_a_max() {
        let hw = devices::m2_pro();
        let settings = HypercubeAtomicReduceMicrokernel::settings(&hw)[0];
        let item = ReduceWorkItem::along_f32(ReduceOperation::Max);
        assert_eq!(
            HypercubeAtomicReduceMicrokernel::candidates(&hw, &item, &settings).count(),
            0
        );
    }

    /// The cube's own Decline comes through: a range of plane widths.
    #[test]
    fn it_declines_where_its_cube_declines() {
        let hw = devices::rdna_range();
        let settings = HypercubeAtomicReduceMicrokernel::settings(&hw)[0];
        let item = ReduceWorkItem::along_f32(ReduceOperation::Sum);
        let declined = HypercubeAtomicReduceMicrokernel::best(&hw, &item, &settings);
        assert!(
            matches!(declined, Err(Decline::ImprecisePlaneDim { .. })),
            "{declined:?}"
        );
    }

    #[test]
    fn its_part_is_the_cube_it_is_built_from() {
        let (lanes, planes, lane_slots) = (32, 4, 32);
        let item = ReduceWorkItem::along_f32(ReduceOperation::Sum);
        let part = CubeStagedReduceMicrokernel::new(item, lanes, planes, lane_slots);
        assert_eq!(HypercubeAtomicReduceMicrokernel::new(part).part(), part);
    }

    /// Rows 300 values long along the reduced dimension, a value a line, in three parts.
    mod along_three_parts {
        use super::*;
        reduce_tests!(HypercubeAtomicMicrokernelLauncher::new(4, 3), Along, 300, 1, Plain, [
            sum: ReduceOperation::Sum,
        ]);
    }

    /// The same across it, each lane its own row.
    mod across_three_parts {
        use super::*;
        reduce_tests!(HypercubeAtomicMicrokernelLauncher::new(4, 3), Across, 300, 4, Plain, [
            sum: ReduceOperation::Sum,
        ]);
    }

    #[test]
    fn it_displays_its_part() {
        let (lanes, planes, lane_slots) = (32, 4, 32);
        let item = ReduceWorkItem::along_f32(ReduceOperation::Sum);
        let part = CubeStagedReduceMicrokernel::new(item, lanes, planes, lane_slots);
        let hypercube = HypercubeAtomicReduceMicrokernel::new(part);
        assert_eq!(hypercube.to_string(), part.to_string());
    }
}
