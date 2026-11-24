use crate::reduce::cube::{
    CubeStagedReduceMicrokernel, CubeStagedReduceMicrokernelSettings,
    cube_staged_reduce_microkernel,
};
use crate::reduce::{Accumulable, ReduceOperand, ReduceOperation, ReduceWorkItem};
use cubecl::ir::Type;
use cubecl::ir::features::AtomicUsage;
use cubecl::tile::launch::Vectorization;
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// Rows split across cubes, each adding the sum of its part into the rows' cells of an output that
/// holds zero before it, by atomic adds.
///
/// It sums, and runs where the cube's microkernel runs and the device adds atomically in the
/// accumulator's element, which its output is written in.
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

/// The atomic hypercube of each of the cube's candidates, where `item` sums and the device adds
/// atomically in its accumulator's element.
fn candidates_adding_atomically(
    hw: &Hardware,
    item: &ReduceWorkItem,
    settings: &CubeStagedReduceMicrokernelSettings,
) -> impl Iterator<Item = HypercubeAtomicReduceMicrokernel> {
    let adds = hw
        .properties
        .atomic_type_usage(Type::atomic(item.accumulator))
        .contains(AtomicUsage::Add);
    let sums = item.operation == ReduceOperation::Sum;
    CubeStagedReduceMicrokernel::candidates(hw, item, settings)
        .filter(move |_| adds && sums)
        .map(HypercubeAtomicReduceMicrokernel::new)
}

impl Microkernel for HypercubeAtomicReduceMicrokernel {
    type WorkItem = ReduceWorkItem;
    type Settings = CubeStagedReduceMicrokernelSettings;
    type Operand = ReduceOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the atomic hypercube needs the cube reduce, a sum, and an atomic add of the accumulator's element";

    /// As the cube it is built from.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        CubeStagedReduceMicrokernel::settings(hw)
    }

    fn candidates(
        hw: &Hardware,
        item: &ReduceWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        candidates_adding_atomically(hw, item, settings)
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

    /// An atomic hypercube of a sum, on an M2 Pro.
    fn best_of(operation: ReduceOperation) -> Result<HypercubeAtomicReduceMicrokernel, ()> {
        let hw = devices::m2_pro();
        let settings = HypercubeAtomicReduceMicrokernel::settings(&hw)[0];
        let item = ReduceWorkItem::along_f32(operation);
        HypercubeAtomicReduceMicrokernel::best(&hw, &item, &settings).map_err(|_| ())
    }

    #[test]
    fn it_reduces_no_max() {
        assert!(best_of(ReduceOperation::Max).is_err());
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
        reduce_tests!(HypercubeAtomicMicrokernelLauncher { planes: 4, parts: 3 }, Along, 300, 1, Plain, [
            sum: ReduceOperation::Sum,
        ]);
    }

    /// The same across it, each lane its own row.
    mod across_three_parts {
        use super::*;
        reduce_tests!(HypercubeAtomicMicrokernelLauncher { planes: 4, parts: 3 }, Across, 300, 4, Plain, [
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
