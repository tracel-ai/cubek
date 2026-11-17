use crate::reduce::cube::{CubeReduceMicrokernel, CubeReduceMicrokernelSettings};
use crate::reduce::{ReduceOperand, ReduceWorkItem};
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// A row split across cubes, each reducing its part with the cube's microkernel, then storing its
/// partial apart: the last of the row's cubes to finish merges the partials and writes the row.
///
/// It runs where the cube's microkernel runs and where one cube's stores reach another within a
/// launch (`device_memory_scope`), which is how the cubes meet: on WGSL, whose storage barrier
/// orders a workgroup's own memory only, the last cube was seen reading another's partial as the
/// zero it held before the write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeReduceMicrokernel {
    /// The cube microkernel each cube reduces its part with.
    pub part: CubeReduceMicrokernel,
}

impl HypercubeReduceMicrokernel {
    /// Bytes of shared memory each cube allocates with its lines `width` lanes wide: its part's,
    /// and the slot its first unit hands the cube its arrival through.
    pub fn shared_memory_bytes_at_width(&self, width: usize) -> usize {
        self.part.shared_memory_bytes_at_width(width) + size_of::<u32>()
    }
}

impl core::fmt::Display for HypercubeReduceMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.part)
    }
}

impl Microkernel for HypercubeReduceMicrokernel {
    type WorkItem = ReduceWorkItem;
    type Settings = CubeReduceMicrokernelSettings;
    type Operand = ReduceOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the hypercube reduce needs the cube reduce, and one cube's stores reaching another within a launch";

    /// The cube microkernel's.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        CubeReduceMicrokernel::settings(hw)
    }

    fn candidates(
        hw: &Hardware,
        item: &ReduceWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        // The cubes meet through device memory, which must stay coherent across them in a launch.
        let meets = hw.properties.features.device_memory_scope;
        CubeReduceMicrokernel::candidates(hw, item, settings)
            .filter(move |_| meets)
            .map(|part| Self { part })
    }

    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<ReduceOperand>) {
        self.part.vectorize(&(), vectorization);
    }

    /// Each unit holds its carrier and one line; each cube a slot of the carrier a plane, and the
    /// merging unit the merged carrier and one partial.
    fn estimate(
        &self,
        _item: &ReduceWorkItem,
        _: &(),
        _planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        let slots = self.part.item.operation.written_per_row() as u32;
        estimate.work = Some(UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: 4 * slots + 2,
        });
    }
}
