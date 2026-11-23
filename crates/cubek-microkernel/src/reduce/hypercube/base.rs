use super::microkernel::hypercube_last_arrival_reduce_microkernel;
use crate::reduce::cube::{CubeStagedReduceMicrokernel, CubeStagedReduceMicrokernelSettings};
use crate::reduce::{ReduceOperand, ReduceWorkItem};
use cubecl::tile::Tile;
use cubecl::tile::launch::Vectorization;
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};
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
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeLastArrivalReduceMicrokernel {
    /// The cube microkernel each cube reduces its part with.
    #[cube(comptime)]
    part: CubeStagedReduceMicrokernel,
}

impl HypercubeLastArrivalReduceMicrokernel {
    /// A hypercube whose cubes each reduce their part with `part`.
    pub fn new(part: CubeStagedReduceMicrokernel) -> Self {
        Self { part }
    }

    /// The widest line its part's carrier keeps apart.
    ///
    /// # Panics
    ///
    /// As the unit microkernel's [`widest_line`](crate::reduce::unit::UnitDirectReduceMicrokernel::widest_line).
    pub fn widest_line(&self) -> Option<usize> {
        self.part.widest_line()
    }

    pub fn part(&self) -> CubeStagedReduceMicrokernel {
        self.part
    }
}

#[cube]
impl HypercubeLastArrivalReduceMicrokernel {
    /// One cube's part of a row of `input`, the row's last cube writing it to `out` and `indices` (`hypercube_last_arrival_reduce_microkernel`).
    ///
    /// # Panics
    ///
    /// Where its item writes the indices beside the values and `indices` is none, or writes
    /// indices its carrier does not keep; or where its carrier keeps indices and
    /// `partial_indices` is none.
    #[allow(
        clippy::too_many_arguments,
        reason = "the operands and the cube's place in its row, as the microkernel takes them"
    )]
    pub fn execute<In: Numeric, Acc: Numeric, Out: Numeric, Idx: Numeric, N: Size>(
        &self,
        input: &Tile,
        out: &mut Tile,
        indices: &ComptimeOption<Tile>,
        partials: &mut Tile,
        partial_indices: &ComptimeOption<Tile>,
        arrivals: &[Atomic<u32>],
        counter: usize,
        part: usize,
        parts: usize,
        first_line: usize,
        values: usize,
    ) {
        hypercube_last_arrival_reduce_microkernel::<In, Acc, Out, Idx, N>(
            input,
            out,
            indices,
            partials,
            partial_indices,
            arrivals,
            counter,
            part,
            parts,
            first_line,
            values,
            comptime!(HypercubeLastArrivalReduceMicrokernel { part: self.part }),
        )
    }
}

impl HypercubeLastArrivalReduceMicrokernel {
    /// Bytes of shared memory each cube allocates with its lines `width` lanes wide: its part's,
    /// and the slot its first unit hands the cube its arrival through.
    pub fn shared_memory_bytes_at_width(&self, width: usize) -> usize {
        self.part.shared_memory_bytes_at_width(width) + size_of::<u32>()
    }
}

impl core::fmt::Display for HypercubeLastArrivalReduceMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.part)
    }
}

impl Microkernel for HypercubeLastArrivalReduceMicrokernel {
    type WorkItem = ReduceWorkItem;
    type Settings = CubeStagedReduceMicrokernelSettings;
    type Operand = ReduceOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the hypercube reduce needs the cube reduce, and one cube's stores reaching another within a launch";

    /// The cube microkernel's.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        CubeStagedReduceMicrokernel::settings(hw)
    }

    fn candidates(
        hw: &Hardware,
        item: &ReduceWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        // The cubes meet through device memory, which must stay coherent across them in a launch.
        let meets = hw.properties.features.device_memory_scope;
        CubeStagedReduceMicrokernel::candidates(hw, item, settings)
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
        let slots = self.part.item().operation.written_per_row() as u32;
        estimate.work = Some(UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: 4 * slots + 2,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::cube::CubeStagedReduceMicrokernel;
    use crate::reduce::{ReduceOperation, ReduceWorkItem};
    use cubek_std::devices;

    /// A hypercube's cube holds its part's shared memory and one arrival slot beside it.
    #[test]
    fn a_cube_holds_its_parts_memory_and_an_arrival_slot() {
        const LANES: usize = 32;
        const PLANES: usize = 4;
        const LANE_SLOTS: usize = 32;
        const WIDTH: usize = 4;
        let item = ReduceWorkItem::along_f32(ReduceOperation::Sum);
        let part = CubeStagedReduceMicrokernel::new(item, LANES, PLANES, LANE_SLOTS);
        let hypercube = HypercubeLastArrivalReduceMicrokernel::new(part);
        assert_eq!(
            hypercube.shared_memory_bytes_at_width(WIDTH),
            part.shared_memory_bytes_at_width(WIDTH) + size_of::<u32>()
        );
    }

    /// A hypercube's widest line is its part's.
    #[test]
    fn its_widest_line_is_its_parts() {
        const LANES: usize = 32;
        const PLANES: usize = 4;
        const LANE_SLOTS: usize = 32;
        let item = ReduceWorkItem::along_f32(ReduceOperation::TopK { k: 4 });
        let part = CubeStagedReduceMicrokernel::new(item, LANES, PLANES, LANE_SLOTS);
        let hypercube = HypercubeLastArrivalReduceMicrokernel::new(part);
        assert_eq!(hypercube.widest_line(), part.widest_line());
    }

    /// Its cubes meet through device memory, so without device memory scope it has no candidate.
    #[test]
    fn it_needs_device_memory_scope() {
        let mut hw = devices::m2_pro();
        devices::features(&mut hw).device_memory_scope = false;
        let item = ReduceWorkItem::along_f32(ReduceOperation::Sum);
        let settings = HypercubeLastArrivalReduceMicrokernel::settings(&hw)[0];
        assert!(HypercubeLastArrivalReduceMicrokernel::best(&hw, &item, &settings).is_err());
    }
}
