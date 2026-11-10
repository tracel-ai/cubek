use crate::reduce::{ReduceOperand, ReduceWorkItem};
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// A row split across cubes, each reducing its part as the cube's microkernel reduces a row, then
/// storing its partial apart: the last of the row's cubes to finish merges the partials and writes
/// the row.
///
/// It runs where the device commits to a plane width, as the cube's does, and where one cube's
/// stores reach another within a launch (`device_memory_scope`), which is how the cubes meet: on
/// WGSL, whose storage barrier orders a workgroup's own memory only, the last cube was seen reading
/// another's partial as the zero it held before the write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeReduceMicrokernel {
    pub item: ReduceWorkItem,
    /// Units of a plane, as wide as the device commits to.
    pub lanes: usize,
    /// Planes of each cube.
    pub planes: usize,
}

/// How many planes each cube runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeReduceMicrokernelSettings {
    pub planes: usize,
}

impl HypercubeReduceMicrokernelSettings {
    /// The planes worth trying: cubes of four, then of eight.
    pub const ALL: [Self; 2] = [Self { planes: 4 }, Self { planes: 8 }];
}

impl core::fmt::Display for HypercubeReduceMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "planes={}", self.planes)
    }
}

impl core::fmt::Display for HypercubeReduceMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:?} {:?} lanes={} planes={}",
            self.item.operation, self.item.layout, self.lanes, self.planes
        )
    }
}

impl Microkernel for HypercubeReduceMicrokernel {
    type WorkItem = ReduceWorkItem;
    type Settings = HypercubeReduceMicrokernelSettings;
    type Operand = ReduceOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the hypercube reduce needs a plane width the device commits to, cubes that hold their planes, and one cube's stores reaching another within a launch";

    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        HypercubeReduceMicrokernelSettings::ALL.to_vec()
    }

    fn candidates(
        hw: &Hardware,
        item: &ReduceWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let planes = settings.planes;
        // The cubes meet through device memory, which must stay coherent across them in a launch.
        let fits = planes <= hw.full_cube_planes() && hw.properties.features.device_memory_scope;
        hw.plane_dim()
            .ok()
            .filter(move |_| fits)
            .into_iter()
            .map(move |lanes| Self {
                item: *item,
                lanes: lanes as usize,
                planes,
            })
    }

    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<ReduceOperand>) {
        self.item.require(vectorization);
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
        let slots = self.item.operation.written_per_row() as u32;
        estimate.work = Some(UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: 4 * slots + 2,
        });
    }
}
