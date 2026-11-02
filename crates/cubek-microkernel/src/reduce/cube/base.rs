use crate::reduce::{ReduceOperand, ReduceWorkItem};
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// A row reduced by one cube: its units read consecutive lines, each folding every `units`-th into
/// a partial of its own; each plane's partials meet in its instructions, then the planes' in
/// shared memory, one slot of the carrier a plane.
///
/// It runs where the device commits to a plane width, as the plane's does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeReduceMicrokernel {
    pub item: ReduceWorkItem,
    /// Units of a plane, as wide as the device commits to.
    pub lanes: usize,
    /// Planes of the cube.
    pub planes: usize,
}

/// How many planes a cube runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeReduceMicrokernelSettings {
    pub planes: usize,
}

impl CubeReduceMicrokernelSettings {
    /// The planes worth trying: a cube of four, then of eight.
    pub const ALL: [Self; 2] = [Self { planes: 4 }, Self { planes: 8 }];
}

impl core::fmt::Display for CubeReduceMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "planes={}", self.planes)
    }
}

impl core::fmt::Display for CubeReduceMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:?} {:?} lanes={} planes={}",
            self.item.operation, self.item.layout, self.lanes, self.planes
        )
    }
}

impl Microkernel for CubeReduceMicrokernel {
    type WorkItem = ReduceWorkItem;
    type Settings = CubeReduceMicrokernelSettings;
    type Operand = ReduceOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the cube reduce needs a plane width the device commits to, and a cube that holds its planes";

    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        CubeReduceMicrokernelSettings::ALL.to_vec()
    }

    fn candidates(
        hw: &Hardware,
        item: &ReduceWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let planes = settings.planes;
        let fits = planes <= hw.full_cube_planes();
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

    /// Each unit holds its carrier and one line; the cube a slot of the carrier a plane.
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
            registers: 2 * slots + 2,
        });
    }
}
