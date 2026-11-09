use crate::rms_norm::{HeldRowShape, RmsNormOperand, RmsNormWorkItem};
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// A run of a row owned by one plane: each lane holds its lines in registers, and their partials
/// sum across the plane in one instruction.
///
/// It runs where the device commits to a plane width: its lanes sum across the plane, so a
/// plane launched wider than the device runs it would sum part of the run. A plane owns a whole
/// row at this scope; a cube's planes each own a run of one ([`CubeRmsNormMicrokernel`]).
///
/// [`CubeRmsNormMicrokernel`]: crate::rms_norm::cube::CubeRmsNormMicrokernel
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneRmsNormMicrokernel {
    /// Units of the plane, as wide as the device commits to.
    ///
    /// An extent the kernel compiles, an exception to the crate's rule taken for the same reason
    /// the plane shuffle takes it: each lane's registers are sized by its share of the run, which is
    /// the run over the lanes.
    pub lanes: usize,
    /// Elements of the row the plane holds: the whole row, or a cube's plane's run of it. Like
    /// `dim`, an extent the kernel compiles, the exception
    /// [`RmsNormPartitioning::kernel`](crate::rms_norm::RmsNormPartitioning::kernel)
    /// argues: the lanes size their registers by it, and the autotune key holds the dim exactly.
    pub run: usize,
    /// The dim the whole row spans, which its scale divides by.
    pub dim: usize,
    /// The width the operands bind at ([`RmsNormWorkItem::width`]), which the run is checked
    /// against the lanes' registers at.
    pub width: usize,
}

/// A plane's microkernel has nothing a measurement settles: its lanes are the device's and its
/// run is the row's, or its cube's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneRmsNormMicrokernelSettings;

impl core::fmt::Display for PlaneRmsNormMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("whole-plane")
    }
}

impl PlaneRmsNormMicrokernel {
    /// The plane microkernel over `run` elements of `item`'s row on `hw`, where the device commits
    /// to a plane width and its lanes hold the run within their registers.
    pub fn new(hw: &Hardware, item: &RmsNormWorkItem, run: usize) -> Option<Self> {
        let microkernel = Self {
            lanes: hw.plane_dim().ok()? as usize,
            run,
            dim: item.dim,
            width: item.width(hw),
        };
        (microkernel.held_row().per_unit() <= HeldRowShape::MAX_LINES_PER_UNIT)
            .then_some(microkernel)
    }

    /// How a lane's share of the run lies at the width the operands bind at.
    pub fn held_row(&self) -> HeldRowShape {
        HeldRowShape {
            lines: self.run.div_ceil(self.width),
            units: self.lanes,
        }
    }
}

/// Its lanes and its run.
impl core::fmt::Display for PlaneRmsNormMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "lanes={} run={}", self.lanes, self.run)
    }
}

impl Microkernel for PlaneRmsNormMicrokernel {
    type WorkItem = RmsNormWorkItem;
    type Settings = PlaneRmsNormMicrokernelSettings;
    type Operand = RmsNormOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the plane microkernel needs a plane width the device commits to, and a row its lanes hold within `HeldRowShape::MAX_LINES_PER_UNIT` lines each";

    /// Its one setting: it has no knob.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![PlaneRmsNormMicrokernelSettings]
    }

    /// The plane over the whole row, where the device commits to a width and the row fits its
    /// lanes' registers.
    fn candidates(
        hw: &Hardware,
        item: &RmsNormWorkItem,
        _settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        Self::new(hw, item, item.dim).into_iter()
    }

    /// Every operand along the plane's run ([`RmsNormOperand::require_along_run`]).
    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<RmsNormOperand>) {
        RmsNormOperand::require_along_run(self.run, vectorization);
    }

    /// Its lanes hold their share of the run in registers ([`HeldRowShape::work`]). The reads are
    /// its routine's to count.
    fn estimate(
        &self,
        item: &RmsNormWorkItem,
        _: &(),
        _planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        estimate.work = Some(self.held_row().work(self.width, item.element));
    }
}
