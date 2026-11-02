use crate::rms_norm::plane::PlaneRmsNormMicrokernel;
use crate::rms_norm::{RmsNormOperand, RmsNormWorkItem};
use core::num::NonZeroUsize;
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// A row owned by one cube: its planes each run the plane microkernel over a run of the row, and
/// the planes' sums meet in shared memory, one slot a plane, which every unit then reads.
///
/// What a row too long for one plane's registers, or too few rows to fill the device a plane at a
/// time, wants: a decode step's single row of the model's width. It runs where its plane
/// microkernel does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeRmsNormMicrokernel {
    /// What each plane runs, over its run of the row.
    pub plane: PlaneRmsNormMicrokernel,
    /// Planes splitting the row: the settings' target snapped to a count that divides the row's
    /// lines at the width the operands bind at and leaves every lane a line.
    pub planes: usize,
}

/// How many planes a row is split between.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeRmsNormMicrokernelSettings {
    /// A target, not a demand ([`CubeRmsNormMicrokernel::planes`]).
    pub planes: NonZeroUsize,
}

impl CubeRmsNormMicrokernelSettings {
    pub const DEFAULT: Self = Self {
        planes: NonZeroUsize::new(4).unwrap(),
    };
}

impl core::fmt::Display for CubeRmsNormMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "planes={}", self.planes)
    }
}

impl CubeRmsNormMicrokernel {
    /// Bytes of shared memory the planes' sums meet in: one `f32` a plane.
    pub fn shared_memory(&self) -> usize {
        self.planes * size_of::<f32>()
    }
}

/// Its planes and what each runs.
impl core::fmt::Display for CubeRmsNormMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "planes={} {}", self.planes, self.plane)
    }
}

impl Microkernel for CubeRmsNormMicrokernel {
    type WorkItem = RmsNormWorkItem;
    type Settings = CubeRmsNormMicrokernelSettings;
    type Operand = RmsNormOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the cube runs where its plane microkernel does: a plane width the device commits to, and a run its lanes hold";

    /// Planes a row, its one knob: four, two, eight and sixteen, each snapped to what divides the
    /// row and leaves every lane a line.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        [4, 2, 8, 16]
            .map(|planes| Self::Settings {
                planes: NonZeroUsize::new(planes).unwrap(),
            })
            .into()
    }

    /// The most planes up to the target that divide the row's lines at the bound width, leave
    /// every lane at least one, and fit a cube, each running the plane microkernel over its run,
    /// where the device runs that. Under a packed store each run is whole blocks; the cube encodes
    /// its whole row at once, so a run need not be whole words of scales.
    fn candidates(
        hw: &Hardware,
        item: &RmsNormWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let (width, lines) = (item.width(hw), item.lines(hw));
        let lanes = hw.plane_dim().ok().map(|lanes| lanes as usize);
        lanes
            .and_then(|lanes| {
                let ceiling = settings
                    .planes
                    .get()
                    .min((lines / lanes).max(1))
                    .min((hw.properties.hardware.max_units_per_cube as usize / lanes).max(1));
                // A plane's run of a packed store's stage is whole blocks, which the stage states
                // the row in.
                let planes = (1..=ceiling).rev().find(|&planes| {
                    lines.is_multiple_of(planes)
                        && item.store.holds_owner_run(lines / planes * width)
                })?;
                let run = lines / planes * width;
                PlaneRmsNormMicrokernel::new(hw, item, run).map(|plane| Self { plane, planes })
            })
            .into_iter()
    }

    /// Its plane microkernel's, over a plane's run.
    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<RmsNormOperand>) {
        self.plane.vectorize(&(), vectorization);
    }

    /// Its plane microkernel's work, which every plane holds.
    fn estimate(
        &self,
        item: &RmsNormWorkItem,
        _: &(),
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        self.plane.estimate(item, &(), planes, estimate);
    }
}
