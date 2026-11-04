use crate::reduce::{ReduceOperand, ReduceWorkItem};
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// A row reduced by one cube: its units read consecutive lines, each accumulating every `units`-th into
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::plane::{PlaneReduceMicrokernel, PlaneReduceMicrokernelSettings};
    use crate::reduce::unit::{UnitReduceMicrokernel, UnitReduceMicrokernelSettings};
    use crate::reduce::{ReduceLayout, ReduceOperation};
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_std::devices;

    const F32: ElemType = ElemType::Float(FloatKind::F32);

    fn item() -> ReduceWorkItem {
        ReduceWorkItem {
            operation: ReduceOperation::ArgMax,
            input: F32,
            accumulator: F32,
            layout: ReduceLayout::Along,
            with_indices: false,
        }
    }

    /// The unit runs everywhere; the plane and the cube only where the device commits to a plane
    /// width, and decline with their reason where it does not.
    #[test]
    fn the_plane_and_the_cube_need_a_width_the_device_commits_to() {
        for hw in [
            devices::strix_halo(),
            devices::m2_pro(),
            devices::h100(),
            devices::wgpu_no_subgroups(),
            devices::rdna_range(),
        ] {
            let commits = hw.plane_dim().is_ok();
            assert!(
                UnitReduceMicrokernel::best(&hw, &item(), &UnitReduceMicrokernelSettings).is_ok()
            );
            assert_eq!(
                PlaneReduceMicrokernel::best(&hw, &item(), &PlaneReduceMicrokernelSettings).is_ok(),
                commits,
                "{hw:?}"
            );
            for settings in CubeReduceMicrokernel::settings(&hw) {
                let cube = CubeReduceMicrokernel::best(&hw, &item(), &settings);
                assert_eq!(cube.is_ok(), commits, "{settings} on {hw:?}");
                if let Ok(cube) = cube {
                    assert_eq!(cube.lanes as u32, hw.plane_dim().unwrap());
                }
            }
        }
    }

    /// A cube of more planes than the device holds declines.
    #[test]
    fn a_cube_the_device_cannot_hold_declines() {
        let hw = devices::m2_pro();
        let planes = hw.full_cube_planes() + 1;
        let settings = CubeReduceMicrokernelSettings { planes };
        assert!(CubeReduceMicrokernel::best(&hw, &item(), &settings).is_err());
    }
}
