use crate::matmul::dense::cube::{CubeStagedDelivery, CubeStagedFill};
use core::num::NonZeroUsize;
use cubecl::tile::launch::Refusal;

/// Why a cube does not run a warpgroup MMA its plane microkernel elected: what only the cube's
/// fill and fold decide, beside the device and product rules the plane microkernel checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeStagedWgmmaRefusal {
    /// Its stages arrive by anything but the TMA engine, which alone lays the swizzled rows its
    /// descriptors read.
    Unlanded,
    /// No plane is set aside to fill its stages: the walk's slots would be released as each
    /// consume returns rather than once the MMAs reading them are done.
    NoFillingPlanes,
    /// Its sum is folded, along `K` or through a bounce, where it drains straight into its output.
    Folded,
}

impl CubeStagedWgmmaRefusal {
    /// Why a cube filled as `fill`, its planes split `planes_along_k` ways along `K` and its sum
    /// folded where `folds`, cannot run a warpgroup MMA; `None` where it can.
    pub fn new(fill: &CubeStagedFill, planes_along_k: NonZeroUsize, folds: bool) -> Option<Self> {
        if fill.delivery != CubeStagedDelivery::Tma {
            Some(Self::Unlanded)
        } else if fill.filling_planes == 0 {
            Some(Self::NoFillingPlanes)
        } else if planes_along_k.get() > 1 || folds {
            Some(Self::Folded)
        } else {
            None
        }
    }
}

impl From<CubeStagedWgmmaRefusal> for Refusal {
    fn from(refusal: CubeStagedWgmmaRefusal) -> Self {
        let details = match refusal {
            CubeStagedWgmmaRefusal::Unlanded => {
                "a warpgroup MMA reads its stages as the TMA engine lands them"
            }
            CubeStagedWgmmaRefusal::NoFillingPlanes => {
                "a warpgroup MMA releases its stages once its MMAs are done, which needs planes \
                 of their own to fill them"
            }
            CubeStagedWgmmaRefusal::Folded => {
                "a warpgroup MMA's sum drains straight into its output, which no fold reaches"
            }
        };
        Refusal::Declined { why: details }
    }
}
