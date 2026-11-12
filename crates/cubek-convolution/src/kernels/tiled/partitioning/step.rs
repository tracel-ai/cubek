use crate::kernels::tiled::{ConvProblem, ConvRoles};
use cubecl::tile::Axis;
use cubecl::tile::launch::Refusal;

/// What a cmma instruction is declined with where a step of the contraction holds several taps
/// beside the channels, which are fewer than the instruction's depth.
pub const CMMA_STEP_ACROSS_TAPS: Refusal = Refusal::Declined {
    why: "cmma steps through the channels alone, and a step of taps beside fewer channels than the instruction's depth is not one its fragments are loaded for",
};

/// What one step of the contraction, the instruction's `k` cells, is made of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvStep {
    /// The contracted axis alone: a whole step is `k` cells of it, the last one cut short.
    Alone { k: usize },
    /// A few cells of the contracted axis, rounded up to a power of two, and as many of the
    /// second contracted axis as fill the rest of the step.
    Beside { beside: usize, cells: usize },
}

impl ConvStep {
    pub fn new(problem: &ConvProblem, k: usize) -> Self {
        let cells = problem.extent(problem.roles().axes().k);
        match (k / cells.next_power_of_two()).max(1) {
            1 => Self::Alone { k },
            beside => Self::Beside {
                beside,
                cells: k / beside,
            },
        }
    }

    pub(super) fn beside(self) -> usize {
        match self {
            Self::Alone { .. } => 1,
            Self::Beside { beside, .. } => beside,
        }
    }

    /// The refusal a cmma instruction meets in this step: cells of a tap axis beside the channels.
    /// A step beside an output position axis is served.
    pub(super) fn cmma_declined(self, roles: &ConvRoles) -> Option<Refusal> {
        let beside_taps = ConvRoles::SPATIAL
            .iter()
            .any(|&[_, tap]| tap == roles.contracted.beside);
        (self.beside() > 1 && beside_taps).then_some(CMMA_STEP_ACROSS_TAPS)
    }

    pub(super) fn cells(self) -> usize {
        match self {
            Self::Alone { k } => k,
            Self::Beside { cells, .. } => cells,
        }
    }

    /// The axis a stage walks and the steps along it: the contracted axis alone, or the second
    /// one where it shares the step.
    pub(super) fn walk(self, roles: &ConvRoles, problem: &ConvProblem) -> (Axis, usize) {
        match self {
            Self::Alone { k } => {
                let axis = roles.axes().k;
                (axis, problem.extent(axis).div_ceil(k))
            }
            Self::Beside { beside, .. } => {
                let axis = roles.contracted.beside;
                (axis, problem.extent(axis).div_ceil(beside))
            }
        }
    }
}
