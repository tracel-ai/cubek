//! Where the [`Arg`](super::Arg) builder lands the bounds-check ([`Boundaries`]).

use cubecl::zspace::SmallVec;

use super::boundary_policy::BoundaryPolicy;
use crate::{Axis, Boundary, Projection, Refusal, Space};

/// Where the bounds-check lands: on the coordinate axes that can leave the buffer, only those.
pub(crate) struct Boundaries {
    /// One mode per coordinate axis; empty when nothing is checked.
    pub(crate) modes: SmallVec<[Option<Boundary>; Space::MAX_RANK]>,
}

impl Boundaries {
    /// The mode: the stated policy, else whether the operand overhangs or underflows.
    fn mode(
        policy: Option<BoundaryPolicy>,
        projection: &Projection,
        addressed: &[Axis],
        concrete: &Space,
        overhangs: &[Axis],
    ) -> Option<Boundary> {
        match policy {
            Some(BoundaryPolicy::Unchecked) => None,
            Some(BoundaryPolicy::Every(boundary)) => Some(boundary),
            None => {
                let overhangs = addressed
                    .iter()
                    .filter(|&&axis| concrete.contains(axis))
                    .any(|axis| overhangs.contains(axis));
                (overhangs || projection.may_underflow()).then_some(Boundary::Zero)
            }
        }
    }

    /// `concrete` is the launch's real-extent space and `overhangs` the axes its tiles reach past.
    pub(crate) fn new(
        policy: Option<BoundaryPolicy>,
        projection: &Projection,
        addressed: &[Axis],
        concrete: &Space,
        overhangs: &[Axis],
        width: usize,
    ) -> Result<Self, Refusal> {
        let boundary = Self::mode(policy, projection, addressed, concrete, overhangs);
        let coords = projection.untiled();
        let coord_rank = projection.coordinate_rank();
        if coord_rank == 0 {
            return Err(Refusal::NoCoordinate);
        }

        // Whether coordinate axis `pa` is in bounds by construction; only an identity map can be.
        let settled = |pa: usize| match coords.physical_axis(pa).identity_axis() {
            Some(axis) => concrete.contains(axis) && !overhangs.contains(&axis),
            None => false,
        };

        if boundary.is_some() && width > 1 && !settled(coord_rank - 1) {
            return Err(Refusal::UncheckableVectorEdge);
        }

        let modes: SmallVec<[Option<Boundary>; Space::MAX_RANK]> = (0..coord_rank)
            .map(|pa| boundary.filter(|_| !settled(pa)))
            .collect();
        // "Nothing is checked" has one form: the empty list.
        Ok(match modes.iter().any(Option::is_some) {
            true => Boundaries { modes },
            false => Boundaries {
                modes: SmallVec::new(),
            },
        })
    }
}
