//! The smallest physical box a gathered operand's sub-tile can be staged in, and the mapping that
//! addresses it: what turns a stage from a copy of the *logical* tile into a copy of the *window*
//! the tile reads.

use cubecl::zspace::SmallVec;

use crate::{Axis, MAX_AXES, PhysicalAxisMap, Projection};

/// The compacted stage of a [`Projection`]: per physical axis, how many cells the stage holds and
/// what step in the source one of its cells is, plus the projection the stage is addressed by.
///
/// A gathered operand's sub-tile reads a *window* of its buffer, and several logical cells read the
/// same physical one (that is what a gather is). Staging the logical tile therefore replicates
/// elements, roughly by the tap count. Staging the window instead holds each element once, and the
/// gather stays where it already was, at the leaf's read through
/// [`AxisProjection`](crate::AxisProjection).
///
/// The window on physical axis `pa` is the set of offsets `Σ aᵢ·sᵢ` its terms reach, with
/// `0 ≤ aᵢ < eᵢ`. Two numbers describe it:
///
/// - its **step** `g = gcd{ sᵢ : eᵢ > 1 }`, since every reachable offset is a multiple of `g`, so
///   the stage stores every `g`-th one and drops the rest;
/// - its **extent** `1 + Σ (eᵢ - 1)·(sᵢ/g)`, the bounding box of what is left.
///
/// Offsets the terms cannot reach but the box still spans (taps `{0, 1}` at stride `3` reach
/// `{0, 1, 3, 4, …}`, whose step is `1`) stay as padding: they are filled and never read. Sizing
/// them out exactly is a numeric-semigroup problem, and the step already covers what a stride or a
/// dilation produces.
///
/// So the compacted box is not always smaller than the logical tile it replaces. It wins when the
/// taps outrun the stride and the windows overlap (the usual convolution), and loses when the
/// stride outruns them: `3` output steps of `2` taps at stride `3` are `6` logical cells over a
/// box of `8`, two of which are padding. A gathered stage is sized by its window either way, so
/// that case costs both the extra smem and the gmem reads that fill it.
///
/// A term whose axis does not move (`eᵢ = 1`) sits at a fixed offset the window's origin absorbs,
/// so its coefficient is unobservable: the only coordinate it is ever multiplied by is `0`. It is
/// emitted as `1` rather than as `sᵢ/g`, which need not divide, so the compacted projection still
/// satisfies [`Projection::validate`] (a `0` coefficient would read as "this axis addresses no
/// physical axis").
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Compaction {
    steps: SmallVec<[usize; MAX_AXES]>,
    extents: SmallVec<[usize; MAX_AXES]>,
    projection: Projection,
}

impl Compaction {
    /// Compact `projection`'s window, `extent_of` giving each logical axis's extent over the
    /// sub-tile being staged. A [`direct`](Projection::direct) projection compacts to itself: every
    /// step is `1` and every extent is the axis's own, so a plain operand's stage is the tile it
    /// always was.
    pub fn of(projection: &Projection, extent_of: impl Fn(Axis) -> usize) -> Compaction {
        let rank = projection.physical_rank();
        let mut steps = SmallVec::new();
        let mut extents = SmallVec::new();
        let mut physical = Vec::with_capacity(rank);

        for pa in 0..rank {
            let terms = projection.physical_axis(pa).terms();
            // Only a term that moves contributes an offset, so only its coefficient constrains the
            // step; a single-tap axis (extent 1) sits at a fixed offset the window's origin absorbs.
            let step = terms
                .iter()
                .filter(|t| extent_of(t.axis) > 1)
                .fold(0, |g, t| gcd(g, t.scale.get()));
            let step = step.max(1);
            // `step` divides every moving coefficient by construction; a non-moving one need not
            // divide, and its value is unobservable, so it is pinned rather than truncated.
            let scaled: Vec<(Axis, usize)> = terms
                .iter()
                .map(|t| {
                    let scale = if extent_of(t.axis) > 1 {
                        t.scale.get() / step
                    } else {
                        1
                    };
                    (t.axis, scale)
                })
                .collect();
            extents.push(
                1 + scaled
                    .iter()
                    .map(|&(axis, scale)| (extent_of(axis) - 1) * scale)
                    .sum::<usize>(),
            );
            steps.push(step);
            physical.push(PhysicalAxisMap::affine(&scaled));
        }

        let projection = Projection::new(projection.logical_axes(), &physical);
        // The compacted map is what the staged tile is addressed by, so it has to be as legal as
        // the operand's own: same axes, same innermost identity, still untiled.
        projection.validate();

        Compaction {
            steps,
            extents,
            projection,
        }
    }

    /// How the stage's own logical axes address its cells: `projection` with every moving
    /// coefficient divided by its axis's step. This is what the staged tile carries as its
    /// [`projection`](crate::MemData), so its reads and its [`at`](crate::Tile::at) descents run
    /// through the same machinery a gmem operand's do.
    pub fn projection(&self) -> &Projection {
        &self.projection
    }

    /// The step per physical axis: what a stage coordinate is multiplied by to land on the source
    /// coordinate it was filled from. All `1` exactly when [`is_dense`](Compaction::is_dense).
    pub fn steps(&self) -> &[usize] {
        &self.steps
    }

    /// The cell count per physical axis, in elements.
    pub fn extents(&self) -> &[usize] {
        &self.extents
    }

    /// Whether the window has no holes to skip, so a fill reads the source box straight through and
    /// emits exactly what a direct operand's fill does. True for every direct operand, and for a
    /// gather whose taps land on consecutive offsets (a unit-dilation convolution, at any stride).
    pub fn is_dense(&self) -> bool {
        self.steps.iter().all(|&g| g == 1)
    }

    /// [`extents`](Compaction::extents) with the innermost in `vector_size`-wide lines, the unit
    /// every physical shape in this crate is counted in.
    ///
    /// The innermost physical axis is one logical axis at coefficient `1`
    /// ([`Projection::validate`]), so it is never gathered: its step is `1`, its window has no
    /// holes, and its extent is the logical edge itself. A stage therefore keeps the store's full
    /// line width whatever the outer axes do, and compaction only ever removes whole lines.
    pub fn line_extents(&self, vector_size: usize) -> Vec<usize> {
        let last = self.extents.len() - 1;
        assert!(
            self.steps[last] == 1 && self.extents[last].is_multiple_of(vector_size),
            "Compaction: the innermost physical axis is addressed in {vector_size}-wide lines, so \
             it must be an ungathered multiple of that width (extent {}, step {})",
            self.extents[last],
            self.steps[last]
        );
        let mut lines: Vec<usize> = self.extents.to_vec();
        lines[last] /= vector_size;
        lines
    }

    /// How many cells the stage allocates, in lines.
    pub fn cells(&self, vector_size: usize) -> usize {
        self.line_extents(vector_size).iter().product()
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OH: Axis = Axis(0);
    const RH: Axis = Axis(1);
    const CI: Axis = Axis(2);

    /// `Ih <- Oh*stride + Rh*dilation`, `Ci` passed through: the shape every convolution input
    /// carries.
    fn conv(stride: usize, dilation: usize) -> Projection {
        Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::affine(&[(OH, stride), (RH, dilation)]),
                PhysicalAxisMap::of(CI),
            ],
        )
    }

    fn extents(oh: usize, rh: usize, ci: usize) -> impl Fn(Axis) -> usize {
        move |a| match a {
            OH => oh,
            RH => rh,
            _ => ci,
        }
    }

    /// A direct operand compacts to itself: the stage is the logical tile, as it has always been.
    #[test]
    fn direct_compacts_to_itself() {
        let p = Projection::direct(&[OH, CI]);
        let c = Compaction::of(&p, extents(8, 1, 16));
        assert!(c.is_dense());
        assert_eq!(c.steps(), &[1, 1]);
        assert_eq!(c.extents(), &[8, 16]);
        assert_eq!(c.projection(), &p);
    }

    /// Unit stride and dilation: the window is the receptive field, `oh + rh - 1` against the
    /// `oh * rh` cells the logical stage held.
    #[test]
    fn a_unit_stride_window_is_the_receptive_field() {
        let c = Compaction::of(&conv(1, 1), extents(8, 3, 16));
        assert!(c.is_dense());
        assert_eq!(c.extents(), &[10, 16]);
        assert_eq!(c.cells(4), 10 * 4);
    }

    /// Stride 2 with unit-dilation taps still reaches every offset (`gcd(2, 1) = 1`), so the box is
    /// dense and only the extent grows.
    #[test]
    fn a_strided_window_with_adjacent_taps_stays_dense() {
        let c = Compaction::of(&conv(2, 1), extents(8, 3, 16));
        assert!(c.is_dense());
        // 1 + 7*2 + 2*1
        assert_eq!(c.extents(), &[17, 16]);
    }

    /// A single tap at stride 2 reaches only the even offsets, so the stage keeps every second row
    /// and its projection loses the stride. `Rh` does not move, so its coefficient is the pinned
    /// `1` whatever the dilation was: the stage's `Rh` coordinate is always `0`.
    #[test]
    fn a_single_tap_at_stride_two_halves_the_stage() {
        for dilation in [1, 3] {
            let c = Compaction::of(&conv(2, dilation), extents(8, 1, 16));
            assert!(!c.is_dense());
            assert_eq!(c.steps(), &[2, 1]);
            assert_eq!(c.extents(), &[8, 16]);
            assert_eq!(c.projection().scale(0, OH), 1);
            assert_eq!(c.projection().scale(0, RH), 1);
        }
    }

    /// Stride and dilation sharing a factor: the whole window is on the even lattice, so the stage
    /// stores half of the bounding box and both coefficients halve.
    #[test]
    fn a_shared_factor_quotients_the_window() {
        let c = Compaction::of(&conv(2, 2), extents(8, 3, 16));
        assert_eq!(c.steps(), &[2, 1]);
        // Bounding box 1 + 7*2 + 2*2 = 19, on the even lattice: 1 + 7 + 2 = 10.
        assert_eq!(c.extents(), &[10, 16]);
        assert_eq!(c.projection().scale(0, OH), 1);
        assert_eq!(c.projection().scale(0, RH), 1);
    }

    /// Taps the step cannot describe stay as padding: `{0, 1}` at stride 3 reaches
    /// `{0, 1, 3, 4, 6, 7}`, whose gcd is 1, so the box spans the holes.
    #[test]
    fn unreachable_offsets_inside_the_box_stay_as_padding() {
        let c = Compaction::of(&conv(3, 1), extents(3, 2, 16));
        assert!(c.is_dense());
        assert_eq!(c.extents(), &[8, 16]);
    }

    /// The innermost axis is addressed in lines, so its extent must divide by the store width.
    #[test]
    #[should_panic(expected = "addressed in 4-wide lines")]
    fn a_ragged_innermost_extent_is_refused() {
        Compaction::of(&conv(1, 1), extents(8, 3, 6)).line_extents(4);
    }

    /// A constant offset shifts source placement without changing compacted extents.
    #[test]
    fn padded_projection_compacts_identically() {
        let p = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::affine_with_offset(&[(OH, 2), (RH, 1)], -2),
                PhysicalAxisMap::of(CI),
            ],
        );
        let c = Compaction::of(&p, extents(8, 3, 16));
        assert!(c.is_dense());
        assert_eq!(c.steps(), &[1, 1]);
        assert_eq!(c.extents(), &[17, 16]);
        assert_eq!(c.projection().offset(0), 0);
    }
}
