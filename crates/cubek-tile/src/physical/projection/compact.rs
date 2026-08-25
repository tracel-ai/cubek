//! The smallest physical box a gathered operand's sub-tile can be staged in, and the mapping that
//! addresses it: what turns a stage from a copy of the *logical* tile into a copy of the *window*
//! the tile reads.

use cubecl::zspace::SmallVec;

use super::gcd;
use crate::{Axis, MAX_AXES, PhysicalAxisMap, Projection, Scale};

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
/// physical axis"). A [`Dynamic`](crate::Scale) one is exempt: pinning it would drop its slot from
/// the coefficient carrier the stage inherits from its source.
///
/// A `Dynamic` coefficient or divisor is sized by its bound rather than its value. The step goes to
/// `1` wherever one moves (a runtime coefficient need share no factor with anything, so there is no
/// lattice to quotient by), and the extent is the widest field the bound admits, which dominates
/// every window the launch can then ask for. The compacted mapping keeps the term dynamic: the box
/// is comptime, addressing it is not.
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
    /// always was. `vector_size` is the width the stage is served at, threaded through to
    /// [`Projection::validate`].
    pub fn of(
        projection: &Projection,
        vector_size: usize,
        extent_of: impl Fn(Axis) -> usize,
    ) -> Compaction {
        let rank = projection.physical_rank();
        let mut steps = SmallVec::new();
        let mut extents = SmallVec::new();
        let mut physical = Vec::with_capacity(rank);

        for pa in 0..rank {
            let axis_map = projection.physical_axis(pa);
            // The stage is placed at offset 0 (gmem windowing absorbs the offset), so a static
            // divisor can reduce away here even if the source operand carried a Dynamic offset.
            let scaled: Vec<(Axis, Scale)> =
                axis_map.terms().iter().map(|t| (t.axis, t.scale)).collect();
            let stage_map = PhysicalAxisMap::scaled(&scaled).over(axis_map.divisor());
            let terms = stage_map.terms();

            if stage_map.is_rational() {
                // A rational axis advances by one physical cell on some steps and none on others,
                // so its window has no single step to quotient by. Its step is 1 (dense, no holes
                // to skip), and its extent is the conservative receptive field over all possible
                // runtime phase residues: 1 + ⌊(Σ (extent - 1) * scale + divisor - 1) / divisor⌋.
                // Under a bound (a Dynamic coefficient or divisor) that field is the widest one the
                // bound admits, so the box still holds every window the launch can ask for.
                let d = stage_map.divisor().bound();
                let field: usize = terms
                    .iter()
                    .filter(|t| extent_of(t.axis) > 1)
                    .map(|t| (extent_of(t.axis) - 1) * t.scale.bound())
                    .sum();
                extents.push(1 + field.div_ceil(d));
                steps.push(1);
                physical.push(stage_map);
            } else {
                // Only a term that moves contributes an offset, so only its coefficient constrains the
                // step; a single-tap axis (extent 1) sits at a fixed offset the window's origin absorbs.
                //
                // A moving Dynamic coefficient has no comptime lattice to quotient by (the runtime
                // value need share no factor with anything), so any of those forces the dense step.
                let step = terms
                    .iter()
                    .filter(|t| extent_of(t.axis) > 1)
                    .try_fold(0, |g, t| match t.scale {
                        Scale::Static(s) => Some(gcd(g, s)),
                        Scale::Dynamic { .. } => None,
                    })
                    .unwrap_or(1);
                let step = step.max(1);
                // `step` divides every moving static coefficient by construction; a non-moving one
                // need not divide, and its value is unobservable, so it is pinned rather than
                // truncated. A Dynamic one passes through untouched: `step` is 1 wherever one
                // moves, and pinning a non-moving one would drop its slot from the coefficient
                // carrier, which the stage inherits verbatim from its source
                // ([`MemData::fill_from`](crate::MemData)) and must therefore index identically.
                let scaled: Vec<(Axis, Scale)> = terms
                    .iter()
                    .map(|t| {
                        let scale = match t.scale {
                            Scale::Static(s) if extent_of(t.axis) > 1 => Scale::Static(s / step),
                            Scale::Static(_) => Scale::Static(1),
                            dynamic => dynamic,
                        };
                        (t.axis, scale)
                    })
                    .collect();
                extents.push(
                    1 + scaled
                        .iter()
                        .map(|&(axis, scale)| (extent_of(axis) - 1) * scale.bound())
                        .sum::<usize>(),
                );
                steps.push(step);
                physical.push(PhysicalAxisMap::scaled(&scaled));
            }
        }

        let projection = Projection::new(projection.logical_axes(), &physical);
        // The compacted map is what the staged tile is addressed by, so it has to be as legal as
        // the operand's own: same axes, same innermost identity, still untiled.
        projection.validate(vector_size);

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
    /// line width whatever the outer axes do, and compaction pads the innermost axis out to whole
    /// lines.
    ///
    /// That rounding is only sound for a *padded* stage, one served wider than the source it is
    /// filled from. `fill_extent` is where the two boxes meet, so it is what refuses an
    /// extent that is not a whole number of source lines, on every path a fill can take.
    pub fn line_extents(&self, vector_size: usize) -> Vec<usize> {
        let last = self.extents.len() - 1;
        assert!(
            self.steps[last] == 1,
            "Compaction: the innermost physical axis is addressed in {vector_size}-wide lines, so \
             it must be an ungathered axis (extent {}, step {})",
            self.extents[last],
            self.steps[last]
        );
        let mut lines: Vec<usize> = self.extents.to_vec();
        lines[last] = lines[last].div_ceil(vector_size);
        lines
    }

    /// How many cells the stage allocates, in lines.
    pub fn cells(&self, vector_size: usize) -> usize {
        self.line_extents(vector_size).iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Divisor, Offset};

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

    /// [`extents`] for a projection with no `CI` axis at all, so there is no third extent to name.
    fn extents2(oh: usize, rh: usize) -> impl Fn(Axis) -> usize {
        move |a| match a {
            OH => oh,
            RH => rh,
            _ => unreachable!("axis not spanned by this projection"),
        }
    }

    /// A direct operand compacts to itself: the stage is the logical tile, as it has always been.
    #[test]
    fn direct_compacts_to_itself() {
        let p = Projection::direct(&[OH, CI]);
        let c = Compaction::of(&p, 4, extents(8, 1, 16));
        assert!(c.is_dense());
        assert_eq!(c.steps(), &[1, 1]);
        assert_eq!(c.extents(), &[8, 16]);
        assert_eq!(c.projection(), &p);
    }

    /// Unit stride and dilation: the window is the receptive field, `oh + rh - 1` against the
    /// `oh * rh` cells the logical stage held.
    #[test]
    fn a_unit_stride_window_is_the_receptive_field() {
        let c = Compaction::of(&conv(1, 1), 4, extents(8, 3, 16));
        assert!(c.is_dense());
        assert_eq!(c.extents(), &[10, 16]);
        assert_eq!(c.cells(4), 10 * 4);
    }

    /// Stride 2 with unit-dilation taps still reaches every offset (`gcd(2, 1) = 1`), so the box is
    /// dense and only the extent grows.
    #[test]
    fn a_strided_window_with_adjacent_taps_stays_dense() {
        let c = Compaction::of(&conv(2, 1), 4, extents(8, 3, 16));
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
            let c = Compaction::of(&conv(2, dilation), 4, extents(8, 1, 16));
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
        let c = Compaction::of(&conv(2, 2), 4, extents(8, 3, 16));
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
        let c = Compaction::of(&conv(3, 1), 4, extents(3, 2, 16));
        assert!(c.is_dense());
        assert_eq!(c.extents(), &[8, 16]);
    }

    /// A ragged innermost extent is padded out to whole lines.
    #[test]
    fn a_ragged_innermost_extent_is_padded() {
        let lines = Compaction::of(&conv(1, 1), 4, extents(8, 3, 6)).line_extents(4);
        assert_eq!(lines, &[10, 2]);
    }

    /// A 1-D resample gathers its only axis, with no passthrough axis behind it: illegal at any
    /// vectorized width, but at `vector_size == 1` there are no lines to protect, so the compacted
    /// window's own innermost axis may stay gathered too.
    #[test]
    fn a_scalar_gather_may_compact_its_only_axis() {
        let p = Projection::new(&[OH, RH], &[PhysicalAxisMap::affine(&[(OH, 2), (RH, 1)])]);
        let c = Compaction::of(&p, 1, extents2(8, 3));
        assert!(c.is_dense());
        assert_eq!(c.extents(), &[17]);
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
        let c = Compaction::of(&p, 4, extents(8, 3, 16));
        assert!(c.is_dense());
        assert_eq!(c.steps(), &[1, 1]);
        assert_eq!(c.extents(), &[17, 16]);
        assert_eq!(c.projection().offset(0), Offset::Static(0));
    }

    /// A dynamic offset shifts source placement without changing compacted extents.
    #[test]
    fn dynamic_offset_projection_compacts_identically() {
        let p = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::affine_with_offset(&[(OH, 2), (RH, 1)], Offset::Dynamic),
                PhysicalAxisMap::of(CI),
            ],
        );
        let c = Compaction::of(&p, 4, extents(8, 3, 16));
        assert!(c.is_dense());
        assert_eq!(c.steps(), &[1, 1]);
        assert_eq!(c.extents(), &[17, 16]);
        assert_eq!(c.projection().offset(0), Offset::Static(0));
    }

    /// A moving runtime coefficient has no comptime lattice to quotient by, so the box goes dense
    /// and is sized at the coefficient's `max`, matching what the same map spelled statically at
    /// that bound compacts to. The coefficient itself survives into the stage's own mapping: the
    /// box is bounded at comptime, but addressing it still needs the runtime value.
    #[test]
    fn a_dynamic_coefficient_compacts_dense_against_its_max() {
        let p = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::scaled(&[(OH, Scale::Dynamic { max: 2 }), (RH, Scale::Static(1))]),
                PhysicalAxisMap::of(CI),
            ],
        );
        let c = Compaction::of(&p, 4, extents(8, 3, 16));
        assert!(c.is_dense());
        // 1 + 7*2 + 2*1, the box `conv(2, 1)` fills.
        assert_eq!(c.extents(), &[17, 16]);
        assert_eq!(
            Compaction::of(&conv(2, 1), 4, extents(8, 3, 16)).extents(),
            c.extents()
        );
        assert!(c.projection().physical_axis(0).has_dynamic_scale());
        assert_eq!(c.projection().dynamic_scale_index(0, 0), Some(0));
    }

    /// A step the static coefficients would otherwise share is given up as soon as one of them
    /// moves at runtime: `gcd(2, 2)` is `2`, but a runtime coefficient need share no factor with
    /// anything, so the lattice cannot be assumed and the box stays dense.
    #[test]
    fn a_moving_dynamic_coefficient_gives_up_the_lattice() {
        let p = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::scaled(&[(OH, Scale::Static(2)), (RH, Scale::Dynamic { max: 2 })]),
                PhysicalAxisMap::of(CI),
            ],
        );
        let c = Compaction::of(&p, 4, extents(8, 3, 16));
        assert!(c.is_dense());
        assert_eq!(c.steps(), &[1, 1]);
        // 1 + 7*2 + 2*2, the bounding box `conv(2, 2)` quotients but this one cannot.
        assert_eq!(c.extents(), &[19, 16]);
    }

    /// A non-moving term is pinned to `1` only when it is static: pinning a `Dynamic` one would
    /// drop its slot from the coefficient carrier, which the stage inherits from its source
    /// position for position and so must index identically.
    #[test]
    fn a_non_moving_dynamic_coefficient_keeps_its_carrier_slot() {
        let p = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::scaled(&[(OH, Scale::Static(2)), (RH, Scale::Dynamic { max: 4 })]),
                PhysicalAxisMap::of(CI),
            ],
        );
        // RH does not move, so it contributes nothing to the box: 1 + 7*2 on the even lattice.
        let c = Compaction::of(&p, 4, extents(8, 1, 16));
        assert_eq!(c.steps(), &[2, 1]);
        assert_eq!(c.extents(), &[8, 16]);
        assert_eq!(p.dynamic_scale_index(0, 1), Some(0));
        assert_eq!(c.projection().dynamic_scale_index(0, 1), Some(0));
    }

    /// A rational axis steps one physical cell on some outputs and none on others, so its window
    /// has no lattice to quotient. It compacts densely (step = 1) with conservative extent.
    #[test]
    fn rational_axis_compacts_dense_with_conservative_extent() {
        let p = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::affine(&[(OH, 2), (RH, 3)]).over(3),
                PhysicalAxisMap::of(CI),
            ],
        );
        // OH: 8, RH: 3, CI: 16
        // field = (8-1)*2 + (3-1)*3 = 14 + 6 = 20
        // extent = 1 + (20 + 3 - 1) / 3 = 1 + 22 / 3 = 1 + 7 = 8
        let c = Compaction::of(&p, 4, extents(8, 3, 16));
        assert!(c.is_dense());
        assert_eq!(c.steps(), &[1, 1]);
        assert_eq!(c.extents(), &[8, 16]);
        assert!(c.projection().physical_axis(0).is_rational());
        assert_eq!(c.projection().divisor(0), Divisor::Static(3));
    }

    /// A dynamic divisor sizes the same box its `min` spelled statically would, and carries over
    /// into the stage's mapping bound and all, so the stage divides by the same runtime value.
    #[test]
    fn a_dynamic_divisor_compacts_against_its_min() {
        let p = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::affine(&[(OH, 2), (RH, 3)]).over(Divisor::Dynamic { min: 3 }),
                PhysicalAxisMap::of(CI),
            ],
        );
        let c = Compaction::of(&p, 4, extents(8, 3, 16));
        assert!(c.is_dense());
        // Identical to the static `over(3)` box above: field 20 over divisor 3 is 1 + 7 = 8.
        assert_eq!(c.extents(), &[8, 16]);
        assert_eq!(c.projection().divisor(0), Divisor::Dynamic { min: 3 });
        assert_eq!(c.projection().dynamic_divisor_index(0), Some(0));
    }

    /// The box a bound sizes dominates every divisor the launch may then pass, which is what makes
    /// staging one safe: a wider divisor reads a narrower window out of the same allocation.
    #[test]
    fn a_dynamic_divisor_box_holds_every_divisor_above_its_min() {
        let bounded = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::affine(&[(OH, 2), (RH, 3)]).over(Divisor::Dynamic { min: 2 }),
                PhysicalAxisMap::of(CI),
            ],
        );
        let box_extent = Compaction::of(&bounded, 4, extents(8, 3, 16)).extents()[0];
        for d in 2..24 {
            let exact = Projection::new(
                &[OH, RH, CI],
                &[
                    PhysicalAxisMap::affine(&[(OH, 2), (RH, 3)]).over(d),
                    PhysicalAxisMap::of(CI),
                ],
            );
            // `over` reduces a divisor the coefficients cancel, which is a different shape.
            if exact.physical_axis(0).is_rational() {
                assert!(Compaction::of(&exact, 4, extents(8, 3, 16)).extents()[0] <= box_extent);
            }
        }
    }

    /// A fraction whose coefficients cancel the divisor reduces on the stage (where offset is 0)
    /// even if the source operand carried a dynamic offset, recovering the integer gcd step.
    #[test]
    fn dynamic_offset_with_cancelling_divisor_compacts_as_integers() {
        let p_dynamic_offset = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::scaled_with_offset(
                    &[(OH, Scale::Static(16)), (RH, Scale::Static(8))],
                    Offset::Dynamic,
                )
                .over(4),
                PhysicalAxisMap::of(CI),
            ],
        );
        let p_integer = Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::affine(&[(OH, 4), (RH, 2)]),
                PhysicalAxisMap::of(CI),
            ],
        );
        let c_dynamic = Compaction::of(&p_dynamic_offset, 4, extents(8, 3, 16));
        let c_integer = Compaction::of(&p_integer, 4, extents(8, 3, 16));

        assert_eq!(c_dynamic.steps(), c_integer.steps());
        assert_eq!(c_dynamic.extents(), c_integer.extents());
        assert_eq!(c_dynamic.projection(), c_integer.projection());
        assert_eq!(c_dynamic.steps(), &[2, 1]);
    }
}
