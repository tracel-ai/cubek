//! The in-kernel evaluation of a [`Projection`]: [`ProjectionInKernel`] turns a tile's logical
//! coordinate into the physical one its window is boxed in, and [`CompactionStep`] undoes the
//! lattice a [`Compaction`] quotients a gathered window by.

use cubecl::{
    prelude::*,
    std::tensor::layout::{CoordsDyn, Layout, LayoutExpand},
};

use crate::*;

/// A [`Layout`] mapping a tile's logical coordinate to its window's physical one:
/// `phys[pa] = (Σ logical[axis] * scale + residue) / divisor`. Folds axes and nothing else, so the
/// window's own [`Boundary`](crate::Boundary) still governs what an out-of-range tap reads.
///
/// Static terms a static divisor divides exactly leave the numerator before the floor
/// ([`static_offset_step`](crate::PhysicalAxisMap)), so a resampling map's taps advance by a static
/// physical step under the one necessary divide.
///
/// Constant offsets belong to the [`Window`](crate::Window) and are omitted here, all but the
/// phase a division starts the floor at, which the [`RuntimeMap`](crate::RuntimeMap) carries.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct ProjectionInKernel {
    /// The tile's per-logical-axis extents, in the space's axis order. The innermost is a
    /// line count, matching the window's innermost physical axis.
    shape: Coords<u32>,
    /// The projection's runtime half: the coefficients a tap folds through, and the phase its
    /// window origin left over. The constant offsets are not among them: a tap is relative to the
    /// window, which they placed.
    map: RuntimeMap,
    #[cube(comptime)]
    space: Space,
    #[cube(comptime)]
    projection: Projection,
    /// The line width the innermost physical axis is addressed in.
    #[cube(comptime)]
    width: usize,
}

#[cube]
impl ProjectionInKernel {
    pub fn new(
        shape: Coords<u32>,
        map: RuntimeMap,
        #[comptime] space: Space,
        #[comptime] projection: Projection,
        #[comptime] width: usize,
    ) -> Self {
        let rank = shape.len();
        comptime!(assert!(
            rank == space.rank(),
            "ProjectionInKernel: shape has {rank} entries but the space spans {} logical axes",
            space.rank()
        ));
        let given = map.coefficients.len();
        comptime!(assert!(
            given == projection.dynamic_coefficient_count(),
            "ProjectionInKernel: the projection has {} Dynamic coefficients and divisors but {given} \
             were given",
            projection.dynamic_coefficient_count()
        ));
        let phases = map.residues.len();
        comptime!(assert!(
            phases == projection.physical_rank(),
            "ProjectionInKernel: the projection has {} physical axes but {phases} residues were given",
            projection.physical_rank()
        ));
        ProjectionInKernel {
            shape,
            map,
            space,
            projection,
            width,
        }
    }
}

/// A term's coefficient in the units its physical axis is addressed in: scalars for every axis but
/// the innermost, addressed in *lines*, whose coefficients divide by the width. A coefficient the
/// width does not divide is refused: the axes above the line change within it; no read serves that.
fn line_scale(
    space: &Space,
    projection: &Projection,
    width: usize,
    pa: usize,
    axis: Axis,
    scale: usize,
) -> usize {
    let innermost = space.axis_at(space.rank() - 1);
    if pa != projection.physical_rank() - 1 || width == 1 || axis == innermost {
        return scale;
    }
    assert!(
        scale.is_multiple_of(width),
        "ProjectionInKernel: {axis:?} steps {scale} values of a physical axis read {width} at a time, \
         so one line spans more than one {axis:?}; serve narrower lines, or give {axis:?} a \
         physical axis of its own"
    );
    scale / width
}

/// The static physical step a term contributes once taken out of its axis's evaluation: under a
/// floor only what the divisor factors out, elsewhere the coefficient itself, `None` for a dynamic
/// coefficient. Panics for a term a rational axis keeps inside its floor: it is not additive there.
fn split_step(map: &PhysicalAxisMap, term: usize) -> Option<usize> {
    if map.is_rational() {
        return Some(map.static_offset_step(term).unwrap_or_else(|| {
            panic!(
                "ProjectionInKernel::advance: {:?} stays inside this axis's floor, so stepping it is \
                 not an addition and the map has to be folded at every position",
                map.terms()[term].axis
            )
        }));
    }
    match map.terms()[term].scale {
        Scale::Static(s) => Some(s),
        Scale::Dynamic { .. } => None,
    }
}

#[cube]
impl ProjectionInKernel {
    /// The source coordinate of `pos` with every axis in `moving` held at zero: the part of the
    /// map a walk over those axes leaves alone, which [`advance`](Self::advance) puts back. This
    /// gives a gather one floor per accumulator cell rather than per tap on its rational axes.
    pub fn anchor(&self, pos: CoordsDyn, #[comptime] moving: Vec<Axis>) -> CoordsDyn {
        let mut out = CoordsDyn::new();

        #[unroll]
        for pa in 0..comptime!(self.projection.physical_rank()) {
            out.push(self.project_axis(&pos, pa, comptime!(moving.clone())));
        }

        out
    }

    /// [`anchor`](Self::anchor) for one physical axis. Factor-local boundary normalization uses
    /// this narrow form so checking one tap does not rebuild unrelated source coordinates.
    pub(crate) fn project_axis(
        &self,
        pos: &CoordsDyn,
        #[comptime] pa: usize,
        #[comptime] moving: Vec<Axis>,
    ) -> u32 {
        let axis_map = comptime!(self.projection.physical_axis(pa));
        let n = comptime!(axis_map.terms().len());

        // Per-term products left in the numerator, summed below (chained, so a single
        // coefficient-1 term folds to the coordinate itself). Under a division the sum starts at
        // the phase the window origin could not absorb.
        let mut terms = Coords::<u32>::new();
        if comptime!(axis_map.is_rational()) {
            terms.push(self.map.residues.at(pa));
        }
        // Exact steps stay outside the numerator so a rational projection takes one spatial floor
        // and adds the tap step to it.
        let mut offsets = Coords::<u32>::new();
        #[unroll]
        for t in 0..n {
            let term = comptime!(axis_map.terms()[t]);
            if comptime!(!moving.contains(&term.axis)) {
                let p = comptime!(self.space.position(term.axis));
                match comptime!(axis_map.static_offset_step(t)) {
                    Some(step) => offsets.push(pos[p].times(comptime!(step as u32))),
                    None => match comptime!(term.scale) {
                        Scale::Static(s) => terms.push(pos[p].times(comptime!(line_scale(
                            &self.space,
                            &self.projection,
                            self.width,
                            pa,
                            term.axis,
                            s
                        )
                            as u32))),
                        Scale::Dynamic { .. } => {
                            terms.push(pos[p].times(self.map.coefficients.at(comptime!(
                                self.projection.dynamic_scale_index(pa, t).unwrap()
                            ))))
                        }
                    },
                }
            }
        }
        let n_kept = terms.len();
        let n_exact = offsets.len();
        let sum = terms.sum(comptime!((0..n_kept).collect::<Vec<_>>()));

        if comptime!(axis_map.is_rational()) {
            match comptime!(axis_map.divisor()) {
                Divisor::Static(d) => {
                    let offset = offsets.sum(comptime!((0..n_exact).collect::<Vec<_>>()));
                    sum.divided_by(comptime!(d as u32)).plus(offset)
                }
                Divisor::Dynamic { .. } => {
                    let divisor = self.map.coefficients.at(comptime!(
                        self.projection.dynamic_divisor_index(pa).unwrap()
                    ));
                    sum.divided_by(divisor)
                }
            }
        } else {
            sum
        }
    }

    /// `anchor` moved to where `pos` places the `moving` axes, which must be the ones it was
    /// [anchored](Self::anchor) against. Every one enters linearly, so the move is an exact add:
    /// the term's own coefficient outside a division, the divisor's static step under one.
    pub fn advance(
        &self,
        anchor: &CoordsDyn,
        pos: CoordsDyn,
        #[comptime] moving: Vec<Axis>,
    ) -> CoordsDyn {
        let mut out = CoordsDyn::new();

        #[unroll]
        for pa in 0..comptime!(self.projection.physical_rank()) {
            let axis_map = comptime!(self.projection.physical_axis(pa));
            let n = comptime!(axis_map.terms().len());

            let mut steps = Coords::<u32>::new();
            steps.push(anchor[pa]);
            #[unroll]
            for t in 0..n {
                let term = comptime!(axis_map.terms()[t]);
                if comptime!(moving.contains(&term.axis)) {
                    let p = comptime!(self.space.position(term.axis));
                    match comptime!(split_step(axis_map, t)) {
                        // In the units this physical axis is read at, like the fold `anchor` did:
                        // the two are added together, so they cannot be counted differently.
                        Some(step) => steps.push(pos[p].times(comptime!(line_scale(
                            &self.space,
                            &self.projection,
                            self.width,
                            pa,
                            term.axis,
                            step
                        )
                            as u32))),
                        None => steps.push(pos[p].times(self.map.coefficients.at(comptime!(
                            self.projection.dynamic_scale_index(pa, t).unwrap()
                        )))),
                    }
                }
            }
            let n_steps = steps.len();
            out.push(steps.sum(comptime!((0..n_steps).collect::<Vec<_>>())));
        }

        out
    }
}

#[cube]
impl Layout for ProjectionInKernel {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        self.anchor(pos, comptime!(Vec::new()))
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.shape.to_dyn()
    }

    /// The logical box. Whether the physical coordinate it maps to is within the operand's valid
    /// data is the [`Window`](crate::Window)'s question, asked one layer down.
    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        self.shape.within(pos)
    }
}

/// A [`Layout`] scaling a *physical* coordinate by one step per axis: `src[pa] = pos[pa] * step`.
/// The inverse of the lattice a [`Compaction`] quotients a gathered operand's window by, so a fill
/// walking the compacted stage lands on the source cells it keeps. Not built for a dense window.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct CompactionStep {
    /// The compacted extents this steps through, innermost a line count.
    shape: Coords<u32>,
    #[cube(comptime)]
    steps: Vec<usize>,
}

#[cube]
impl CompactionStep {
    pub fn new(shape: Coords<u32>, #[comptime] steps: Vec<usize>) -> Self {
        let rank = shape.len();
        comptime!(assert!(
            rank == steps.len(),
            "CompactionStep: shape has {rank} entries but {} steps were given",
            steps.len()
        ));
        CompactionStep { shape, steps }
    }
}

#[cube]
impl Layout for CompactionStep {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let mut out = CoordsDyn::new();

        #[unroll]
        for pa in 0..comptime!(self.steps.len()) {
            out.push(pos[pa].times(comptime!(self.steps[pa] as u32)));
        }

        out
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.shape.to_dyn()
    }

    /// The compacted box, the source cell it steps up to being the [`Window`](crate::Window)'s
    /// question.
    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        self.shape.within(pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OUT: Axis = Axis(0);
    const TAP: Axis = Axis(1);

    /// Outside a floor a term steps by its own coefficient, which is what makes `advance` an
    /// addition.
    #[test]
    fn a_plain_affine_term_steps_by_its_coefficient() {
        let map = PhysicalAxisMap::affine(&[(OUT, 2), (TAP, 3)]);
        assert_eq!(split_step(&map, 0), Some(2));
        assert_eq!(split_step(&map, 1), Some(3));
    }

    /// A dynamic coefficient is a runtime read, not a static step.
    #[test]
    fn a_dynamic_coefficient_has_no_static_step() {
        let map =
            PhysicalAxisMap::scaled(&[(OUT, Scale::Static(2)), (TAP, Scale::Dynamic { max: 4 })]);
        assert_eq!(split_step(&map, 1), None);
    }

    /// Under a floor a term steps by what the divisor factors out: `⌊(x + m·4)/2⌋` moves by `2`
    /// per `m`, exactly.
    #[test]
    fn a_divisible_term_steps_by_what_the_floor_factors_out() {
        let map = PhysicalAxisMap::affine(&[(OUT, 3), (TAP, 4)]).over(2);
        assert_eq!(split_step(&map, 1), Some(2));
    }

    /// The same map's indivisible term: `⌊(3·out + …)/2⌋` is not `out` times anything, so it has
    /// to stay anchored rather than be stepped.
    #[test]
    #[should_panic(expected = "stays inside this axis's floor")]
    fn an_indivisible_term_under_a_floor_cannot_be_stepped() {
        let map = PhysicalAxisMap::affine(&[(OUT, 3), (TAP, 4)]).over(2);
        split_step(&map, 0);
    }
}
